import subprocess
import pickle
import os
import pprint

import dogqc.identifier as ident
from dogqc.attributes import Attribute 
from dogqc.codegen import Code, CodeGenerator, Variable
from dogqc.cudalang import * 
from dogqc.types import Type
from dogqc.codegen import CType

"""
def dbAccess ( schema, dbpath, csvpath, doReload=False, waitEnter=False ):
    if not os.path.exists ( dbpath ):
        os.makedirs ( dbpath )
    sizedSchema = Importer.retrieveTableSizes ( schema, csvpath )
    acc = Accessor ( dbpath, sizedSchema )
    if acc.checkLoaded ( dbpath, sizedSchema ):
        acc.schema = Accessor.loadSchema ( dbpath )
    if not acc.checkLoaded ( dbpath, sizedSchema ) or doReload:
        if waitEnter:
            input("Press Enter to OVERWRITE database...")
        acc.initDB ()
        im = Importer ( acc, csvpath )
        im.loadDatabase ()
        acc.writeSchema ()
    return acc
"""

def dbAccess ( dbpath ):
    acc = Accessor ( dbpath, Accessor.loadSchema ( dbpath ))
    return acc


def dbBuild ( schema, dbpath, csvpath ):
    sizedSchema = Importer.retrieveTableSizes ( schema, csvpath )
    acc = Accessor ( dbpath, sizedSchema )
    acc.initDB ()
    im = Importer ( acc, csvpath )
    im.loadDatabase ()
    acc.writeSchema ()
    

class Accessor ( object ):
   
    def __init__ ( self, dbpath, schema ):
        self.dbpath = dbpath
        self.schema = schema

    def initDB ( self ):
        # clear folder
        #  - todo
        pass

    @staticmethod
    def checkLoaded ( dbpath, schema ): 
        loadedSchema = Accessor.loadSchema ( dbpath )
        if loadedSchema is None:
            return False
        else:
            for tableName, tableDict in loadedSchema.items():
                if tableName == "dateformat":
                    continue
                del tableDict["charSizes"]
            return loadedSchema == schema

    @staticmethod
    def loadSchema ( dbpath ): 
        spath = dbpath + "/" + 'dbschema.pickle'
        if not os.path.exists ( spath ):
            return None
        with open( spath, 'rb') as handle:
            schema = pickle.load(handle)
        return schema

    def writeSchema ( self ):
        with open( self.dbpath + "/" + 'dbschema.pickle', 'wb') as handle:
            pickle.dump(self.schema, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def getCodeAccessDatabase ( self, codegen, attributes ):
        code = Code()
        for a in attributes:
            cols = a.inputVariables ( codegen )
            colsFile = a.variables ( codegen )
            for c, f in zip ( cols, colsFile ):
                c.declarePointer( code )
                emit ( assign ( c.get(), mmapFile ( c.dataType,  self.file ( f ) ) ), code )
        return code

    def isHostVector ( self ):
        return False

    def getNumBytes ( self, column ):
        with open ( self.file ( column ), "rb" ) as f:
            sizebin = f.read(8)
            sizenum = int.from_bytes ( sizebin, byteorder='little' )
        return sizenum
    
    def file ( self, var ):
        return self.dbpath + '/' + var.get()




class CSVCodegen ( object ):

    i = 0
    
    def getInit ( self, filename, numColumns ): 
        self.varname = "reader" + str(CSVCodegen.i)
        initCode = "io::CSVReader <" + str(numColumns) + ", io::trim_chars<' '>, io::no_quote_escape<'|'> > "
        initCode += self.varname + "(\"" + filename + "\");"
        CSVCodegen.i += 1
        return initCode

    def __init__ ( self, filename, numColumns, code ):
        self.code = code
        code.add ( self.getInit ( filename, numColumns ) )
        self.code = code
        self.numColumns = numColumns
        self.filename = filename
  
    def reset ( self ):
        self.code.add ( self.getInit (self.filename, self.numColumns ) )

    def getLine( self, parsevars ):
        return self.varname + ".read_row(" + ",".join( parsevars ) + ")"



class Importer ( object ):

    def __init__ ( self, accessor, csvpath ):
        self.csvpath = csvpath + "/"
        self.acc = accessor
        self.schema = accessor.schema

    @staticmethod
    def retrieveTableSizes ( schema, csvpath ):
        # get table sizes from csv files
        for table, attributes in schema.items():
            if table == "dateformat":
                continue
            csvFile = csvpath + '/' + table + ".tbl"
            if not os.path.isfile(csvFile):
                print ( "File: " + csvFile + " not found. Quit." )
                quit()
            cmd = [ 'wc', '-l', csvFile ]
            output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
            numRows = int ( output.decode('utf8').split(" ")[0] )
            print('number of rows in ', table, ':', numRows)
            schema[table]["size"] = numRows
        return schema.copy()

    def loadDatabase ( self ):
        codegen = CodeGenerator ( CType.FP32 )
        if "dateformat" in self.schema.keys():
            codegen.types.add ( self.schema["dateformat"] )
        self.codegen = codegen
        tableReaderCode = self.getCodeReadDatabase ( ) 
        codegen.read.add(tableReaderCode)
        codegen.compileCpu("{}/loadDB".format(self.acc.dbpath))
        codegen.execute()
        self.annotateCharSizes ( self.schema )
        self.buildIndexes ( self.acc.dbpath, self.schema )
        self.buildRelationships ( self.acc.dbpath, self.schema  )

    def buildIndexes ( self, dbpath, schema ):

        idxMngr = IndexManager(dbpath)

        for tname, attributes in schema.items():
            if tname == 'dateformat': continue
            #print(attributes)
            indexes = attributes["indexes"]
            index_sizes = {}
            
            for keys in indexes:
                idx_name = 'idx_{}_{}'.format(tname, '_'.join(keys))
                print("create_index", idx_name)
                idxMngr.create_index(idx_name, tname, schema[tname], keys)
                sizes = {}
                for key in keys:
                    cmd = 'wc -c {}/{}_{}_dir_offset'.format(dbpath, idx_name, key)
                    output = subprocess.check_output(cmd, shell=True)
                    sizes[key] = int((int(output.decode('utf8').split(' ')[0]) - 9) / 8)
                    print("\t",key,sizes[key])
                index_sizes[idx_name] = sizes
            attributes["index_sizes"] = index_sizes
        
        #pp.pprint(schema)


    def buildRelationships ( self, dbpath, schema ):

        idxMngr = RelationshipManager(dbpath)

        for f_tname, attributes in schema.items():
            if f_tname == 'dateformat': continue
            #print(attributes)
            indexes = attributes["relationships"]
            for p_tname, keys in indexes:
                f_keys, p_keys = keys[0], keys[1]
                rel_name = "rel_{}_{}__{}_{}".format(f_tname, '_'.join(f_keys), p_tname, '_'.join(p_keys))
                print("create_relationship", rel_name)
                idxMngr.create_relationship(dbpath, rel_name, p_tname, schema[p_tname], p_keys, f_tname, schema[f_tname], f_keys)


    def annotateCharSizes ( self, schema ):
        for tableName, tableDict in schema.items():
            if tableName == "dateformat":
                continue
            tableDict [ "charSizes" ] = dict()
            for attributeName, dataType in tableDict["attributes"].items():
                if dataType == Type.STRING:
                    att = Attribute ( tableDict, attributeName, dataType ) 
                    col = att.variables ( self.codegen, False )[1] # char column
                    filename = self.acc.file ( col )
                    tableDict [ "charSizes" ] [ attributeName ] = os.path.getsize ( filename ) 

    def getCodeReadDatabase ( self ):
        code = Code()
        for table, attributes in self.schema.items():
            if table == "dateformat":
                continue
            code.add ( self.getCodeReadTable ( self.schema[table] ) )
        return code.content

    def getCodeReadTable ( self, table ):
        code = Code()

        tableName = table["name"]
        tableSize = str(table["size"])
        csvFilename = self.csvpath + tableName + ".tbl"
        hasStringAttributes = False # init to False

        # csv reader with one extra column for terminal delimiter
        reader = CSVCodegen ( csvFilename, table["numColumns"] + 1, code)
        simpleParseList = []
        stringParseList = []
        csvParseArgs = []

        # create columns and variables for parsing
        for attributeName, dataType in table [ "attributes" ].items():
            parseType = self.codegen.langType ( dataType )
            if dataType == Type.DATE:
                parseType = "std::string"
            if dataType == Type.STRING:
                parseType = "std::string"
            parseVar = Variable.val ( parseType, attributeName )
            parseVar.declare ( code )
            att = Attribute ( table, attributeName, dataType ) 
            csvParseArgs.append ( parseVar.get ( ) )
            colLen = int ( tableSize )
            if ( att.attType == Type.STRING ):
                charLenVar = Variable.val ( CType.SIZE, ident.charLenVar ( att ) ) 
                charLenVar.declareAssign ( intConst(0), code )
                stringParseList.append ( ( att, parseVar, charLenVar ) )
                hasStringAttributes = True
                # add another offset for ending of the last string
                colLen += 1
            else:
                simpleParseList.append ( ( att, parseVar ) ) 
            
            col = att.variables ( self.codegen, False )[0] # get first column of each attribute
            filename = self.acc.file ( col )
            col.declarePointer ( code )
            emit ( assign ( col.get(), mmapMalloc ( col.dataType, colLen, filename ) ), code )
            if ( att.attType == Type.STRING ):
                emit ( assign ( col.arrayAccess ( intConst(0) ), intConst(0) ), code )
        
        # parser needs additional empty variable due to terminal delimiter
        nothing = Variable.val ( CType.CHAR, ident.nothingVar ( tableName ) )
        nothing.declarePointer( code ) 
        csvParseArgs.append(nothing.get())
         
        # loop over csv 
        with CountingWhileLoop ( reader.getLine ( csvParseArgs ), code ) as loop:
            with IfClause ( larger ( loop.countVar, tableSize ), code ):
                printError ( code )

            # fill simple columns with data and count string sizes
            for att, parseVar in simpleParseList:
                col = att.variables ( self.codegen, False )[0] # get first column of each attribute
                if att.attType == Type.DATE:
                    emit ( assign ( col.arrayAccess ( loop.countVar ), "toDate (" + str(parseVar) + ".c_str())" ), code ) 
                else:
                    emit ( assign ( col.arrayAccess ( loop.countVar ), parseVar.get() ), code ) 

            for att, parseVar, charLenVar in stringParseList:
                col = att.variables ( self.codegen, False )[0] # first column with offset
                emit ( assignAdd ( charLenVar.get(), parseVar.length() ), code )
                emit ( assign ( col.arrayAccess ( add ( loop.countVar, intConst(1) ) ), charLenVar.get() ), code )

        if hasStringAttributes:        
            # size of string columns is now known, allocate
            for att, parseVar, charLenVar in stringParseList:
                col = att.variables ( self.codegen, False )[1] # get second column (currently relevant for strings)
                filename = self.acc.file ( col )
                col.declarePointer ( code )
                emit ( assign ( col.get ( ), mmapMalloc ( col.dataType, charLenVar.get(), filename ) ), code )

            # reset csv reader
            reader.reset( )
            # loop over csv second time to fill strings
            with CountingWhileLoop ( reader.getLine ( csvParseArgs ), code ) as loop:
                for att, parseVar, charLenVar in stringParseList:
                    col = att.variables ( self.codegen, False )
                    emit ( strcpy ( addressof ( col[1].arrayAccess ( col[0].arrayAccess ( loop.countVar ) ) ), parseVar.cstr() ), code )
        
        return code.content     


class IndexManager:

    def __init__(self, db_dir):
        self.db_dir = db_dir
        pass

    def gen_comparator_code(self, tname, table, key_attr_names, compare_id=''):
        c = []
        c.append('bool compare{}(int a, int b)'.format(compare_id) + ' {')

        equals = ''

        for idx, attr_name in enumerate(key_attr_names[0:], 0):
            if idx == 0:
                c.append('\tif ({}[a] < {}[b])'
                    .format(attr_name, attr_name) + '{ return true; }')
            else:
                c.append('\telse if ({}{}[a] < {}[b]) '
                    .format(equals, attr_name, attr_name)
                    + '{ return true; }')
            equals += '{}[a] == {}[b] && '.format(attr_name, attr_name)
        c.append('\treturn false;')
        c.append('}')
        return c

    def create_index(self, index_name, tname, table, attr_names):

        print("create index", index_name, tname, attr_names)

        c = ['#include <bits/stdc++.h>',
            '#include "dogqc/include/mappedmalloc.h"',
            '#include <stdio.h>',
            '#include <string.h>',
            '#include <fcntl.h>']

        for aname in attr_names:
            c.append('int* {};'.format(aname))

        c += self.gen_comparator_code(tname, table, attr_names)
        
        mc = []
        
        for idx, aname in enumerate(attr_names):
            cfname = '{}/{}_{}'.format(self.db_dir, tname, aname)
            mc.append('{} = (int*) map_memory_file("{}");'
                .format(aname, cfname))

        mc.append('int* positions = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position", O_TRUNC);'
            .format(table['size'], self.db_dir, index_name))
        mc.append('for (int i = 0; i < {}; ++i) positions[i] = i;'.format(table['size']))
        mc.append('std::sort(positions, positions + {}, compare);'.format(table['size']))
        

        for idx, aname in enumerate(attr_names):
            mc.append('int size_{} = 0;'.format(aname))
        mc.append('{')
        for idx, aname in enumerate(attr_names):
            mc.append('int v_{} = -1;'.format(aname))
        mc.append('\tfor (int i = 0; i < {}; ++i) '.format(table['size']) + '{')
        mc.append('\t\tbool next = false;')
        for idx, aname in enumerate(attr_names):
            mc.append('\t\tint nv_{} = {}[positions[i]];'.format(aname, aname))
            mc.append('\t\tnext = next || (v_{} != nv_{});'.format(aname, aname))
            mc.append('\t\tif (next) {')
            mc.append('\t\t\tsize_{}++;'.format(aname))
            mc.append('\t\t\tv_{} = nv_{};'.format(aname, aname))
            mc.append('\t\t}')
        mc.append('\t}')
        mc.append('}')

        mc.append('int* {}_offset = (int*) malloc_memory_mapped_file(sizeof(int) * 2, "{}/{}_offset", O_TRUNC);'
            .format(index_name, self.db_dir, index_name))
        mc.append('{}_offset[0] = 0;'.format(index_name))
        mc.append('{}_offset[1] = size_{};'.format(index_name, attr_names[0]))

        for idx, aname in enumerate(attr_names):
            mc.append('int* {}_dir_offset = (int*) malloc_memory_mapped_file(sizeof(int) * 2 * size_{}, "{}/{}_{}_dir_offset", O_TRUNC);'
                .format(aname, aname, self.db_dir, index_name, aname))
            mc.append('int* {}_offset = (int*) malloc_memory_mapped_file(sizeof(int) * 2 * size_{}, "{}/{}_{}_offset", O_TRUNC);'
                .format(aname, aname, self.db_dir, index_name, aname))
            mc.append('int* {}_val = (int*) malloc_memory_mapped_file(sizeof(int) * size_{}, "{}/{}_{}_val", O_TRUNC);'
                .format(aname, aname, self.db_dir, index_name, aname))

        mc.append('{')
        for idx, aname in enumerate(attr_names):
            mc.append('\tint v_{} = -1;'.format(aname))
            mc.append('\tint idx_{} = -1;'.format(aname))
        
        mc.append('\tfor (int i = 0; i < {}; ++i) '.format(table['size']) + '{')
        mc.append('\t\tbool next = false;')
        for idx, aname in enumerate(attr_names):
            mc.append('\t\tint nv_{} = {}[positions[i]];'.format(aname, aname))
            mc.append('\t\tnext = next || (v_{} != nv_{});'.format(aname, aname))
            mc.append('\t\tif (next) {')
            mc.append('\t\t\tv_{} = nv_{};'.format(aname, aname))
            mc.append('\t\t\tidx_{}++;'.format(aname))
            mc.append('\t\t\t{}_val[idx_{}] = v_{};'.format(aname, aname, aname))
            mc.append('\t\t\t{}_dir_offset[idx_{} * 2] = i;'.format(aname, aname))
            mc.append('\t\t}')
            mc.append('\t\t{}_dir_offset[idx_{} * 2 + 1] = i + 1;'.format(aname, aname))
        mc.append('\t}')
        mc.append('}')

        for idx, aname in enumerate(attr_names[1:], 1):
            mc.append('{')
            mc.append('\tint idx = -1;')
            for j in range(idx):
                mc.append('\tint v_{} = -1;'.format(attr_names[j]))
                
            mc.append('\tfor (int i = 0; i < size_{}; ++i)'.format(aname) + '{')
            mc.append('\t\tint pos = positions[{}_dir_offset[2*i]];'.format(aname))
            mc.append('\t\tbool next = false;')

            for j in range(idx):
                prev_aname = attr_names[j]
                mc.append('\t\tint nv_{} = {}[pos];'.format(prev_aname, prev_aname))
                mc.append('\t\tnext = next || (nv_{} != v_{});'.format(prev_aname, prev_aname))
                mc.append('\t\tv_{} = nv_{};'.format(prev_aname, prev_aname))
            
            mc.append('\t\tif (next) {')
            mc.append('\t\t\tidx++;')
            mc.append('\t\t\t{}_offset[idx*2] = i;'.format(attr_names[idx-1]))
            mc.append('\t\t}')
            mc.append('\t\t{}_offset[idx*2+1] = i + 1;'.format(attr_names[idx-1]))
            mc.append('\t}')
            mc.append('\tassert(idx + 1 == size_{});'.format(attr_names[idx-1]))
            mc.append('}')            


        c.append('int main() {')

        c += list(map(lambda x: '\t'+x, mc))
        c.append('}')


        code = '\n'.join(c)

        f = open('{}/{}_build.cpp'.format(self.db_dir, index_name),'w')
        f.write(code)
        f.close()
        #print(code)

        cmd = "g++ {}/{}_build.cpp -o {}/build.out -g -pthread -I{}/..".format(self.db_dir, index_name, self.db_dir, os.path.dirname(os.path.abspath(__file__)))
        print(cmd)
        output = subprocess.check_output(cmd, shell=True)
        

        cmd = "{}/build.out".format(self.db_dir, self.db_dir, index_name)
        output = subprocess.check_output(cmd, shell=True)

        return c


        


class RelationshipManager:

    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.ctype_map = {
            'date': 'unsigned',
            'int': 'int',
            'unsigned long long': 'unsigned long long',
            'float': 'float',
            'double': 'double',
            'char': 'char',
            'boolean': 'bool'
        }

    def gen_comparator_code(self, tname, table, key_attr_names, compare_id=''):
        c = []
        c.append('bool compare{}(int a, int b)'.format(compare_id) + ' {')
        for idx, attr_name in enumerate(key_attr_names[0:], 0):
            if idx == 0:
                c.append('\tif ({}_{}[a] < {}_{}[b])'
                    .format(tname, attr_name, tname, attr_name) + '{ return true; }')
            else:
                c.append('\telse if ({}_{}[a] == {}_{}[b] && {}_{}[a] < {}_{}[b]) '
                    .format(tname, key_attr_names[idx-1], tname, key_attr_names[idx-1], tname, attr_name, tname, attr_name)
                    + '{ return true; }')
        c.append('\treturn false;')
        c.append('}')
        return c

    def gen_eqcomparator_code(self, tname, table, key_attr_names, compare_id=''):
        c = []

        c.append('bool compare{}(int a, int b)'.format(compare_id) + ' {')
        for idx, attr_name in enumerate(key_attr_names[0:], 0):
            if idx == 0:
                if len(key_attr_names) > 1:
                    c.append('\tif ({}_{}[a] < {}_{}[b])'
                        .format(tname, attr_name, tname, attr_name) + '{ return true; }')
                else:
                    c.append('\tif ({}_{}[a] <= {}_{}[b])'
                        .format(tname, attr_name, tname, attr_name) + '{ return true; }')
            elif idx < len(key_attr_names) - 1:
                c.append('\telse if ({}_{}[a] == {}_{}[b] && {}_{}[a] < {}_{}[b]) '
                    .format(tname, key_attr_names[idx-1], tname, key_attr_names[idx-1], tname, attr_name, tname, attr_name)
                    + '{ return true; }')
            else:
                c.append('\telse if ({}_{}[a] == {}_{}[b] && {}_{}[a] <= {}_{}[b]) '
                    .format(tname, key_attr_names[idx-1], tname, key_attr_names[idx-1], tname, attr_name, tname, attr_name)
                    + '{ return true; }')
        c.append('\treturn false;')
        c.append('}')
        return c

    def gen_code_for_relationship(self, dbpath, rel_name, p_tname, p_table, p_attr_names, f_tname, f_table, f_attr_names):

        c = []
        for idx, attr_name in enumerate(p_attr_names):
            cfname = '{}/{}_{}'.format(self.db_dir, p_tname, attr_name)
            c.append('{}_{} = (int*) map_memory_file("{}");'
                .format(p_tname, attr_name, cfname))
        
        for idx, attr_name in enumerate(f_attr_names):
            cfname = '{}/{}_{}'.format(self.db_dir, f_tname, attr_name)
            c.append('{}_{} = (int*) map_memory_file("{}");'
                .format(f_tname, attr_name, cfname))

        c.append('int* positions_p = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position", O_TRUNC);'
            .format(p_table['size'], self.db_dir, rel_name))
        c.append('int* positions_p_rev = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_rev", O_TRUNC);'
            .format(p_table['size'], self.db_dir, rel_name))

        c.append('int* positions_f = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_f", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))
        c.append('int* positions_f_rev = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_f_rev", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))

        c.append('int* offsets = (int*) malloc_memory_mapped_file(sizeof(int) * {} * 2, "{}/{}_offset", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))

        c.append("assert(positions_p != NULL);")
        
        c.append('for (int i = 0; i < {}; ++i)'.format(p_table['size']) + ' { positions_p[i] = i; }')
        c.append('std::sort(positions_p, positions_p + {}, compare_p);'.format(p_table['size']))
        c.append('for (int i = 0; i < {}; ++i)'.format(p_table['size']) + ' { positions_p_rev[positions_p[i]] = i; }')
        c.append('for (int i = 1; i < {}; ++i)'.format(p_table['size']) + ' { assert(compare_p_eq(positions_p[i-1], positions_p[i])); }')

        c.append('for (int i = 0; i < {}; ++i)'.format(f_table['size']) + ' { positions_f[i] = i; }')
        c.append('std::sort(positions_f, positions_f + {}, compare_f);'.format(f_table['size']))
        c.append('for (int i = 0; i < {}; ++i)'.format(f_table['size']) + ' { positions_f_rev[positions_f[i]] = i; }')
        c.append('for (int i = 1; i < {}; ++i)'.format(f_table['size']) + ' { assert(compare_f_eq(positions_f[i-1], positions_f[i])); }')
        
        c.append('for (int f = 0; f < {} * 2; ++f)'.format(f_table['size']) + '{ offsets[f] = -1; }')

        c.append('int p = 0;')
        c.append('for (int f_rev = 0; f_rev < {}; ++f_rev)'.format(f_table['size']) + ' {')

        c.append('\tint f = positions_f[f_rev];')

        cc = []
        for i, f_attr_name in enumerate(f_attr_names):
            cc.append('{}_{}[f_prev] == {}_{}[f]'.format(
                f_tname, f_attr_name, f_tname, f_attr_name))    

        c.append('\tif (f_rev > 0) {')
        c.append('\t\tint f_prev = positions_f[f_rev-1];')
        c.append('\t\tif ({})'.format(' && '.join(cc)) + '{')
        c.append('\t\toffsets[2*f] = offsets[2*f_prev];')
        c.append('\t\toffsets[2*f+1] = offsets[2*f_prev+1];')
        c.append('\t\tcontinue;')
        c.append('\t\t}')
        c.append('\t}')
        

        # a0 < b0 || (a0 == b0 && (a1 < b1 || (a1 == b1 && (a2 < b2))))
        cc = ""
        for i, p_attr_name in enumerate(p_attr_names):
            idx = len(p_attr_names) - 1 - i
            f_attr_name = f_attr_names[idx]
            if cc != "": cc = " || ({})".format(cc)
            if idx > 0:
                cc = '{}_{}[positions_p[p]] == {}_{}[f] && ({}_{}[positions_p[p]] < {}_{}[f])'.format(
                        p_tname, p_attr_names[idx-1], f_tname, f_attr_names[idx-1],
                        p_tname, p_attr_names[idx], f_tname, f_attr_names[idx],
                        cc)
            else:
                cc = '{}_{}[positions_p[p]] < {}_{}[f]{}'.format(p_tname, p_attr_names[idx], f_tname, f_attr_names[idx], cc)

        c.append('\twhile (p < {} && ({}))'.format(p_table['size'], cc) + '{ ++p; }')
        cc = []
        for idx, p_attr_name in enumerate(p_attr_names):
            f_attr_name = f_attr_names[idx]
            cc.append('{}_{}[positions_p[p]] == {}_{}[f]'.format(p_tname, p_attr_name, f_tname, f_attr_name))
        #cc.reverse()

        c.append('\twhile (p < {} && ({}))'.format(p_table['size'], ' && '.join(cc)) + ' {')
        c.append('\t\tif (offsets[2*f] == -1) { offsets[2*f] = p; }')
        c.append('\t\toffsets[2*f+1] = ++p;')
        c.append('\t}')       
        c.append('}')

        c.append('free_memory_mapped_file("{}/{}_position_rev");'.format(self.db_dir, rel_name))
        c.append('free_memory_mapped_file("{}/{}_position_f");'.format(self.db_dir, rel_name))
        c.append('free_memory_mapped_file("{}/{}_position_f_rev");'.format(self.db_dir, rel_name))

        cc = []
        for idx, p_attr_name in enumerate(p_attr_names):
            f_attr_name = f_attr_names[idx]
            cc.append('{}_{}[p] == {}_{}[f]'.format(p_tname, p_attr_name, f_tname, f_attr_name))
        #cc.reverse()
        c.append('FILE* pFile = fopen("{}/{}_histogram.txt", "w");'.format(dbpath, rel_name))
        c.append('for (int f = 0; f < {}; ++f)'.format(f_table['size']) + '{')
        c.append('\tint offset = offsets[2*f]; int end = offsets[2*f+1];')
        c.append('\tfor (int i = offset; i < end; ++i) {')
        c.append('\t\tint p = positions_p[i];')
        c.append('\t\tassert({});'.format(' && '.join(cc)))
        c.append('\t}')
        c.append('\tfprintf(pFile, "%d %d\\n", f, end - offset );')
        c.append('}')
        c.append('fclose(pFile);')

        '''
        c.append('std::cout << "Primary Table ({})" << std::endl;'.format(','.join(p_attr_names)))
        c.append('for (int p = 0; p < {}; ++p)'.format(p_table['size']) + ' {')
        c.append('\tstd::cout << positions_p[p] << "\t";')
        for f_attr_name in f_attr_names:
            c.append('\tstd::cout << {}_{}[p] << ",";'.format(p_tname, p_attr_name))
        c.append('\tstd::cout << offsets[p];')
        c.append('\tstd::cout << std::endl;')       
        c.append('}')

        c.append('std::cout << "Foregin Table ({})" << std::endl;'.format(','.join(f_attr_names)))
        c.append('for (int f = 0; f < {}; ++f)'.format(f_table['size']) + ' {')
        c.append('\tstd::cout << positions_f[f] << "\t";')
        for f_attr_name in f_attr_names:
            c.append('\tstd::cout << {}_{}[f] << ",";'.format(f_tname, f_attr_name))
        c.append('\tstd::cout << std::endl;')       
        c.append('}')
        '''
        return c


    def create_relationship(self, dbpath, rel_name, p_tname, p_table, p_attr_names, f_tname, f_table, f_attr_names):

        print("create relationship {}".format(rel_name))

        c = ['#include <bits/stdc++.h>',
            '#include "dogqc/include/mappedmalloc.h"',
            '#include <stdio.h>',
            '#include <string.h>',
            '#include <fcntl.h>']

        print(p_attr_names)
        print(f_attr_names)

        for attr_name in p_attr_names:
            c.append('int* {}_{};'.format(p_tname, attr_name))
        for attr_name in f_attr_names:
            c.append('int* {}_{};'.format(f_tname, attr_name))

        c += self.gen_comparator_code(p_tname, p_table, p_attr_names, '_p')
        c += self.gen_eqcomparator_code(p_tname, p_table, p_attr_names, '_p_eq')
        c += self.gen_comparator_code(f_tname, f_table, f_attr_names, '_f')
        c += self.gen_eqcomparator_code(f_tname, f_table, f_attr_names, '_f_eq')

        mc = self.gen_code_for_relationship(dbpath, rel_name, p_tname, p_table, p_attr_names, f_tname, f_table, f_attr_names)
        c.append('int main() {')
        c += list(map(lambda x: '\t' + x, mc))
        c.append('}')

        code = '\n'.join(c)

        f = open('{}/{}_build.cpp'.format(self.db_dir, rel_name),'w')
        f.write(code)
        f.close()
        #print(code)

        cmd = "g++ {}/{}_build.cpp -o {}/build.out -g -pthread -I{}/..".format(self.db_dir, rel_name, self.db_dir, os.path.dirname(os.path.abspath(__file__)))
        print(cmd)
        output = subprocess.check_output(cmd, shell=True)
        

        cmd = "{}/build.out".format(self.db_dir, self.db_dir, rel_name)
        output = subprocess.check_output(cmd, shell=True)

    def gen_code_for_relationship_on_gpu(self, dbpath, rel_name, p_tname, p_table, p_attr_names, f_tname, f_table, f_attr_names):

        c = []
        
        suffix = 'p'
        tname = p_tname
        size = p_table['size']
        
        for idx, attr_name in enumerate(p_attr_names):
            c.append(f'int* col_{suffix}_{idx} = (int*) map_memory_file("{self.db_dir}/{tname}_{attr_name}");'
                .format(p_tname, attr_name, cfname))
            c.append(f'thrust::device_vector<int> d_col_p_{idx}(col_p_{idx},col_p_{idx}+{size});')
            c.append(f'int* d_col_p_{idx}_ptr =  thrust::raw_pointer_cast(d_col_p_{idx}.data());')

        d_position = f'd_position_{suffix}'
        d_flag = f'd_flag_{suffix}'
        d_key = f'd_key_{suffix}'

        c.append(f'thrust::device_vector<int> {d_position}({size}, 1);')
        c.append(f'thrust::exclusive_scan({d_position}.begin(),{d_position}.end(),{d_position}.begin());')
        c.append(f'thrust::device_vector<int> {d_flag}({size}, 0);')
        c.append(f'thrust::device_vector<unsigned long long> {d_key}({size});')
        c.append(f'int* {d_position_p} = thrust::raw_pointer_cast({d_position}.data());')
        c.append(f'int* {d_flag} = thrust::raw_pointer_cast(d_flag_p.data());')
        c.append(f'unsigned long long* d_key_p_ptr = thrust::raw_pointer_cast(d_key_p.data());')

        for idx, attr_name in enumerate(p_attr_names):
            c.append(f'Themis::Trie::SetKey<<<{size}/1024+1,1024>>>(d_key_p_ptr, d_flag_p_ptr, d_position_p_ptr, d_col_p_{idx}_ptr, {size});')
            c.append(f'cudaDeviceSynchronize();')
            c.append(f'thrust::sort_by_key(d_key_p.begin(), d_key_p.end(), d_position_p.begin());')
            c.append(f'Themis::Trie::SetFlag<<<{size}/1024+1,1024>>>(d_key_p_ptr, d_flag_p_ptr, {size});')
            c.append(f'cudaDeviceSynchronize();')
            c.append(f'thrust::inclusive_scan(d_flag_p.begin(), d_flag_p.end(), d_flag_p.begin());')

        

        


        c.append('int* positions_p = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position", O_TRUNC);'
            .format(p_table['size'], self.db_dir, rel_name))
        c.append('int* positions_p_rev = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_rev", O_TRUNC);'
            .format(p_table['size'], self.db_dir, rel_name))

        c.append('int* positions_f = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_f", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))
        c.append('int* positions_f_rev = (int*) malloc_memory_mapped_file(sizeof(int) * {}, "{}/{}_position_f_rev", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))

        c.append('int* offsets = (int*) malloc_memory_mapped_file(sizeof(int) * {} * 2, "{}/{}_offset", O_TRUNC);'
            .format(f_table['size'], self.db_dir, rel_name))



    def create_relationship_on_gpu(self, dbpath, rel_name, p_tname, p_table, p_attr_names, f_tname, f_table, f_attr_names):
        print("create relationship {}".format(rel_name))
        print(p_attr_names)
        print(f_attr_names)

        c = ['#include <bits/stdc++.h>',
            '#include "dogqc/include/mappedmalloc.h"',
            '#include <stdio.h>',
            '#include <string.h>',
            '#include <fcntl.h>',
            '#include <thrust/host_vector.h>',
            '#include <thrust/device_vector.h>',
            '#include <thrust/generate.h>',
            '#include <thrust/sort.h>',
            '#include <thrust/scan.h>',
            '#include <thrust/copy.h>',
            '#include <algorithm>',
            '#include <cstdlib>',
            '#include <time.h>',
            ]

        








        

        

        
        
        
        