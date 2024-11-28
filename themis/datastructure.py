from dogqc.types import Type
from dogqc.cudalang import CType
from dogqc.kernel import KernelCall
from dogqc.relationalAlgebra import Reduction

from themis.code import Code

def langType ( relDataType ):
    internalTypeMap = {}
    internalTypeMap [ Type.INT ] = CType.INT
    internalTypeMap [ Type.ULL ] = CType.ULL
    internalTypeMap [ Type.DATE ] = CType.UINT
    internalTypeMap [ Type.CHAR ] = CType.CHAR
    internalTypeMap [ Type.FLOAT ] = CType.FP32
    internalTypeMap [ Type.DOUBLE ] = CType.FP32
    internalTypeMap [ Type.STRING ] = CType.STR_TYPE
    internalTypeMap [ Type.PTR_INT ] = CType.PTR_INT
    return internalTypeMap [ relDataType ]


class Datastructure:
    
    def __init__(self):
        pass
    
    def __str__(self):
        return self.name
    
    def declarePayload(self, opId, attrs):
        c = Code()
        c.add(f'struct Payload{opId}' + '{')
        for attrId, attr in attrs.items():
            dType = langType(attr.dataType)
            c.add(f'{dType} {attr.id_name};')
        c.add('};')
        return c

    def cuMalloc(self, name, dtype, size):
        #size = int((size-1)/128) * 128 + 128
        c = Code()
        c.add(f'{dtype}* {name};')
        c.add(f'cudaMalloc((void**) &{name}, {size} * sizeof({dtype}));')
        return c

    def mmap(self, name, dtype, size):
        c = Code()
        dbpath = KernelCall.args.dbpath
        c.add(f'{dtype}* mmap_{name} = ({dtype}*) map_memory_file("{dbpath}/{name}");')
        c.add(self.cuMalloc(name, dtype, size))
        c.add(f'cudaMemcpy({name}, mmap_{name}, {size} * sizeof({dtype}), cudaMemcpyHostToDevice);')
        return c
    
    def initArray(self, name, val, size):
        gSize, bSize = KernelCall.defaultGridSize,KernelCall.defaultBlockSize
        c = Code()
        c.add(f'initArray<<<{gSize}, {bSize}>>>({name},{val},{size});')
        return c

    def genDeclarationInGlobal(self):
        return Code()
    
    def genDeclarationInMain(self):
        return Code()
    
    def genCodeBeforeExecution(self):
        return Code()
    
    def genCodeAfterExecution(self):
        return Code()

    def genArgsForKernelDeclaration(self):
        return Code()
    
    def genArgsForKernelCall(self):
        return Code()


class BaseColumn(Datastructure):
    
    def __init__(self, tableName, attr):
        self.tableName = tableName
        self.name = f"{tableName}_{attr.name}"
        self.attr = attr
        #print(self.tableName, self.attr.id_name)
        
    def genDeclarationInMain(self):
        c = Code()
        schema, attr, name = KernelCall.args.schema[self.tableName], self.attr, self.name
        #print(schema, self.tableName, self.attr.id_name)
        if attr.dataType == Type.STRING:
            c.add(self.mmap(name + '_offset', CType.SIZE, schema['size']+1))
            c.add(self.mmap(name + '_char', CType.CHAR, schema['charSizes'][attr.name]))
        else:
            c.add(self.mmap(name, langType(attr.dataType), schema['size']))
        return c
    
    def genArgsForKernelDeclaration(self):
        if self.attr.dataType == Type.STRING:
            return f'size_t* {self.name}_offset, char* {self.name}_char,'
        else:
            dType = langType(self.attr.dataType)
            return f'{dType}* {self.name},'
    
    def genArgsForKernelCall(self):
        if self.attr.dataType == Type.STRING:
            return f'{self.name}_offset, {self.name}_char,'
        else:
            return f'{self.name},'


class TempTable(Datastructure):
    
    def __init__(self, name, size, attrs):
        self.name = name
        self.attrs = attrs
        self.size = size

    def genDeclarationInMain(self):
        c = Code()
        c.add(self.cuMalloc(f'nout_{self.name}', CType.INT, 1))
        c.add(self.initArray(f'nout_{self.name}', CType.zeroValue[CType.INT], 1))
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(self.cuMalloc(f'{self.name}_{attr.name}', dtype, self.size))
        return c

    def genArgsForKernelDeclaration(self):
        c = Code()
        c.add(f'int* nout_{self.name},')
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(f'{dtype}* {self.name}_{attr.name},')
        return c
    
    def genArgsForKernelCall(self):
        c = Code()
        c.add(f'nout_{self.name},')
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(f'{self.name}_{attr.name},')
        return c


class ResultTable(Datastructure):
    
    def __init__(self, name, size, attrs):
        self.name = name
        self.attrs = attrs
        self.size = size

    def genDeclarationInMain(self):
        c = Code()
        c.add(self.cuMalloc(f'nout_{self.name}', CType.INT, 1))
        c.add(self.initArray(f'nout_{self.name}', CType.zeroValue[CType.INT], 1))
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(self.cuMalloc(f'{self.name}_{attr.id_name}', dtype, self.size))
        return c

    def genArgsForKernelDeclaration(self):
        c = Code()
        c.add(f'int* nout_{self.name},')
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(f'{dtype}* {self.name}_{attr.id_name},')
        return c
    
    def genArgsForKernelCall(self):
        c = Code()
        c.add(f'nout_{self.name},')
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = CType.STR_TYPE
            c.add(f'{self.name}_{attr.id_name},')
        return c
    
class Index(Datastructure):
    
    def __init__(self, rel_name, from_tname, to_tname):
        self.from_tname = from_tname
        self.to_tname = to_tname
        self.name = rel_name
        
    def genDeclarationInMain(self):
        c = Code()
        schema = KernelCall.args.schema
        from_tsize = schema[self.from_tname]['size']
        to_tsize = schema[self.to_tname]['size']
        c.add(self.mmap(f'{self.name}_offset', CType.INT, 2*from_tsize))
        c.add(self.mmap(f'{self.name}_position', CType.INT, to_tsize))
        return c
        
    def genArgsForKernelDeclaration(self):
        return f'int* {self.name}_offset, int* {self.name}_position,'
    
    def genArgsForKernelCall(self):
        return f'{self.name}_offset, {self.name}_position,'

class AggHT(Datastructure):
    
    def __init__(self, opId, size, keyAttrs, condAttrs, aggAttrs):
        self.opId = opId
        self.name = f'aht{opId}'
        self.size = size
        self.keyAttrs = keyAttrs
        self.condAttrs = condAttrs
        self.aggAttrs = aggAttrs
        self.hasCount = False
        self.hasAvg = False
        self.countAttr = None
        for attrId, ( attr, inputIdentifier, reductionType ) in aggAttrs.items():
            if reductionType == Reduction.COUNT:
                self.hasCount = True
                self.countAttr = attr
            self.hasAvg = self.hasAvg or reductionType == Reduction.AVG

    def genDeclarationInGlobal(self):
        attrs = {}
        attrs.update(self.keyAttrs)
        attrs.update(self.condAttrs)
        return self.declarePayload(self.opId, attrs)

    def genDeclarationInMain(self):
        gSize = KernelCall.defaultGridSize
        bSize = KernelCall.defaultBlockSize
        c = Code()
        c.add(self.cuMalloc(self.name, f'agg_ht<Payload{self.opId}>', self.size))
        c.add(f'initAggHT<<<{gSize},{bSize}>>>({self.name},{self.size});')
        
        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            dtype = langType(attr.dataType)
            c.add(self.cuMalloc(f'{self.name}_{attr.id_name}', dtype, self.size))

        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            dtype = langType(attr.dataType)
            if reductionType == Reduction.SUM or reductionType == Reduction.COUNT or reductionType == Reduction.AVG:
                val = CType.zeroValue[dtype]
            elif reductionType == Reduction.MAX:
                val = CType.minValue[dtype]
            elif reductionType == Reduction.MIN:
                val = CType.maxValue[dtype]
            c.add(self.initArray(f'{self.name}_{attr.id_name}', val, self.size))
            
        return c

    def genArgsForKernelDeclaration(self):
        c = Code()
        c.add(f'agg_ht<Payload{self.opId}>* aht{self.opId},')
        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            dtype = langType(attr.dataType)
            c.add(f'{dtype}* {self.name}_{attr.id_name},')
        return c
    
    def genArgsForKernelCall(self):
        c = Code()
        c.add(f'aht{self.opId},')
        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            c.add(f'{self.name}_{attr.id_name},')
        return c

class MultiHT(Datastructure):
    
    def __init__(self, opId, size, psize, attrs):
        self.name = f'jht{opId}'
        self.size = size
        self.psize = psize
        self.attrs = attrs
        self.opId = opId

    def genDeclarationInGlobal(self):
        return self.declarePayload(self.opId, self.attrs)
    
    def genDeclarationInMain(self):
        c = Code()
        c.add(self.cuMalloc(self.name, 'multi_ht', self.size))
        c.add(self.cuMalloc(f'{self.name}_offset', CType.INT, 1))
        c.add(self.cuMalloc(f'{self.name}_payload', f'Payload{self.opId}', self.psize))

        gSize = KernelCall.defaultGridSize
        bSize = KernelCall.defaultBlockSize
        c.add(f'initMultiHT<<<{gSize},{bSize}>>>({self.name},{self.size});')
        c.add(self.initArray(f'{self.name}_offset', CType.zeroValue[CType.INT], 1))
        #c.add(self.initArray(f'{self.name}_payload', f'Payload{self.opId}', self.psize))
        return c

    def genArgsForKernelDeclaration(self):
        return f'multi_ht* {self.name}, int* {self.name}_offset, Payload{self.opId}* {self.name}_payload,'
    
    def genArgsForKernelCall(self):
        return f'{self.name}, {self.name}_offset, {self.name}_payload,'       

class UniqueHT(Datastructure):
    
    def __init__(self, opId, size, attrs):
        self.name = f'ujht{opId}'
        self.size = size
        self.attrs = attrs
        self.opId = opId
    
    def genDeclarationInGlobal(self):
        return self.declarePayload(self.opId, self.attrs)
    
    def genDeclarationInMain(self):
        c = Code()
        gSize, bSize = KernelCall.defaultGridSize, KernelCall.defaultBlockSize
        c.add(self.cuMalloc(self.name, f'unique_ht<Payload{self.opId}>', self.size))
        c.add(f'initUniqueHT<<<{gSize},{bSize}>>>({self.name},{self.size});')
        return c
        
    def genArgsForKernelDeclaration(self):
        return f'unique_ht<Payload{self.opId}>* {self.name},'
    
    def genArgsForKernelCall(self):
        return f'{self.name},'
    

class Datastructures:

    def __init__(self):
        self.components = {}
        
    def add(self, ds):
        self.components[ds.name] = ds
        
    def update(self, dss):
        self.components.update(dss.components)
        
    def keys(self):
        return self.components.keys()

    def genDeclarationInGlobal(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genDeclarationInGlobal()
            code.add(sub_code)
        return code

    def genDeclarationInMain(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genDeclarationInMain()
            code.add(sub_code)
        return code

    def genCodeBeforeExecution(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genCodeBeforeExecution()
            code.add(sub_code)
        return code
    
    def genCodeAfterExecution(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genCodeAfterExecution()
            code.add(sub_code)
        return code

    def genArgsForKernelDeclaration(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genArgsForKernelDeclaration()
            code.add(sub_code)
        return code
    
    def genArgsForKernelCall(self):
        code = Code()
        for cid, c in self.components.items():
            sub_code = c.genArgsForKernelCall()
            code.add(sub_code)
        return code