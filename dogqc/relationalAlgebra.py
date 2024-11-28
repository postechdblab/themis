import sys
from enum import Enum
from collections import OrderedDict
from dogqc.scalarAlgebra import ScalarExpr
from dogqc.scalarAlgebra import EqualsExpr
from dogqc.scalarAlgebra import AndExpr
from dogqc.scalarAlgebra import AttrExpr
from dogqc.types import Type
from dogqc.util import listWrap, intToHuman, formatDOTStr
from dogqc.kernel import KernelCall

class Table ( object ):

    def __init__(self, tableType):
        self.id = id
        self.type = tableType
        self.attributes = OrderedDict()
        self.keys = []



class Attribute ( object ):

    def __init__ ( self, name, id, opId, table=None, tableName=None, size=None ):
        self.name = name
        self.id = id
        self.creation = opId
        self.numElements = None
        self.lastUse = 0
        self.dataType = None
        self.identifiers = [ name ]
        if tableName != None:
            self.identifiers.append(tableName + '.' + name)
        self.tableName = tableName
        self.table = table
        self.id_name = 'attr{}_{}'.format(self.id, self.name)

    def printSelf(self,ctxt):
        print(self.name,self.id,self.creation,self.numElements,self.lastUse,self.dataType,self.identifiers)

    def toDic(self):
        return {
            "name": self.name,
            "id": self.id,
            "creation": self.creation,
            "numElements": self.numElements,
            "lastUse": self.lastUse,
            "dataType": self.dataType,
            "identifiers": self.identifiers,
            "table": self.tableName
        }

    def __str__(self):
        return "{}, id - {}, creation - {}, numElements - {}, lastUse - {}, dataType - {} {}".format(self.name,self.id,self.creation,self.numElements,self.lastUse,self.dataType,self.identifiers)

    def toString(self):
        return "{}, id - {}, creation - {}, numElements - {}, lastUse - {}, dataType - {} {}".format(self.name,self.id,self.creation,self.numElements,self.lastUse,self.dataType,self.identifiers)

class Context ( object ):

    def __init__ ( self, database ):
        self.database = database
        self.schema = dict ( database.schema )
        
        self.opIdNum = 0
        self.attributeIdNum = 0
        self.tableVersions = dict()
        self.tables = dict()

    def opId ( self ): 
        self.opIdNum += 1
        return self.opIdNum

    def createTable ( self, identifier ):
        table = dict ()
        table["name"] = identifier
        table["attributes"] = OrderedDict()
        self.schema [ identifier ] = table
        return table

    def resolveTable ( self, identifier ):
        #print(self.schema.keys())
        table = self.schema [ identifier ]
        id = 0
        if identifier in self.tableVersions:
            id = self.tableVersions [ identifier ]
        id += 1
        self.tableVersions [ identifier ] = id
        return ( id, table )



    def createAttribute ( self, attributeName, opId, table=None, tableName=None ):
        self.attributeIdNum += 1
        id = self.attributeIdNum
        attr = Attribute ( attributeName, id, opId, table=table, tableName=tableName ) 
        return ( id, attr )
   

class RelationalAlgebra ( object ):

    def __init__ ( self, acc ):
        self.ctxt = Context ( acc )
        # keep track of sets in list
        self.selectionNodes = []
        self.joinNodes = []
        self.semiJoinNodes = []
        self.antiJoinNodes = []

    def scan ( self, table, alias=None ):         
        return Scan ( self.ctxt, table, alias )

    def indexjoin ( self, condition, ftable, fkeys, ptable, alias, pkeys, tupleNum, unique, child, doLB=False, doConvert=True):
        return IndexJoin (self.ctxt, condition, ftable, fkeys, ptable, alias, pkeys, tupleNum, unique, doLB, doConvert, child)

    def mwayindexjoin ( self, conditions, ftable, fkeys, ptable, alias, pkeys, tupleNum, child):
        return MWayIndexJoin (self.ctxt, conditions, ftable, fkeys, ptable, alias, pkeys, tupleNum, child)

    def indexscan ( self, conditions, table, alias, tupleNum, multimatch, child):
        op = IndexScan ( self.ctxt, conditions, table, alias, tupleNum, multimatch, child)
        #for idx, cond in enumerate(conditions[1:], 1):
        #    op = IndexProbe (self.ctxt, idx, conditions, table, alias, op)
        #op.tupleNum = tupleNum
        return op
    
    def exist ( self, search_keys, tupleNum, doLB, child ):
        return Exist (self.ctxt, search_keys, tupleNum, doLB, child )

    def intersection (self, vname, variables, tries, child):
        return Intersection( self.ctxt, vname, variables, tries, child )
        pass

    def triescan(self, vname, variables, tries):
        return TrieScan( self.ctxt, vname, variables, tries)
 
    def selection ( self, condition, tupleNum, doLB, child):
        node = Selection ( self.ctxt, condition, tupleNum, doLB, child)       
        self.selectionNodes.append ( node )
        return node
 
    def projection ( self, attributes, child ):
        return Projection ( self.ctxt, attributes, child )       
    
    def map ( self, attrName, expression, child ):
        return Map ( self.ctxt, attrName, expression, child )

    def multimap ( self, attrs, child ):
        return MultiMap ( self.ctxt, attrs, child )
    
    def createtemp ( self, identifier, child ):
        return Materialize.temp ( self.ctxt, identifier, child )
    
    def result ( self, child ):
        return Materialize.result ( self.ctxt, child )
 
    def join ( self, condition, tupleNum, multimatch, doLB, leftChild, rightChild ):
        return self.innerjoin ( condition, None, tupleNum, multimatch, doLB, leftChild, rightChild )
    
    def crossjoin ( self, condition, tupleNum, leftChild, rightChild ):
        # conditions type ScalarExpr
        return CrossJoin ( self.ctxt, Join.CROSS, condition, tupleNum, leftChild, rightChild )

    def nestloop ( self, condition, leftChild, rightChild ):
        return NestLoop (self.ctxt, leftChild, rightChild)

    def innerjoin ( self, equalityConditions, otherConditions, tupleNum, multimatch, doLB, leftChild, rightChild ):
        # condition type list of identifier tuples (equalities)
        node = EquiJoin ( self.ctxt, Join.INNER, equalityConditions, otherConditions, tupleNum, multimatch, doLB, leftChild, rightChild )
        self.joinNodes.append ( node )
        return node

    def mwayjoin ( self, condition, children):
        return MultiwayJoin(self.ctxt, condition, children)
    
    def semijoin ( self, equalityConditions, otherConditions, tupleNum, doLB, leftChild, rightChild ):
        node = EquiJoin ( self.ctxt, Join.SEMI, equalityConditions, otherConditions, tupleNum, False, doLB, leftChild, rightChild )
        #node.htSizeFactor = htSizeFactor
        self.semiJoinNodes.append ( node )
        return node

    def antijoin ( self, equalityConditions, otherConditions, tupleNum, doLB, leftChild, rightChild ):
        node = EquiJoin ( self.ctxt, Join.ANTI, equalityConditions, otherConditions, tupleNum, False, doLB, leftChild, rightChild )
        self.antiJoinNodes.append ( node )
        return node
 
    def outerjoin ( self, equalityConditions, otherConditions, tupleNum, multimatch, doLB, leftChild, rightChild ):
        return EquiJoin ( self.ctxt, Join.OUTER, equalityConditions, otherConditions, tupleNum, multimatch, doLB, leftChild, rightChild )

    def aggregation ( self, groupAttributes, aggregates, tupleNum, doLB, child ):
        print(groupAttributes, aggregates, tupleNum, child)
        return Aggregation ( self.ctxt, groupAttributes, aggregates, tupleNum, doLB, child )

    
    # visualize plan
    def showGraph ( self, plan ):
        from graphviz import Digraph
        plan = listWrap ( plan )
        graph = Digraph ()
        graph.graph_attr['rankdir'] = 'BT'
        for node in plan:
            node.toDOT ( graph )
        file = open("query.dot","w") 
        file.write ( graph.source )
        print ( graph )
        graph.view()
    
    def resolveAlgebraPlan ( self, plan, cfg ):
        plan = listWrap ( plan )
        # add query result as root
        plan [ len ( plan ) - 1 ] = self.result ( plan [ len ( plan ) - 1 ] ) 
        translationPlan = list()
        for node in plan:
            node.resolve ( self.ctxt )
            attr = node.prune ()
            num = node.configure ( cfg, self.ctxt )
        return plan

    def translateToCompilerPlan ( self, plan, translator ):
        translationPlan = list()
        for node in plan:
            translationPlan.append ( node.translate ( translator, self.ctxt ) )
        return translationPlan



class AlgExpr ( object ):
   
    # create attributes
    def __init__ ( self, ctxt ):
        self.opId = ctxt.opId ()
        self.inRelation = dict()
        self.outRelation = dict()
        self.ctxt = ctxt
        self.parent = None
        self.cid = None
 
    # resolve attributes and determine inRelation and outRelation
    def resolve ( self, ctxt ):
        pass
    
    # prune outRelation
    def prune ( self ):
        pass
    
    # set tupleNum
    def configure ( self, cfg, ctxt ):
        pass
    
    # translates relational operator to language
    def translate ( self, translator, ctxt ):
        pass

    def touchAttribute ( self, id, attributes = None ):
        #print(attributes)
        if attributes == None: attributes = self.inRelation
        attributes[ id ].lastUse = self.opId

    def resolveAttribute ( self, identifier, attributes=None ):
        #print(attributes)
        if attributes == None: attributes = self.inRelation

        #print(list(map(lambda x: x.name, self.inRelation.values())))

        idents, ambigs = self.buildIdentifiers ( attributes )
        if identifier in ambigs:
            raise SyntaxError ( "Identifier " + identifier + " is ambiguous in operator " +
                self.DOTcaption() + " (" + str ( self.opId ) + ")" )
        if identifier not in idents:
            print(idents)
            raise NameError ( "Attribute " + identifier + " not found in " + 
                self.DOTcaption() + " operator (" + str ( self.opId ) + ")" )
        id = idents [ identifier ]
        self.touchAttribute ( id, attributes )
        return attributes [ id ]

    def buildIdentifiers ( self, attributes ):
        identifiers = dict()
        ambiguousIdentifiers = dict()
        for id, att in attributes.items():
            for ident in att.identifiers:
                if ident in identifiers:
                    ambiguousIdentifiers [ ident ] = True
                else:
                    identifiers [ ident ] = id
        return ( identifiers, ambiguousIdentifiers )
    
    

    def pruneOutRelation ( self ):
        prunedAtts = dict()
        for id, attr in self.outRelation.items():
            if attr.lastUse > self.opId:
                prunedAtts[id] = attr
        self.outRelation = prunedAtts

    # ---- plan visualization ---- 
    def toDOT ( self, graph ):
        cap = self.DOTcaption() + " (" + str ( self.opId ) + ")"
        sub = ""
        if len ( self.outRelation ) > 0:
            sub = self.DOTsubcaption() 
        dstr = formatDOTStr ( cap, sub )
        graph.node ( str(self.opId), dstr )
        #self.DOTscalarExpr( graph )

    def DOTcaption ( self ):
        return self.__class__.__name__
    
    def DOTsubcaption ( self ):
        return ""
    
    def DOTscalarExpr ( self, graph ):
        pass
    
    def edgeDOTstr ( self ):
        if len ( self.outRelation ) == 0:
            return ""
        labelList = []
        #labelList = list ( map ( lambda x: x.name + " (" + str(x.id) + ")", self.outRelation.values() ) )
        labelList.append ( """<FONT POINT-SIZE="10"><b>""" + intToHuman ( self.tupleNum ) + "</b></FONT>" )
        res = formatDOTStr ( None, labelList )
        return res
    

class LiteralAlgExpr ( AlgExpr ):
    
    def __init__ ( self, ctxt ):
        AlgExpr.__init__ ( self, ctxt )
        self.opType = "literal"

    def prune ( self ):
        self.pruneOutRelation()
    
    def translate ( self, translator, ctxt ):
        return translator.translate ( self )

    def printSelf(self,ctxt):
        print("Literal")


class UnaryAlgExpr ( AlgExpr ):
    
    def __init__ ( self, ctxt, child ):
        AlgExpr.__init__ ( self, ctxt )
        self.child = child
        child.parent = self
        child.cid = 0
    
    def prune ( self ):
        self.child.prune()
        self.pruneOutRelation()
    
    def translate ( self, translator, ctxt ):
        inObj = self.child.translate ( translator, ctxt )
        return translator.translate ( self, inObj )
    
    def toDOT ( self, graph ):
        self.child.toDOT ( graph )
        AlgExpr.toDOT ( self, graph )
        graph.edge ( str ( self.child.opId ), str ( self.opId ), self.child.edgeDOTstr() )

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        #print("Unary")


class BinaryAlgExpr ( AlgExpr ):
    
    def __init__ ( self, ctxt, leftChild, rightChild ):
        AlgExpr.__init__ ( self, ctxt )
        self.leftChild = leftChild
        self.rightChild = rightChild

        leftChild.parent = self
        rightChild.parent = self
        leftChild.cid = 1
        rightChild.cid = 2
    
    def prune ( self ):
        self.leftChild.prune()
        self.rightChild.prune()
        self.pruneOutRelation()
    
    def translate ( self, translator, ctxt ):
        inObjLeft = self.leftChild.translate ( translator, ctxt )
        inObjRight = self.rightChild.translate ( translator, ctxt )
        return translator.translate ( self, inObjLeft, inObjRight )
    
    def toDOT ( self, graph ):
        self.leftChild.toDOT ( graph )
        self.rightChild.toDOT ( graph )
        AlgExpr.toDOT ( self, graph )
        graph.edge ( str ( self.leftChild.opId ), str ( self.opId ), self.leftChild.edgeDOTstr() )
        graph.edge ( str ( self.rightChild.opId ), str ( self.opId ), self.rightChild.edgeDOTstr() )

    def printSelf(self,ctxt):
        self.leftChild.printSelf(ctxt)
        self.rightChild.printSelf(ctxt)
        #print("Binary")


class NaryAlgExpr (AlgExpr):

    def __init__( self, ctxt, children ):
        AlgExpr.__init__( self, ctxt )
        self.children = children

    def prune ( self ):
        for child in self.children:
            child.prune()
        self.pruneOutRelation()

    def translate( self, translator, ctxt ):

        return None

    def toDOT ( self, graph ):
        pass

    def printSelf(self, ctxt):
        for child in self.children(ctxt):
            child.printSelf(ctxt)




class IndexScan (UnaryAlgExpr):

    def __init__( self, ctxt, conditions, tableName, alias, tupleNum, multimatch, child):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.scanAttributes = dict()
        self.scanTableId, self.table = ctxt.resolveTable ( tableName )
        self.tableName = tableName
        self.tableAlias = alias
        self.opType = "indexscan"
        self.multimatch = multimatch
        self.conditions = conditions
        self.tupleNum = tupleNum


    def resolve ( self, ctxt ):

        inChild = self.child.resolve ( ctxt )
        self.inRelation.update ( inChild )

        schema = self.ctxt.schema[self.tableName]
        indexes = schema['indexes']
        index_sizes = schema['index_sizes']

        id, attr = ctxt.createAttribute ( 'tid', self.opId )
        self.tid = attr
        self.tid.dataType = Type.INT
        #attr.identifiers.append (self.table["name"] + ".tid" )
        if self.tableAlias != None:
            attr.identifiers.append( self.tableAlias + ".tid" )

        for (attrName, type) in self.table [ "attributes" ].items():
            id, attr = ctxt.createAttribute ( attrName, self.opId, table=self.table, tableName=self.tableName )
            attr.dataType = type
            attr.numElements = self.table [ "size" ]
            #attr.identifiers.append ( self.table["name"] + "." + attrName )
            if self.tableAlias != None:
                attr.identifiers.append ( self.tableAlias + "." + attrName )
            self.scanAttributes [ id ] = attr
            print(attrName)

        self.inRelation.update (self.scanAttributes)
        print(list(map(lambda x: x.name, self.inRelation.values())))

        self.buildKeyAttributes = OrderedDict()
        self.probeKeyAttributes = OrderedDict()
        #print(self.inRelation)
        buildKeys = []
        for ( ident1, ident2 ) in self.conditions:
            print(ident1, ident2)
            att1 = self.resolveAttribute ( ident1 )
            att2 = self.resolveAttribute ( ident2 )
        
            if att1.id in self.scanAttributes and att2.id in inChild:
                self.buildKeyAttributes[att1.id] = att1
                self.probeKeyAttributes[att2.id] = att2
                buildKeys.append(att1.name)
            elif att1.id in inChild and att2.id in self.scanAttributes:
                self.probeKeyAttributes[att1.id] = att1
                self.buildKeyAttributes[att2.id] = att2
                buildKeys.append(att2.name)
            else:
                assert(False)

        b = set([])
        for i, bkey in enumerate(buildKeys):
            for j, keys in enumerate(indexes):
                if j in b or i >= len(keys) or keys[i] != bkey:
                    b.add(j)        
        idx = -1
        for i, keys in enumerate(indexes):
            if i not in b:
                idx = i
                break
        assert(idx != -1)
        self.idx_name = 'idx_{}_{}'.format(self.tableName, '_'.join(indexes[idx]))
        self.sizes = index_sizes[self.idx_name]

        self.outRelation = {}
        self.outRelation.update(inChild)
        self.outRelation.update(self.scanAttributes)
        return self.outRelation


    def configure ( self, cfg, ctxt ):
        inChildTupleNum = self.child.configure ( cfg, ctxt )

        if self.tupleNum == None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                self.selectivity = 1.0
            self.tupleNum = int(self.selectivity * inChildTupleNum)

        return self.tupleNum

    def DOTcaption ( self ):
        
        return 'index scan'

    def DOTsubcaption (self):
        return list ( map ( lambda x,y: x[1].name + "=" + y[1].name, self.buildKeyAttributes.items(), 
            self.probeKeyAttributes.items() ) )

    def printSelf(self, ctxt):
        pass

    def genPipes(self, ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)

        if self.multimatch:
            ctxt['pipelines'][-1].append([])


class IndexJoin (UnaryAlgExpr):

    def __init__ ( self, ctxt, condition, ftable, fkeys, tableName, alias, pkeys, tupleNum, unique, doLB, doConvert, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.scanAttributes = dict()
        self.scanTableId, self.table = ctxt.resolveTable ( tableName )
        self.ftable = ftable
        self.tableName = tableName
        self.tableAlias = alias
        self.opType = "indexjoin"
        self.unique = unique
        self.condition = condition
        self.tupleNum = tupleNum
        self.joinType = Join.INNER
        self.pkeys = pkeys
        self.fkeys = fkeys
        self.doShuffle = doLB
        self.doConvert = doConvert
        self.doLB = False if self.unique else True
        
        self.outputCard = '1' if self.unique else 'm'

    def resolve ( self, ctxt ):

        inChild = self.child.resolve ( ctxt )
        self.inRelation.update (inChild )
        
        print(list(map(lambda x: x.name, inChild.values())))
        
        self.conditionAttributes = self.condition.resolve ( self )

        self.condAttr = list(self.conditionAttributes.values())[0]
        self.rel_name = 'rel_{}_{}__{}_{}'.format(self.ftable, #self.condAttr.tableName, 
            '_'.join(self.fkeys), 
            self.tableName, 
            '_'.join(self.pkeys))

        id, attr = ctxt.createAttribute ( 'tid', self.opId, table=self.table, tableName=self.table['name'] )
        self.tid = attr

        attr.dataType = Type.INT
        attr.numElements = self.table ["size"]
        #attr.identifiers.append (self.table["name"] + ".tid" )
        if self.tableAlias != None:
            attr.identifiers.append( self.tableAlias + ".tid" )
        self.scanAttributes [ id ] = attr

        for (attrName, type) in self.table [ "attributes" ].items():
            id, attr = ctxt.createAttribute ( attrName, self.opId, table=self.table, tableName=self.table["name"] )
            attr.dataType = type
            attr.numElements = self.table [ "size" ]
            #attr.identifiers.append ( self.table["name"] + "." + attrName )
            if self.tableAlias != None:
                attr.identifiers.append ( self.tableAlias + "." + attrName )
            self.scanAttributes [ id ] = attr

        self.outRelation.update ( self.inRelation )
        self.outRelation.update ( self.scanAttributes )
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        if self.tupleNum is None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                a = list(self.conditionAttributes.values())[0].table['size']
                b = self.table['size']
                self.selectivity = float(b) / float(a) if a < b else 1.0 
            self.tupleNum = int ( childTupleNum * self.selectivity )
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.condition.toDOT ( graph )
        graph.edge ( str ( self.condition.exprId ) , str ( self.opId ) )
        
    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("IndexJoin {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        
    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'indexjoin'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        self.touched = 0
        ctxt['pipelines'][-1][-1].append(self)
        if self.unique == False:
            ctxt['pipelines'][-1].append([])
        return

        operator = {}
        operator['opId'] = self.opId
        operator['type'] = 'IndexJoin'
        operator['tupleNum'] = self.tupleNum
        operator['selectivity'] = self.selectivity
        operator['usingAttributes'] = list(map(lambda x: x.toDic(), self.conditionAttributes.values()))
        operator['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))
        ctxt['pipelines'][-1].append(operator)


class MWayIndexJoin (UnaryAlgExpr):

    def __init__ ( self, ctxt, conditions, ftable, fkeys, tableName, alias, pkeys, tupleNum, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.scanAttributes = dict()
        self.scanTableId, self.table = ctxt.resolveTable ( tableName )
        self.ftable = ftable
        self.tableName = tableName
        self.tableAlias = alias
        self.opType = "mwayindexjoin"
        self.conditions = conditions
        self.tupleNum = tupleNum
        self.pkeys = pkeys
        self.fkeys = fkeys
        self.doConvert = False
        self.doLB = True
        self.outputCard = 'm'

    def resolve ( self, ctxt ):

        inChild = self.child.resolve ( ctxt )
        self.inRelation.update (inChild )
        
        print(list(map(lambda x: x.name, inChild.values())))
        
        self.condAttrs = []
        for cond in self.conditions:
            self.condAttrs.append(list(cond.resolve(self).values())[0])
            
        self.rel_name = 'rel_{}_{}__{}_{}'.format(self.ftable, #self.condAttr.tableName, 
            '_'.join(self.fkeys), 
            self.tableName, 
            '_'.join(self.pkeys))

        id, attr = ctxt.createAttribute ( 'tid', self.opId, table=self.table, tableName=self.table['name'] )
        self.tid = attr

        attr.dataType = Type.INT
        attr.numElements = self.table ["size"]
        #attr.identifiers.append (self.table["name"] + ".tid" )
        if self.tableAlias != None:
            attr.identifiers.append( self.tableAlias + ".tid" )
        self.scanAttributes [ id ] = attr

        for (attrName, type) in self.table [ "attributes" ].items():
            id, attr = ctxt.createAttribute ( attrName, self.opId, table=self.table, tableName=self.table["name"] )
            attr.dataType = type
            attr.numElements = self.table [ "size" ]
            #attr.identifiers.append ( self.table["name"] + "." + attrName )
            if self.tableAlias != None:
                attr.identifiers.append ( self.tableAlias + "." + attrName )
            self.scanAttributes [ id ] = attr

        self.outRelation.update ( self.inRelation )
        self.outRelation.update ( self.scanAttributes )
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        if self.tupleNum is None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                a = self.condAttrs[0].table['size']
                b = self.table['size']
                self.selectivity = float(b) / float(a) if a < b else 1.0 
            self.tupleNum = int ( childTupleNum * self.selectivity )
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.condition.toDOT ( graph )
        graph.edge ( str ( self.condition.exprId ) , str ( self.opId ) )
        
    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("IndexJoin {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        
    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'indexjoin'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        self.touched = 0
        ctxt['pipelines'][-1][-1].append(self)
        ctxt['pipelines'][-1].append([])
        return


class Exist (UnaryAlgExpr):

    def __init__ ( self, ctxt, searchKeys, tupleNum, doLB, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.opType = "exist"
        self.searchKeys = searchKeys        
        self.tupleNum = tupleNum
        self.doLB = doLB
        self.outputCard = '1'

    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        self.searchAttributes = [ x.resolve(self) for x in self.searchKeys ]
        self.searchAttrs = [ list(x.values())[0] for x in self.searchAttributes ]
        self.rel_name = 'rel_vertex_id__edge_src'
        self.outRelation = self.inRelation
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        if self.tupleNum == None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                self.selectivity = 1.0
            self.tupleNum = int ( childTupleNum * self.selectivity )
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.condition.toDOT ( graph )
        graph.edge ( str ( self.condition.exprId ) , str ( self.opId ) )
        
    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Exist {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        
    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'exist'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        ctxt['pipelines'][-1].append([])
        return


class Scan ( LiteralAlgExpr ):

    def __init__ ( self, ctxt, tableName, alias):
        LiteralAlgExpr.__init__ ( self, ctxt )
        self.scanAttributes = dict()
        self.scanTableId, self.table = ctxt.resolveTable ( tableName )
        self.isTempScan = ( "isTempTable" in self.table )
        self.tableAlias = alias
        self.opType = "scan"
        self.tableName = tableName
    
    def resolve ( self, ctxt ):

        id, attr = ctxt.createAttribute ( 'tid', self.opId, table=self.table, tableName=self.table['name'] )
        
        self.tid = attr
        
        attr.dataType = Type.INT
        attr.numElements = self.table ["size"]
        if self.tableAlias != None:
            attr.identifiers.append( self.tableAlias + ".tid" )
        self.scanAttributes [ id ] = attr

        for (attrName, type) in self.table [ "attributes" ].items():
            id, attr = ctxt.createAttribute ( attrName, self.opId, table=self.table, tableName=self.table["name"] )
            attr.dataType = type
            attr.numElements = self.table [ "size" ]
            if self.tableAlias != None:
                attr.identifiers.append ( self.tableAlias + "." + attrName )
            if self.isTempScan:
                attr.sourceId = self.table [ "sourceAttributes" ] [ attrName ]
            self.scanAttributes [ id ] = attr

        self.outRelation = self.scanAttributes
        #print(self.opId, 'scan', list(self.scanAttributes.keys()))
        return self.outRelation
    
    def configure ( self, cfg, ctxt ):
        self.tupleNum = self.table["size"]
        return self.tupleNum
    
    def DOTcaption ( self ):
        if self.isTempScan:
            return "Tempscan"
        else:
            return "Scan"

    def DOTsubcaption ( self ):
        return self.table["name"] + " (" + str ( self.scanTableId ) + ")"

    def printSelf(self,ctxt):
        print("Scan {} tupleNum {}".format(self.opId, self.tupleNum))
        print("\t,",self.table)
        print("\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.scanAttributes.values()))))

    def genPipes(self,ctxt):
        ctxt['pipelines'][-1][-1].append(self)
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'scan'
        myCtxt['table'] = self.table['name']
        myCtxt['children'] = [ ]
        return myCtxt

class IntersectionSelection( UnaryAlgExpr ):

    def __init__(self, opId, tid, arr, arr_val, joinAttributes, tupleNum):
        self.opId = opId
        self.tid = tid
        self.arr = arr
        self.arr_val = arr_val
        self.joinAttributes = joinAttributes
        self.tupleNum = tupleNum        
        self.inRelation = {}
        self.outRelation = {}
        self.opType = 'intersectionselection'

class Intersection( UnaryAlgExpr ):

    def __init__ ( self, ctxt, vname, variables, tries, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.scanAttributes = dict()
        self.tries = tries
        self.variables = variables
        self.vname = vname
        self.joinAttributes = OrderedDict()
        self.opType = 'intersection'

        self.tables = []
        for (tableName, keys, alias, conditions) in tries:
            scanTableId, table = ctxt.resolveTable ( tableName )
            self.tables.append((scanTableId, table))
        pass

    def resolve ( self, ctxt ):

        inChild = self.child.resolve(ctxt)
        self.inRelation.update(inChild)

        id, self.arr = ctxt.createAttribute ( 'arr', self.opId, table=None, tableName=None )
        self.arr.dataType = Type.PTR_INT

        id, self.arr_val = ctxt.createAttribute ( self.vname, self.opId )
        self.arr_val.dataType = Type.INT

        id, self.tid = ctxt.createAttribute ( 'tid', self.opId, table=None, tableName=None )
        self.tid.dataType = Type.INT

        self.tids = []
        for idx, ((tableName, keys, alias, conditions), (scanTableId, table)) in enumerate(zip(self.tries, self.tables)):

            print(tableName, keys, alias)

            # find the appropriate trie
            trieName = 'idx_{}_{}'.format(tableName, '_'.join(keys))
            for trieCandName in table['index_sizes'].keys():
                if trieName == trieCandName[:len(trieName)]:
                    trieName = trieCandName
                    break
            sizes = table['index_sizes'][trieName]

            id, attrTid = ctxt.createAttribute ( 'tid', self.opId, table=table, tableName=tableName )
            self.tids.append(attrTid)
            attrTid.tid = None
            attrTid.dataType = Type.INT
            attrTid.trieName = trieName
            attrTid.trieKeys = keys
            attrTid.trieKeyAttrs = [None] * len(keys)
            attrTid.trieSize = sizes
            attrTid.scanTableId = scanTableId
            attrTid.conditions = conditions
            if alias != None: attrTid.identifiers.append(alias + ".tid")

            # create attributes
            for (attrName, attrType) in table [ "attributes" ].items():
                #print("\t", attrName, attrType)
                id, attr = ctxt.createAttribute ( attrName, self.opId, table=table, tableName=tableName )
                attr.tid = attrTid
                attr.dataType = attrType
                attr.numElements = table['size']
                if alias != None: attr.identifiers.append ( alias + "." + attrName )
                self.scanAttributes[id] = attr
                attr.keyIdx = -1
                for keyIdx, key in enumerate(attr.tid.trieKeys):
                    if key == attrName:
                        attr.keyIdx = keyIdx
                        attrTid.trieKeyAttrs[keyIdx] = attr
                        break
                attr.isLastKey = attrName == keys[-1]

        for ident in self.variables:
            try:
                attr = self.resolveAttribute(ident)
            except:
                attr = self.resolveAttribute(ident, self.scanAttributes)
            self.joinAttributes[attr.id] = attr

        self.outRelation.update(self.inRelation)
        self.outRelation.update(self.scanAttributes)        
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure(cfg, ctxt)
        if len(self.tables) > 0:
            self.tupleNum = childTupleNum * max(list(map(lambda x: x[1]['size'], self.tables)))
        else:
            self.tupleNum = childTupleNum
        #self.tupleNum = self.minAttr.tid.trieSize[self.minAttr.name]
        return self.tupleNum

    def DOTcaption ( self ):
        return 'trie scan'


    def DOTsubcaption ( self ):
        return self.variables

    def printSelf(self, ctxt):
        return

    def genPipes(self,ctxt):

        self.child.genPipes(ctxt)

        ctxt['pipelines'][-1][-1].append(self)
        sel = IntersectionSelection('{}_s'.format(self.opId), self.tid, self.arr, self.arr_val, self.joinAttributes, self.tupleNum)
        sel.outRelation.update(self.outRelation)
        
        sel.inRelation.update(self.outRelation)
        sel.inRelation.update(self.joinAttributes)
        
        sel.inRelation[self.tid.id] = self.tid
        sel.inRelation[self.arr.id] = self.arr
        if self.arr_val.id in sel.inRelation: 
            sel.inRelation.pop(self.arr_val.id)

        for attrId, attr in self.scanAttributes.items():
            if attrId in sel.inRelation:
                sel.inRelation.pop(attrId)

        for tid in self.tids:
            if tid.id in sel.inRelation:
                sel.inRelation.pop(tid.id)

        self.outRelation = sel.inRelation

        ctxt['pipelines'][-1].append([sel])
        ctxt['pipelines'][-1].append([])
        return


class TrieScan ( LiteralAlgExpr ):

    def __init__ ( self, ctxt, vname, variables, tries):
        LiteralAlgExpr.__init__ ( self, ctxt )
        self.scanAttributes = dict()
        self.tries = tries
        self.variables = variables
        self.opType = 'triescan'
        self.tables = []
        self.joinAttributes = OrderedDict()
        self.vname = vname

        for (tableName, keys, alias, conditions) in tries:
            scanTableId, table = ctxt.resolveTable ( tableName )
            self.tables.append((scanTableId, table))


    def resolve ( self, ctxt ):

        id, attr = ctxt.createAttribute ( 'arr', self.opId, table=None, tableName=None )
        self.arr = attr
        self.arr.dataType = Type.PTR_INT

        id, attr = ctxt.createAttribute ( self.vname, self.opId )
        self.arr_val = attr
        self.arr_val.dataType = Type.INT

        id, attr = ctxt.createAttribute ( 'tid', self.opId, table=None, tableName=None )
        self.tid = attr
        self.tid.dataType = Type.INT

        self.tids = []

        for idx, ((tableName, keys, alias, conditions), (scanTableId, table)) in enumerate(zip(self.tries, self.tables)):

            print(tableName, keys, alias)

            # find the appropriate trie
            trieName = 'idx_{}_{}'.format(tableName, '_'.join(keys))
            for trieCandName in table['index_sizes'].keys():
                if trieName == trieCandName[:len(trieName)]:
                    trieName = trieCandName
                    break
            sizes = table['index_sizes'][trieName]

            id, attrTid = ctxt.createAttribute ( 'tid', self.opId, table=table, tableName=tableName )
            self.tids.append(attrTid)
            attrTid.tid = None
            attrTid.dataType = Type.INT
            attrTid.trieName = trieName
            attrTid.trieKeys = keys
            attrTid.trieKeyAttrs = [None] * len(keys)
            attrTid.trieSize = sizes
            attrTid.scanTableId = scanTableId
            attrTid.conditions = conditions
            if alias != None: attrTid.identifiers.append(alias + ".tid")

            # create attributes
            for (attrName, attrType) in table [ "attributes" ].items():
                #print("\t", attrName, attrType)
                id, attr = ctxt.createAttribute ( attrName, self.opId, table=table, tableName=tableName )
                attr.tid = attrTid
                attr.dataType = attrType
                attr.numElements = table['size']
                if alias != None: attr.identifiers.append ( alias + "." + attrName )
                self.scanAttributes[id] = attr
                attr.keyIdx = -1
                for keyIdx, key in enumerate(attr.tid.trieKeys):
                    if key == attrName:
                        attr.keyIdx = keyIdx
                        attrTid.trieKeyAttrs[keyIdx] = attr
                        break
                attr.isLastKey = attrName == keys[-1]

        minAttr = None
        for ident in self.variables:
            try:
                attr = self.resolveAttribute(ident)
            except:
                attr = self.resolveAttribute(ident, self.scanAttributes)
            self.joinAttributes[attr.id] = attr
            if minAttr is None or attr.tid.trieSize[attr.name] < minAttr.tid.trieSize[minAttr.name]:
                minAttr = attr
        self.minAttr = minAttr

        self.outRelation = self.scanAttributes
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        self.tupleNum = self.minAttr.tid.trieSize[self.minAttr.name]
        return self.tupleNum

    def DOTcaption ( self ):
        return 'trie scan'


    def DOTsubcaption ( self ):
        return self.variables

    def printSelf(self, ctxt):
        return

    def genPipes(self,ctxt):
        ctxt['pipelines'][-1][-1].append(self)
        ctxt['pipelines'][-1].append([])
        return




class Selection ( UnaryAlgExpr ):

    def __init__ ( self, ctxt, condition, tupleNum, doLB, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.condition = condition
        self.opType = "selection"
        self.tupleNum = tupleNum
        self.doLB = doLB
        self.outputCard = '1'
    
    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        self.conditionAttributes = self.condition.resolve ( self )
        self.outRelation = self.inRelation
        return self.outRelation
    
    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        if self.tupleNum == None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                self.selectivity = 1.0
            self.tupleNum = int ( childTupleNum * self.selectivity )
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.condition.toDOT ( graph )
        graph.edge ( str ( self.condition.exprId ) , str ( self.opId ) )
        
    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Selection {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        print("\t",self.condition.gen({}))
        #print("\t",self.condition.translate(ctxt))
        print("\t",)
        print("\t","/".join(list(map(lambda x: x.toString(), self.conditionAttributes.values()))))

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'selection'
        myCtxt['children'] = [ self.child.toJson()]
        return myCtxt

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        ctxt['pipelines'][-1].append([])
        return

class Map ( UnaryAlgExpr ):

    def __init__ ( self, ctxt, attrName, expression, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.expression = expression
        self.attrName = attrName
        self.opType = "map"
    
    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        self.mappedAttributes = self.expression.resolve ( self )
        self.attrId, attr = ctxt.createAttribute ( self.attrName, self.opId )
        self.mapAttr = attr
        self.mapAttr.dataType = self.expression.type
        self.mapStringAttributes = [ att for (id, att) in self.mappedAttributes.items() if att.dataType == Type.STRING ]
        res = dict ( self.inRelation ) 
        res [ self.mapAttr.id ] = self.mapAttr
        self.outRelation = res
        return res
    
    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        self.tupleNum = childTupleNum
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.expression.toDOT ( graph )
        graph.edge ( str ( self.expression.exprId ) , str ( self.opId ) )

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Map {} tupleNum: {}".format(self.opId, self.tupleNum))
        ctxt["attrName"] = "attr{}_{}".format(self.attrId, self.attrName)
        print("\t",self.expression.gen(ctxt))
        print("\t",self.attrName)
        print("\tout relation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.outRelation.values()))))
        print("\tmappedAttributes\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.mappedAttributes.values()))))

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        return

        operator = {}
        operator['opId'] = self.opId
        operator['type'] = 'Map'
        operator['tupleNum'] = self.tupleNum
        operator['usingAttributes'] = list(map(lambda x: x.toDic(), self.mappedAttributes.values()))
        operator['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))
        operator['expr'] = self.expression
        operator['attrName'] = self.attrName
        operator['attrId'] = self.attrId
        ctxt['pipelines'][-1].append(operator)

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'map'
        myCtxt['children'] = [ self.child.toJson()]
        return myCtxt


class MultiMap ( UnaryAlgExpr ):

    def __init__ ( self, ctxt, attrs, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.attrs = attrs
        self.opType = "multimap"
    
    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        self.mappedAttributes = dict()
        self.attrIds = []
        self.mapAttrs = []
        for attrName, expression in self.attrs:
            self.mappedAttributes.update(expression.resolve(self))
            attrId, attr = ctxt.createAttribute ( attrName, self.opId )
            self.attrIds.append(attrId)
            attr.dataType = expression.type
            self.mapAttrs.append(attr)
        self.mapStringAttributes = [ att for (id, att) in self.mappedAttributes.items() if att.dataType == Type.STRING ]
        res = dict ( self.inRelation ) 
        for attr in self.mapAttrs:
            res[attr.id] = attr
        self.outRelation = res
        return res
    
    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        self.tupleNum = childTupleNum
        return self.tupleNum

    def DOTscalarExpr ( self, graph ):
        self.expression.toDOT ( graph )
        graph.edge ( str ( self.expression.exprId ) , str ( self.opId ) )

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Map {} tupleNum: {}".format(self.opId, self.tupleNum))
        ctxt["attrName"] = "attr{}_{}".format(self.attrId, self.attrName)
        print("\t",self.expression.gen(ctxt))
        print("\t",self.attrName)
        print("\tout relation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.outRelation.values()))))
        print("\tmappedAttributes\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.mappedAttributes.values()))))

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'map'
        myCtxt['children'] = [ self.child.toJson()]
        return myCtxt



class Join ( Enum ):
    # handled by CrossJoin
    CROSS = 1
    # handled by EquiJoin
    INNER = 2
    SEMI  = 3
    ANTI  = 4
    OUTER = 5


class MultiwayJoin ( NaryAlgExpr ):

    def __init__( self, ctxt, variables, children ):
        NaryAlgExpr.__init__ (self, ctxt, children )
        self.keyIdentities = variables
        self.opType = "multiwayjoin"

    def prune ( self ):
        super().prune()

    def resolve ( self, ctxt ):
        self.inChildren = []
        #print(self.children)
        for child in self.children:
            inChild = child.resolve( ctxt )
            self.inRelation.update ( inChild )
            self.inChildren.append(inChild)

        self.keyAttributes = [ None ] * len(self.keyIdentities)
        for ident in self.keyIdentities:
            #print(ident)
            att = self.resolveAttribute(ident)
            for idx, inChild in enumerate(self.inChildren):
                if att.id in inChild:
                    self.keyAttributes[idx] = att
                    break

        return self.inRelation

    def configure (self, cfg, ctxt ):
        if self.tupleNum == None:
            self.tupleNum = int(max (list(map(lambda x: x.configure(cfg, ctxt), self.children))))
        return self.tupleNum
        

    def genPipes(self, ctxt):
        operator = {}
        operator['opId'] = self.opId
        operator['type'] = 'MinProbeTrie'
        operator['tupleNum'] = self.tupleNum
        operator['usingAttributes'] = list(map(lambda x: x.toDic(), self.keyAttributes))
        operator['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))
        
        scans = []
        for child in self.children:
            if child.opType == 'triescan':
                childOp = {}
                childOp['opId'] = child.opId
                childOp['type'] = child.DOTcaption()
                childOp['tupleNum'] = child.tupleNum
                childOp['table'] = child.table['name']
                childOp['trie'] = '{}__{}'.format(child.table['name'], '_'.join(child.keyNames))
                childOp['keys'] = child.keyNames
                childOp['usingAttributes'] = []
                childOp['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > child.opId, child.scanAttributes.values())))
                scans.append(childOp)
            else:
                child.genPipes(ctxt)
        
        operator['scans'] = scans
        operator['isFirst'] = True if len(scans) == len(self.children) else False

        ctxt['pipelines'][-1].append(operator)


        operator = {}
        operator['opId'] = '{}_s'.format(self.opId)
        operator['type'] = 'SemiProbeTrie'
        operator['tupleNum'] = self.tupleNum
        operator['usingAttributes'] = list(map(lambda x: x.toDic(), self.keyAttributes))
        operator['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))

        ctxt['pipelines'][-1].append(operator)

    def toJson ( self ):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'multiwayjoin'
        myCtxt['children'] = list(map(lambda x: x.toJson(), self.children))
        return myCtxt


class FakeSelection( UnaryAlgExpr ):

    def __init__(self, opId, condition, tupleNum, inRelation, outRelation):
        self.condition = condition
        self.opId = opId
        self.opType = "selection"
        self.tupleNum = tupleNum
        self.inRelation = {}
        self.outRelation = {}
        self.inRelation.update(inRelation)
        self.outRelation.update(outRelation)
        self.conditionAttributes = self.condition.resolve ( self )
        self.outputCard = '1'

class AggSelection( UnaryAlgExpr ):

    def __init__(self, opId, tid, tupleNum, inRelation, outRelation):
        self.opId = opId
        self.tid = tid
        self.opType = "aggselection"
        self.tupleNum = tupleNum
        self.inRelation = {}
        self.outRelation = {}
        self.inRelation.update(inRelation)
        self.outRelation.update(outRelation)
        self.outputCard = '1'

class EquiJoin  ( BinaryAlgExpr ):

    def __init__ ( self, ctxt, joinType, equalityConditions, otherConditions, tupleNum, multimatch, doLB, leftChild, rightChild ):
        BinaryAlgExpr.__init__ ( self, ctxt, leftChild, rightChild )
        self.equalities = listWrap ( equalityConditions )
        self.conditions = otherConditions
        self.joinType = joinType
        
        opTypes = {
            Join.INNER: "equijoin",
            Join.SEMI: 'semijoin',
            Join.OUTER: 'outerjoin',
            Join.ANTI: 'antijoin'
        }
        
        self.opType = opTypes[joinType]
        self.tupleNum = tupleNum
        self.multimatch = multimatch
        self.htSizeFactor = None
        self.doLB = doLB
        self.doShuffle = doLB
        self.outputCard = 'm' if self.multimatch else '1'

    
    def prune ( self ):
        super().prune ( )

    def resolve ( self, ctxt ): 
        inLeft = self.leftChild.resolve ( ctxt )
        self.inLeft = inLeft
        
        #print('hi', self.opId, list(self.inLeft.keys()))
        self.inRelation.update ( inLeft )
        inRight = self.rightChild.resolve ( ctxt )
        self.inRelation.update ( inRight )

        id, attr = ctxt.createAttribute ( 'tid', self.opId )
        self.tid = attr
        self.tid.dataType = Type.INT

        self.conditionAttributes = dict()
        if self.conditions != None:
            self.conditionAttributes = self.conditions.resolve ( self )
        self.conditionProbeAttributes = dict()
        self.conditionBuildAttributes = dict()
        for id, att in self.conditionAttributes.items():
            if id in inRight:
                self.conditionProbeAttributes[id] = att
            else:
                self.conditionBuildAttributes[id] = att
        
        # determine source relation for condition attributes
        self.buildKeyAttributes = OrderedDict()
        self.probeKeyAttributes = OrderedDict()
        for ( ident1, ident2 ) in self.equalities:
            att1 = self.resolveAttribute ( ident1 )
            att2 = self.resolveAttribute ( ident2 )
            if att1.id in inLeft and att2.id in inRight:
                self.buildKeyAttributes[att1.id] = att1
                self.probeKeyAttributes[att2.id] = att2
            elif att1.id in inRight and att2.id in inLeft:
                self.buildKeyAttributes[att2.id] = att2
                self.probeKeyAttributes[att1.id] = att1
            else:
                print(self.equalities)
                raise NameError ( "Attributes " + att1.name + " and " + att2.name + "come from the same relation." )
        if self.joinType in [ Join.INNER, Join.OUTER ]:
            self.outRelation = self.inRelation
        else:
            self.outRelation = inRight
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        leftChildTupleNum = self.leftChild.configure ( cfg, ctxt )
        rightChildTupleNum = self.rightChild.configure ( cfg, ctxt )

        if self.htSizeFactor == None:
            try:
                self.htSizeFactor = cfg[self.opId]["htSizeFactor"]
            except:
                self.htSizeFactor = 2.0

        if self.tupleNum == None:
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                self.selectivity = 1.0
            if self.multimatch == False:
                self.tupleNum = rightChildTupleNum * self.selectivity
            else:
                self.tupleNum = int ( max ( leftChildTupleNum, rightChildTupleNum ) * self.selectivity )
        #print(self.opId, self.tupleNum, self.selectivity, self.htSizeFactor)
        return self.tupleNum
    
    def DOTcaption ( self ):
        joinStr = str ( self.joinType )
        # indicate that join has more non-equality conditions
        if self.conditions != None:
            joinStr += '*'
        return joinStr[0] + joinStr[1:].lower()

    def DOTsubcaption ( self ):
        return list ( map ( lambda x,y: x[1].name + "=" + y[1].name, self.buildKeyAttributes.items(), 
            self.probeKeyAttributes.items() ) )

    def printSelf(self,ctxt):
        self.leftChild.printSelf(ctxt)
        self.rightChild.printSelf(ctxt)
        print("EquiJoin {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        print("\t",self.equalities)
        print("\t",self.conditions)
        print("\t",self.joinType)
        #print("\t", self.inRelation)
        #print("\t",self.outRelation)
        #print(self.buildKeyAttributes.values())
        print("\t","/".join(list(map(lambda x: x.toString(), self.buildKeyAttributes.values()))))
        print("\t","/".join(list(map(lambda x: x.toString(), self.probeKeyAttributes.values()))))
        #print("\t",dict(self.probeKeyAttributes))
        print("\t",self.multimatch)

    def genPipes(self, ctxt):
        self.materialized = 0
        pipe = ctxt['pipelines'][-1]
        ctxt['pipelines'][-1] = [[]]
        self.leftChild.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)

        if KernelCall.args.isOldCompiler:
            if self.multimatch and self.joinType == Join.INNER:
                ctxt['pipelines'].append(ctxt['pipelines'][-1])

        ctxt['pipelines'].append(pipe)
        self.rightChild.genPipes(ctxt)

        ctxt['pipelines'][-1][-1].append(self)

        if self.multimatch == False or self.joinType == Join.SEMI:
            ctxt['pipelines'][-1].append([])
            return

        buildKeyAttributes = list(map(lambda x: x.toDic(), self.buildKeyAttributes.values()))
        probeKeyAttributes = list(map(lambda x: x.toDic(), self.probeKeyAttributes.values()))
        attr_b = AttrExpr(buildKeyAttributes[0]['identifiers'][-1])
        attr_b.id = buildKeyAttributes[0]['id']
        attr_p = AttrExpr(probeKeyAttributes[0]['identifiers'][-1])
        attr_p.id = probeKeyAttributes[0]['id']
        condition = EqualsExpr(attr_b, attr_p)
        condition.comparisonType = buildKeyAttributes[0]['dataType']
        for idx in range(len(buildKeyAttributes)-1):
            i = idx + 1
            attr_b = AttrExpr(buildKeyAttributes[i]['identifiers'][-1])
            attr_b.id = buildKeyAttributes[i]['id']
            attr_p = AttrExpr(probeKeyAttributes[i]['identifiers'][-1])
            attr_p.id = probeKeyAttributes[i]['id']
            local_condition = EqualsExpr(attr_b, attr_p)
            local_condition.comparisonType = buildKeyAttributes[i]['dataType']
            condition = AndExpr(condition, local_condition)

        #print("make fake selection", self.outRelation)
        fselection = FakeSelection(f'{self.opId}_s', condition, self.tupleNum, self.inRelation, self.outRelation)
        fselection.child = self
        fselection.parent = self.parent
        fselection.cid = self.cid
        fselection.doLB = self.doLB
        self.doLB = self.multimatch

        if self.cid == 0:
            self.parent.child = fselection
        elif self.cid == 1:
            self.parent.leftChild = fselection
        elif self.cid == 2:
            self.parent.rightChild = fselection

        self.cid = 0
        self.parent = fselection


        self.outRelation = {}
        self.outRelation.update(fselection.outRelation)
        self.outRelation.update(self.buildKeyAttributes)
        self.outRelation.update(self.probeKeyAttributes)

        ctxt["pipelines"][-1].append([fselection])
        ctxt['pipelines'][-1].append([])                        
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'equijoin'
        myCtxt['children'] = [ self.leftChild.toJson(), self.rightChild.toJson() ]
        return myCtxt


class CrossJoin ( BinaryAlgExpr ):

    def __init__ ( self, ctxt, joinType, condition, tupleNum, leftChild, rightChild ):
        BinaryAlgExpr.__init__ ( self, ctxt, leftChild, rightChild )
        self.condition = condition
        self.joinType = joinType
        self.opType = "crossjoin"
        self.tupleNum = tupleNum
        self.outputCard = 'm'
        self.doLB = False

    def resolve ( self, ctxt ):
        inLeft = self.leftChild.resolve ( ctxt )
        self.inRelation.update ( inLeft )
        inRight = self.rightChild.resolve ( ctxt )
        self.inRelation.update ( inRight )

        id, attr = ctxt.createAttribute ( 'tid', self.opId )
        self.tid = attr
        self.tid.dataType = Type.INT

        self.conditionAttributes = self.condition.resolve ( self )
        self.outRelation = self.inRelation
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        leftChildTupleNum = self.leftChild.configure ( cfg, ctxt )
        rightChildTupleNum = self.rightChild.configure ( cfg, ctxt )
        
        if self.tupleNum == None:
            try:
                self.multimatch = cfg[self.opId]["multimatch"]
            except:
                self.multimatch = True
            try:
                self.selectivity = cfg[self.opId]["selectivity"]
            except:
                self.selectivity = 1.0
            self.tupleNum = int ( leftChildTupleNum * rightChildTupleNum * self.selectivity )
        return self.tupleNum
    
    def DOTscalarExpr ( self, graph ):
        self.condition.toDOT ( graph )
        graph.edge ( str ( self.condition.exprId ) , str ( self.opId ) )
    
    def DOTcaption ( self ):
        return str ( self.joinType ).lower() 

    def printSelf(self,ctxt):
        self.leftChild.printSelf(ctxt)
        self.rightChild.printSelf(ctxt)
        print("CrossJoin {} tupleNum: {} selectivity: {}".format(self.opId, self.tupleNum, self.selectivity))
        print("\t",self.condition)
        print("\t",self.joinType)
        print("\t", self.inRelation)
        print("\t",self.outRelation)
        print("\t",self.conditionAttributes)
        print("\t",self.multimatch)

    def genPipes(self, ctxt):
        pipe = ctxt['pipelines'][-1]
        ctxt['pipelines'][-1] = [[]]
        self.leftChild.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        
        ctxt['pipelines'].append(pipe)
        self.rightChild.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        
        if self.condition is None:
            return
        
        fselection = FakeSelection(f'{self.opId}_s', self.condition, self.tupleNum, self.inRelation, self.outRelation)
        fselection.child = self
        fselection.parent = self.parent
        fselection.cid = self.cid
        fselection.doLB = False # TODO

        if self.cid == 0:
            self.parent.child = fselection
        elif self.cid == 1:
            self.parent.leftChild = fselection
        elif self.cid == 2:
            self.parent.rightChild = fselection

        self.cid = 0
        self.parent = fselection

        self.outRelation = {}
        self.outRelation.update(self.leftChild.outRelation)
        self.outRelation.update(self.rightChild.outRelation)
        ctxt["pipelines"][-1].append([fselection])
        ctxt['pipelines'][-1].append([])                        
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'crossjoin'
        myCtxt['children'] = [ self.leftChild.toJson(), self.rightChild.toJson() ]
        return myCtxt


class Reduction ( Enum ):
    SUM   = 1
    COUNT = 2
    AVG   = 3
    MIN   = 4
    MAX   = 5


class Aggregation ( UnaryAlgExpr ):

    def __init__ ( self, ctxt, groupingIdentifiers, aggregates, tupleNum, doLB, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.groupingIdentifiers = listWrap ( groupingIdentifiers ) 
        self.aggregates = listWrap ( aggregates ) 
        self.aggregateTuplesCreated = OrderedDict ()
        self.aggregateAttributes = dict ()
        self.avgAggregates = dict ()
        self.opType = "aggregation"
        self.tupleNum = tupleNum
        self.doLB = doLB
        self.outputCard = '1'

        # add count for avg if necessary
        if any ( [ agg[0] == Reduction.AVG for agg in aggregates ] ):
            if not any ( [ agg[0] == Reduction.COUNT for agg in aggregates ] ):
                self.aggregates.append ( ( Reduction.COUNT, "", "count_agg" ) )

        # create aggregation attributes 
        for aggregate in aggregates:
            dType = Type.DOUBLE
            if len(aggregate) == 4:
                dType = aggregate[3]
            reductionType, inputIdentifier, aliasName = aggregate[:3]
            id, aggAttr = ctxt.createAttribute ( aliasName, self.opId )
            aggAttr.dataType = dType
            if reductionType == Reduction.COUNT:
                aggAttr.dataType = Type.ULL # kjhong
                self.countAttr = aggAttr
            self.aggregateAttributes [ id ] = aggAttr
            if reductionType == Reduction.AVG:
                self.avgAggregates [ id ] = aggAttr
            self.aggregateTuplesCreated [ id ] = ( aggAttr, inputIdentifier, reductionType )
        
    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )

        id, attr = ctxt.createAttribute ( 'tid', self.opId, table=None, tableName=None )
        self.tid = attr
        attr.dataType = Type.INT

        self.groupAttributes = OrderedDict()
        for ident in self.groupingIdentifiers:
            att = self.resolveAttribute ( ident )
            self.groupAttributes [ att.id ] = att

        self.doGroup = len ( self.groupAttributes ) > 0
        self.aggregateInAttributes = OrderedDict()
        self.aggregateTuples = dict()
        for id, ( aggAttr, inputIdentifier, reductionType ) in self.aggregateTuplesCreated.items():
            inId = None
            if inputIdentifier != "":
                inAtt = self.resolveAttribute ( inputIdentifier )
                self.aggregateInAttributes [ inAtt.id ] = inAtt
                inId = inAtt.id
            self.aggregateTuples [ id ] = ( inId, reductionType )
        self.outRelation.update ( self.groupAttributes )
        self.outRelation.update ( self.aggregateAttributes )

        #for attrId, attr in self.outRelation.items():
        #    print("hihi", attrId, attr.name, attr.creation)
        return self.outRelation

    def configure ( self, cfg, ctxt ):
        childTupleNum = self.child.configure ( cfg, ctxt )
        if self.tupleNum == None:
            try:
                self.tupleNum = cfg[self.opId]["numgroups"]
            except:
                self.tupleNum = childTupleNum
            if not self.doGroup:
                self.tupleNum = 1
        
        self.numgroups = self.tupleNum
        return self.tupleNum
    
    def DOTsubcaption ( self ):       
        if len ( self.groupAttributes ) == 0: 
            return ""
        return list ( map ( lambda x: x[1].name, self.groupAttributes.items() ) ) 

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Reduction {} tuplenum: {} numgroups: {}".format(self.opId, self.tupleNum, self.numgroups))
        print("\t",self.groupingIdentifiers)
        print("\t",self.aggregates)
        print("\t",self.aggregateTuplesCreated)
        print("\t",self.aggregateAttributes)
        print("\t",self.avgAggregates)
    
    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        ctxt['pipelines'].append([[self]])
        if self.doGroup:
            inRelation, outRelation = {}, {}
            inRelation.update(self.inRelation)
            inRelation[self.tid.id] = self.tid
            outRelation.update(self.outRelation)
            opId = f'{self.opId}_s'
            aggselection = AggSelection(opId, self.tid, self.tupleNum, inRelation, outRelation)
            aggselection.child = self
            aggselection.parent = self.parent
            aggselection.doLB = self.doLB
            self.doLB = False
            
            if self.cid == 0:
                self.parent.child = aggselection
            elif self.cid == 1:
                self.parent.leftChild = aggselection
            elif self.cid == 2:
                self.parent.rightChild = aggselection

            self.parent = aggselection
            self.cid = 0
            
            self.outRelation[self.tid.id] = self.tid
            ctxt['pipelines'][-1][-1].append(aggselection)
            ctxt['pipelines'][-1].append([])
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'aggregation'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt

class Projection ( UnaryAlgExpr ):
    
    def __init__ ( self, ctxt, identifiers, child ):
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.identifiers = listWrap ( identifiers )
        self.opType = "projection"
    
    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        self.projectionAttributes = dict()
        for ident in self.identifiers:
            att = self.resolveAttribute ( ident )
            self.projectionAttributes [ att.id ] = att  
        self.outRelation = self.projectionAttributes
        return self.outRelation
    
    def configure ( self, cfg, ctxt ):
        self.tupleNum = self.child.configure ( cfg, ctxt )
        return self.tupleNum

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Projection {} tupleNum: {}".format(self.opId, self.tupleNum))
        #print("\t",self.identifiers)
        print("\tinRelation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.inRelation.values()))))
        print("\toutRelation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.outRelation.values()))))

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        #ctxt['pipelines'][-1].append(self)
        return

        operator = {}
        operator['opId'] = self.opId
        operator['type'] = 'Projection'
        operator['tupleNum'] = self.tupleNum
        operator['usingAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))
        operator['outAttributes'] = list(map(lambda x: x.toDic(), filter(lambda x: x.lastUse > self.opId, self.outRelation.values())))
        ctxt['pipelines'][-1].append(operator)

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'projection'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt


class MaterializationType ( Enum ):
    RESULT = 1
    TEMPTABLE = 2


class Materialize ( UnaryAlgExpr ):
    
    def __init__ ( self, ctxt, child ):
        self.table = None
        UnaryAlgExpr.__init__ ( self, ctxt, child )
        self.opType = "materialize"
   
    # -- constructor1 
    def result ( ctxt, child ):
        mat = Materialize ( ctxt, child )
        mat.matType = MaterializationType.RESULT
        mat.isResult = True
        return mat

    # -- constructor2
    def temp ( ctxt, identifier, child ):
        mat = Materialize ( ctxt, child )
        mat.table = ctxt.createTable ( identifier )
        mat.table [ "isTempTable" ] = True
        mat.matType = MaterializationType.TEMPTABLE
        mat.isResult = False
        return mat

    def resolve ( self, ctxt ):
        self.inRelation = self.child.resolve ( ctxt )
        for id, att in self.inRelation.items():
            self.touchAttribute ( id )
        if self.matType == MaterializationType.TEMPTABLE:
            self.table [ "sourceAttributes" ] = dict()
            for id, att in self.inRelation.items():
                self.table [ "attributes" ] [ att.name ] = att.dataType 
                self.table [ "sourceAttributes" ] [ att.name ] = att.id 
            self.table [ "numColumns" ] = len ( self.inRelation )
        self.outRelation = self.inRelation
        return self.outRelation
   
    # skip pruning for this operator 
    def prune ( self ):
        self.child.prune()
    
    def configure ( self, cfg, ctxt ):
        self.tupleNum = self.child.configure ( cfg, ctxt )
        if self.matType == MaterializationType.TEMPTABLE:
            self.table [ "size" ] = self.tupleNum
        return self.tupleNum
    
    def DOTcaption ( self ):
        if self.matType == MaterializationType.RESULT:
            return "Result"
        elif self.matType == MaterializationType.TEMPTABLE:
            return "Temptable"
 
    def DOTsubcaption ( self ):       
        if self.matType == MaterializationType.RESULT:
            return ""
        elif self.matType == MaterializationType.TEMPTABLE:
            return self.table["name"]

    def printSelf(self,ctxt):
        self.child.printSelf(ctxt)
        print("Materialization {} tupleNum: {}".format(self.opId, self.tupleNum))
        print("\tinRelation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.inRelation.values()))))
        print("\toutRelation\n\t\t","\n\t\t".join(list(map(lambda x: x.toString(), self.outRelation.values()))))

    def genPipes(self,ctxt):
        self.child.genPipes(ctxt)
        ctxt['pipelines'][-1][-1].append(self)
        return

    def toJson(self):
        myCtxt = {}
        myCtxt['opId'] = self.opId,
        myCtxt['type'] = 'projection'
        myCtxt['children'] = [ self.child.toJson() ]
        return myCtxt


    
        
