from dogqc.types import Type
from dogqc.cudalang import CType
from dogqc.relationalAlgebra import Reduction
from dogqc.kernel import KernelCall

from themis.code import Code
from themis.datastructure import langType, Datastructures, BaseColumn, TempTable, ResultTable, Index, AggHT, MultiHT, UniqueHT

class Location:
    
    def __init__(self, expr, tid=None):
        self.expr = expr
        self.tid
        pass
    
    def __str__(self):
        if self.tid is None:
            return self.expr
        return self.expr.format(tid)

class OperatorContext:
    
    def __init__(self, pctxt):
        self.attrLoc = {}
        self.pctxt = pctxt


class PhysicalOperator: 
    
    def initAttrs(self):
        self.inAttrs = {}
        self.usingAttrs = {}
        self.generatingAttrs = {}
        self.mappedAttrs = {}
        self.outAttrs = {}
        
    
    def __init__(self, lop, name, touched):
        self.opId = lop.opId
        self.lop = lop
        self.name = name
        self.touched = touched
        self.dss = Datastructures()
        self.tid = None
        self.initAttrs()
        

    def __str__(self):
        
        # inAttrs = '[' + ','.join(list(map(lambda x: str(x.id_name), self.inAttrs.values()))) + ']'
        # usingAttrs = '[' + ','.join(list(map(lambda x: str(x.id_name), self.usingAttrs.values()))) + ']'
        # generatingAttrs = '[' + ','.join(list(map(lambda x: str(x.id_name), self.generatingAttrs.values()))) + ']'
        # outAttrs = '[' + ','.join(list(map(lambda x: str(x.id_name), self.outAttrs.values()))) + ']'
        
        inAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.inAttrs.values()))) + ']'
        usingAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.usingAttrs.values()))) + ']'
        generatingAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.generatingAttrs.values()))) + ']'
        outAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.outAttrs.values()))) + ']'

        dss = '[' + ','.join(list(map(lambda x: str(x), self.dss.keys()))) + ']'

        return f"op {self.lop.opId}, {self.name}, touched: {self.touched}, attrs: {inAttrs}/{usingAttrs}/{generatingAttrs}/{outAttrs}, dss: {dss}"
    
    def resolveAttributes(self, next_op):
        if next_op != None:
            self.outAttrs.update(next_op.inAttrs)
        for attr_id, attr in self.outAttrs.items():
            if attr_id not in self.generatingAttrs:
                self.inAttrs[attr_id] = attr
        for attr_id, attr in self.usingAttrs.items():
            if attr_id not in self.generatingAttrs:
                self.inAttrs[attr_id] = attr
        if self.tid is not None:
            for attrId, attr in self.generatingAttrs.items():
                if attrId == self.tid.id:
                    self.pipe.tids[self.tid.id] = self.tid
                else:
                    self.pipe.attr2tid[attrId] = self.tid.id 

    def resolveAttrsToMaterialize(self, materializedAttrs):
        pass
        

    def resolveDatastructures(self):
        pass
    
    def resolveTidsOfAttrs(self):
        pass

    def genPipeVar(self):
        return Code()
    
    def genSubpipeVar(self):
        return Code()

    def genLocalVar(self, declaredAttrs):
        return Code()

    def genOperation(self):
        return Code()
    
    def genCodeAfterExecution(self):
        return Code()

    def genCodeBeforeExecution(self):
        return Code()
    
    def genPipePostCode(self):
        return Code()
    
    
    
class Scan(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'scan', touched)
        self.tid = lop.tid
    
    def resolveAttributes(self, next_op):
        self.generatingAttrs[self.tid.id] = self.tid
        tableName = self.lop.table['name']
        for attrId, attr in self.lop.outRelation.items():
            if attrId in self.lop.scanAttributes:
                self.generatingAttrs[attrId] = attr
                
        # Generate origin expr for data load
        for attrId, attr in self.generatingAttrs.items():
            if attrId == self.tid.id:
                exp = self.tid.id_name
            else:
                colName = f"{tableName}_{attr.name}"                
                if Type.STRING == attr.dataType and not self.lop.isTempScan:
                    exp = f"stringScan({colName}_offset,{colName}_char,{self.tid.id_name})"
                else:
                    exp = f"{colName}[{self.tid.id_name}]"
            self.pipe.originExpr[attrId] = exp

        super().resolveAttributes(next_op)
    
    def resolveDatastructures(self):
        tableName = self.lop.table['name']
        if self.lop.isTempScan:
            attrs = {}
            for attrId, attr in self.lop.outRelation.items():
                if attrId not in self.lop.scanAttributes: continue
                if self.tid.id == attrId: continue
                attrs[attrId] = attr
            ds = TempTable(tableName, self.lop.tupleNum, attrs)
            self.dss.add(ds)
        else:
            for attrId, attr in self.lop.outRelation.items():
                if attrId not in self.lop.scanAttributes: continue
                if attrId == self.tid.id: continue
                #print(tableName, attr.id_name)
                ds = BaseColumn(tableName, attr)
                self.dss.add(ds)

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        return c
    
    def genTableSize(self):
        if self.lop.isTempScan:
            tableName = self.lop.table['name']
            return f'*nout_{tableName}'
        else:
            return str(self.lop.tupleNum)

class IndexJoin(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'indexjoin', touched)
        self.ftable = lop.ftable
        self.ptable = lop.table['name']
        self.rel_name = lop.rel_name
        assert(lop.tid is not None)
        self.tid = lop.tid
        self.doConvert = lop.doConvert
        print(lop.rel_name, lop.doConvert)
        
    def resolveAttributes(self, next_op):
        self.usingAttrs[self.lop.condAttr.id] = self.lop.condAttr
        for attrId, attr in self.lop.outRelation.items():
            if attrId in self.lop.scanAttributes:
                self.generatingAttrs[attrId] = attr
        #print(self.rel_name)
        self.generatingAttrs[self.tid.id] = self.tid
        
        # Generate origin expr for data load
        tableName = self.ptable
        for attrId, attr in self.generatingAttrs.items():
            colName = f"{tableName}_{attr.name}"
            if attrId == self.tid.id:
                exp = self.tid.id_name
            elif Type.STRING == attr.dataType:
                exp = f"stringScan({colName}_offset,{colName}_char,{self.tid.id_name})"
            else:
                exp = f"{colName}[{self.tid.id_name}]"
            self.pipe.originExpr[attrId] = exp
        
        super().resolveAttributes(next_op)
        
    def resolveDatastructures(self):
        ds = Index(self.rel_name, self.ftable, self.ptable)
        self.dss.add(ds)
        for attrId, attr in self.lop.outRelation.items():
            if attrId not in self.lop.scanAttributes: continue
            if attrId == self.tid.id: continue
            ds = BaseColumn(self.ptable, attr)
            self.dss.add(ds)
    
    def genLocalVar(self, declaredAttrs):
        c = Code()
        c.add(f'Range local{self.opId}_range;')
        if self.lop.unique and self.tid.id not in declaredAttrs:
            c.add(f'int {self.tid.id_name};')
        return c
    
    def genConvertTid(self, var):
        if self.doConvert:
            return f'indexGetPid({self.rel_name}_position, {var})'
        else: return var

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        c.add(f"active = indexProbeMulti({self.rel_name}_offset, {self.lop.condAttr.id_name}, local{self.opId}_range.start, local{self.opId}_range.end);")
        if self.lop.unique:
            var = f'local{self.opId}_range.start'
            c.add(f'{self.tid.id_name} = {self.genConvertTid(var)};')
        return c

class MWayIndexJoin(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'mwayindexjoin', touched)
        self.ftable = lop.ftable
        self.ptable = lop.table['name']
        self.rel_name = lop.rel_name
        assert(lop.tid is not None)
        self.tid = lop.tid
        self.doConvert = lop.doConvert
        print(lop.rel_name, lop.doConvert)
        
    def resolveAttributes(self, next_op):
        
        for attr in self.lop.condAttrs:
            self.usingAttrs[attr.id] = attr

        for attrId, attr in self.lop.outRelation.items():
            if attrId in self.lop.scanAttributes:
                self.generatingAttrs[attrId] = attr
        #print(self.rel_name)
        self.generatingAttrs[self.tid.id] = self.tid
        
        # Generate origin expr for data load
        tableName = self.ptable
        for attrId, attr in self.generatingAttrs.items():
            colName = f"{tableName}_{attr.name}"
            if attrId == self.tid.id:
                exp = self.tid.id_name
            elif Type.STRING == attr.dataType:
                exp = f"stringScan({colName}_offset,{colName}_char,{self.tid.id_name})"
            else:
                exp = f"{colName}[{self.tid.id_name}]"
            self.pipe.originExpr[attrId] = exp
        
        super().resolveAttributes(next_op)
        
    def resolveDatastructures(self):
        ds = Index(self.rel_name, self.ftable, self.ptable)
        self.dss.add(ds)
        for attrId, attr in self.lop.outRelation.items():
            if attrId not in self.lop.scanAttributes: continue
            if attrId == self.tid.id: continue
            ds = BaseColumn(self.ptable, attr)
            self.dss.add(ds)
    
    def genLocalVar(self, declaredAttrs):
        c = Code()
        c.add(f'Range local{self.opId}_range;')
        return c
    
    def genConvertTid(self, var):
        return var

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        
        for i, attr in enumerate(self.lop.condAttrs):
            c.add ('{')
            if i == 0:
                c.add(f"active = indexProbeMulti({self.rel_name}_offset, {attr.id_name}, local{self.opId}_range.start, local{self.opId}_range.end);")
            else:
                c.add('Range loc_range;')
                c.add(f"if (active) active = indexProbeMulti({self.rel_name}_offset, {attr.id_name}, loc_range.start, loc_range.end);")
                c.add(f'if (active && loc_range.size() < local{self.opId}_range.size()) local{self.opId}_range = loc_range;')                
            c.add('}')        
        return c


class Exist(PhysicalOperator):

    def __init__(self, lop, touched):
        super().__init__(lop, 'exist', touched)
        self.searchAttrs = self.lop.searchAttrs

    def resolveAttributes(self, next_op):
        for attr in self.lop.searchAttrs:
            self.usingAttrs[attr.id] = attr
        super().resolveAttributes(next_op)
        
    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        c.add(f"Range local{self.opId}_range;")
        baseAttr = self.searchAttrs[0]
        for attr in self.searchAttrs[1:]:
            #c.add("if (active) {")
            c.add(f"active = active && indexProbeMulti (rel_vertex_id__edge_src_offset, {attr.id_name}, local{self.opId}_range.start, local{self.opId}_range.end);")
            c.add(f"active = active && binarySearch(edge_dst, local{self.opId}_range.start, local{self.opId}_range.end, {baseAttr.id_name});")
            #c.add("}")
        return c


class Selection(PhysicalOperator):

    def __init__(self, lop, touched):
        super().__init__(lop, 'selection', touched)

    def resolveAttributes(self, next_op):
        for attrId, attr in self.lop.conditionAttributes.items():
            self.usingAttrs[attrId] = attr
        super().resolveAttributes(next_op)
        
    def genOperation(self):        
        c = Code()
        c.add('// ' + str(self))
        ctxt = OperatorContext(self.pipe)
        for attrId, attr in self.lop.conditionAttributes.items():
            ctxt.attrLoc[attrId] = attr.id_name
        c.add(f'active = {self.lop.condition.gen(ctxt)};')
        return c

class AggSelection(PhysicalOperator):

    def __init__(self, lop, touched):
        super().__init__(lop, 'aggselection', touched)
        self.tid = lop.tid

    def resolveAttributes(self, next_op):
        self.usingAttrs[self.tid.id] = self.tid
        super().resolveAttributes(next_op)
        
    def genOperation(self):        
        c = Code()
        opId = self.opId[:-2]
        c.add(f"active = aht{opId}[{self.tid.id_name}].lock.lock == OnceLock::LOCK_DONE;")
        return c

class Map(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'map', touched)

    def resolveAttributes(self, next_op):        
        self.usingAttrs.update(self.lop.mappedAttributes)
        self.generatingAttrs[self.lop.mapAttr.id] = self.lop.mapAttr
        self.mappedAttrs[self.lop.mapAttr.id] = self.lop.mapAttr
        self.pipe.originExpr[self.lop.mapAttr.id] = self.lop.mapAttr.id_name
        super().resolveAttributes(next_op)
        
    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        ctxt = OperatorContext(self.pipe)
        for attrId, attr in self.lop.mappedAttributes.items():
            ctxt.attrLoc[attrId] = attr.id_name
        
        ctxt.activecode = Code()
        ctxt.langType = langType
        rightExpr = self.lop.expression.gen(ctxt)
        c.add(ctxt.activecode)
        c.add(f'{self.lop.mapAttr.id_name} = ({rightExpr});')
        return c
        

class MultiMap(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'multimap', touched)

    def resolveAttributes(self, next_op):
        self.usingAttrs.update(self.lop.mappedAttributes)        
        for attr in self.lop.mapAttrs:
            self.generatingAttrs[attr.id] = attr
            self.pipe.originExpr[attr.id] = attr.id_name
            self.mappedAttrs[attr.id] = attr
            self.pipe.originExpr[attr.id] =  attr.id_name
        super().resolveAttributes(next_op)
        
    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        ctxt = OperatorContext(self.pipe)
        for attrId, attr in self.lop.mappedAttributes.items():
            ctxt.attrLoc[attrId] = attr.id_name
        c.add('// ' + str(self.lop.mapAttrs))
        for idx in range(len(self.lop.attrs)):
            attrName, expr = self.lop.attrs[idx]
            attr = self.lop.mapAttrs[idx]
                    
            ctxt.activecode = Code()
            ctxt.langType = langType
            rightExpr = expr.gen(ctxt)
            c.add(ctxt.activecode)
            c.add(f'{attr.id_name} = ({rightExpr});')

        return c
        # TODO

class HashJoin(PhysicalOperator):

    def __init__(self, lop, name, touched):
        super().__init__(lop, name, touched)
        self.tid = lop.tid
        #print(name, self.lop.leftChild.tupleNum, self.lop.htSizeFactor)
        self.size = int(self.lop.leftChild.tupleNum * self.lop.htSizeFactor)
        
    def genHashKeyBuild(self, attrs):
        c = Code()
        c.add('uint64_t hash_key = 0;')        
        for attrId, attr in attrs.items():
            if attr.dataType == Type.STRING:
                c.add(f'hash_key = hash(hash_key + stringHash({attr.id_name}));')
            else:
                c.add(f"hash_key = hash(hash_key + ((uint64_t){attr.id_name}));")
        return c

    def genPayloadBuild(self, attrs):
        c = Code()        
        c.add('Payload{} payl;'.format(self.opId))        
        for attrId, attr in attrs.items():
            c.add(f'payl.{attr.id_name} = {attr.id_name};')
        return c

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        if self.touched: # Probe
            c.add(self.genHashKeyBuild(self.lop.probeKeyAttributes))
            c.add(self.genProbe())
        else: # Build
            c.add(self.genHashKeyBuild(self.lop.buildKeyAttributes))
            c.add(self.genPayloadBuild(self.lop.leftChild.outRelation))
            c.add(self.genBuild())
        return c
    
    def resolveAttributes(self, next_op):
        super().resolveAttributes(next_op)

class InnerJoin(HashJoin):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'innerjoin', touched)
        self.multimatch = lop.multimatch
        self.num_touched = 0


    def resolveAttributesForProbe(self, next_op):
        self.generatingAttrs.update(self.lop.buildKeyAttributes)
        self.generatingAttrs[self.lop.tid.id] = self.lop.tid
        self.generatingAttrs.update(self.lop.leftChild.outRelation)
        self.usingAttrs.update(self.lop.probeKeyAttributes)
        if not self.multimatch:
            #self.usingAttrs.update(self.lop.buildKeyAttributes)
            self.usingAttrs.update(self.lop.conditionProbeAttributes)

        # Generate origin expr for data load
        for attrId, attr in self.generatingAttrs.items():
            if attrId == self.tid.id:
                exp = self.tid.id_name
            elif self.multimatch:
                exp = f"jht{self.opId}_payload[{self.tid.id_name}].{attr.id_name}"
            else:
                exp = f"ujht{self.opId}[{self.tid.id_name}].payload.{attr.id_name}"
            self.pipe.originExpr[attrId] = exp

    def resolveAttributesForBuild(self, next_op):
        if self.multimatch and self.num_touched == 0:
            self.usingAttrs.update(self.lop.buildKeyAttributes)
        else:
            self.usingAttrs.update(self.lop.leftChild.outRelation)

    def resolveAttributes(self, next_op):        
        if self.touched: self.resolveAttributesForProbe(next_op)
        else: self.resolveAttributesForBuild(next_op)
        super().resolveAttributes(next_op)

    def resolveDatastructures(self):
        if self.multimatch:
            self.dss.add(MultiHT(self.opId, self.size, self.lop.leftChild.tupleNum*2, self.lop.leftChild.outRelation))
        else:
            self.dss.add(UniqueHT(self.opId, self.size, self.lop.leftChild.outRelation))

    def genLocalVar(self, declaredAttrs):
        c = Code()
        if self.multimatch:
            if self.touched:
                c.add(f'Range local{self.opId}_range;')
        else:
            if self.touched:
                c.add(f'int {self.tid.id_name} = -1;')
        return c

    def genProbe(self):
        c = Code()
        if self.multimatch:
            c.add(f'active &= hashProbeMulti(jht{self.opId}, {self.size}, hash_key, local{self.opId}_range.start, local{self.opId}_range.end);')
        else:
            ctxt = OperatorContext(self.pipe)
            for attrId, attr in self.lop.conditionBuildAttributes.items():
                ctxt.attrLoc[attrId] = f'payload{self.opId}[{self.tid.id_name}].{attr.id_name}'
            for attrId, attr in self.lop.conditionProbeAttributes.items():
                ctxt.attrLoc[attrId] = attr.id_name
            c.add('int numLookups = 0;')
            c.add('int bucketFound = 0;')
            c.add(f'active = hashProbeUnique(ujht{self.opId}, {self.size}, hash_key, numLookups, {self.tid.id_name});')
            c.add('while (active) {')
            c.add('bucketFound = 1;')
            for (attrId_b, attr_b), (attrId_p, attr_p) in zip(list(self.lop.buildKeyAttributes.items()), list(self.lop.probeKeyAttributes.items())):            
                c.add(f'bucketFound = bucketFound && (ujht{self.opId}[{self.tid.id_name}].payload.{attr_b.id_name}) == ({attr_p.id_name});')
            if self.lop.conditions is not None:
                c.add(f'bucketFound = bucketFound && ({self.lop.conditions.gen(ctxt)});')
            c.add('if (bucketFound) break;')
            c.add(f'active = hashProbeUnique(ujht{self.opId}, {self.size}, hash_key, numLookups, {self.tid.id_name});')
            c.add('}')
            c.add('active = bucketFound;')
        return c

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        if self.touched: # Probe
            c.add(self.genHashKeyBuild(self.lop.probeKeyAttributes))
            c.add(self.genProbe())
        else: # Build
            c.add(self.genHashKeyBuild(self.lop.buildKeyAttributes))
            c.add(self.genBuild())
        return c

    def genBuild(self):
        c = Code()
        if self.multimatch:
            if self.num_touched == 0:
                c.add(f'hashCountMulti(jht{self.opId},{self.size},hash_key);')
            elif self.num_touched == 1:
                c.add(self.genPayloadBuild(self.lop.leftChild.outRelation))
                c.add(f'hashInsertMulti(jht{self.opId}, jht{self.opId}_payload, jht{self.opId}_offset, {self.size}, hash_key, &payl);')
            else:
                assert(False)
        else:
            c.add(self.genPayloadBuild(self.lop.leftChild.outRelation))
            c.add(f'hashBuildUnique(ujht{self.opId},{self.size},hash_key,&payl);')
        return c
    
    def genCodeAfterExecution(self):
        c = Code()
        if (not self.touched) and self.multimatch and self.num_touched == 0:
            gSize, bSize = KernelCall.defaultGridSize, KernelCall.defaultBlockSize        
            c.add('cudaDeviceSynchronize();')
            c.add(f'scanMultiHT<<<{gSize},{bSize}>>>(jht{self.opId},{self.size},jht{self.opId}_offset);')
        self.num_touched += 1
        return c

class AggHashJoin(HashJoin): # For semi-join and anti-join
    
    def __init__(self, lop, name, touched):
        super().__init__(lop, name, touched)

    def resolveAttributesForProbe(self):
        self.generatingAttrs.update(self.lop.conditionBuildAttributes)
        self.generatingAttrs.update(self.lop.buildKeyAttributes)
        self.usingAttrs.update(self.lop.probeKeyAttributes)
        self.usingAttrs.update(self.lop.conditionProbeAttributes)
        
        # Generate origin expr for data load
        for attrId, attr in self.generatingAttrs.items():
            if attrId == self.tid.id:
                exp = self.tid.id_name
            else:
                exp = f"aht{self.opId}[{self.tid.id_name}].payload.{attr.id_name}"
            self.pipe.originExpr[attrId] = exp

    def resolveAttributesForBuild(self):
        self.usingAttrs.update(self.lop.buildKeyAttributes)
        self.usingAttrs.update(self.lop.conditionBuildAttributes)

    def resolveAttributes(self, next_op):        
        if self.touched: self.resolveAttributesForProbe()
        else: self.resolveAttributesForBuild()
        super().resolveAttributes(next_op)
        
    def resolveDatastructures(self):
        self.dss.add(AggHT(self.opId, self.size, self.lop.buildKeyAttributes, self.lop.conditionBuildAttributes, {}))

    def genLocalVar(self, declaredAttrs):
        c = Code()
        if self.touched:
            c.add(f'int bucketId{self.opId} = -1;')
        return c

    def genProbe(self):
        c = Code()
        ctxt = OperatorContext(self.pipe)
        for attrId, attr in self.lop.conditionBuildAttributes.items():
            ctxt.attrLoc[attrId] = f'aht{self.opId}[bucketId{self.opId}].payload.{attr.id_name}'
        for attrId, attr in self.lop.conditionProbeAttributes.items():
            ctxt.attrLoc[attrId] = attr.id_name
        c.add('int numLookups = 0;')
        c.add('int bucketFound = 0;')
        c.add(f'active = hashAggregateFindBucket(aht{self.opId}, {self.size}, hash_key, numLookups, bucketId{self.opId});')
        c.add('while (active) {')
        c.add('bucketFound = 1;')
        for (attrId_b, attr_b), (attrId_p, attr_p) in zip(list(self.lop.buildKeyAttributes.items()), list(self.lop.probeKeyAttributes.items())):  
            loc_left = f'aht{self.opId}[bucketId{self.opId}].payload.{attr_b.id_name}'
            loc_right = attr_p.id_name
            c.add(f'bucketFound = bucketFound && (({loc_left}) == ({loc_right}));')
        if self.lop.conditions is not None:
            c.add(f'bucketFound = bucketFound && ({self.lop.conditions.gen(ctxt)});')
        c.add('if (bucketFound) break;')
        c.add(f'active = hashAggregateFindBucket(aht{self.opId}, {self.size}, hash_key, numLookups, bucketId{self.opId});')
        c.add('}')
        if self.name == 'antijoin':
            c.add('active = !bucketFound;')
        else:
            c.add('active = bucketFound;')
        return c
    
    def genBuild(self):
        c = Code()
        attrs = {}
        attrs.update(self.lop.buildKeyAttributes)
        attrs.update(self.lop.conditionBuildAttributes)
        c.add('int numLookups = 0;')
        c.add('int bucketFound = 0;')
        c.add(f'int bucketId = -1;')
        c.add('while (!bucketFound) {')
        c.add(f'bucketId = hashAggregateGetBucket(aht{self.opId},{self.size},hash_key,numLookups,&payl);')
        c.add('bucketFound = 1;')
        attrs = {}
        attrs.update(self.lop.buildKeyAttributes)
        attrs.update(self.lop.conditionBuildAttributes)
        for attrId, attr in attrs.items():
            left_loc = f'payl.{attr.id_name}'
            right_loc = f'aht{self.opId}[bucketId].payload.{attr.id_name}'
            if attr.dataType == Type.STRING:
                c.add(f"bucketFound &= stringEquals({left_loc}, {right_loc});")
            else:
                c.add(f"bucketFound &= {left_loc} == {right_loc};")        
        c.add('}')
        return c

class SemiJoin(AggHashJoin):

    def __init__(self, lop, touched):
        super().__init__(lop, 'semijoin', touched)


class AntiJoin(AggHashJoin):

    def __init__(self, lop, touched):
        super().__init__(lop, 'antijoin', touched)


class OuterJoin(HashJoin):

    def __init__(self, lop, touched):
        super().__init__(lop, 'outerjoin', touched)
        
    def resolveAttributes(self, next_op):        
        if self.touched:
            #self.generatingAttrs.update(self.lop.buildKeyAttributes)
            self.generatingAttrs[self.tid.id] = self.tid
            self.generatingAttrs.update(self.lop.leftChild.outRelation)
            self.usingAttrs.update(self.lop.probeKeyAttributes)
            if not self.multimatch:
                self.usingAttrs.update(self.lop.buildKeyAttributes)
                self.usingAttrs.update(self.lop.conditionAttributes)
        else:
            self.usingAttrs.update(self.lop.leftChild.outRelation)
        super().resolveAttributes(next_op)

    def resolveDatastructures(self):
        if self.multimatch:
            self.dss.add(MultiHT(self.opId, self.size, self.lop.leftChild.tupleNum*2, self.lop.leftChild.outRelation))
        else:
            self.dss.add(UniqueHT(self.opId, self.size, self.lop.leftChild.outRelation))

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        if self.touched:
            pass
        else:
            pass
        return c

class CrossJoin(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'crossjoin', touched)
        self.tid = self.lop.tid

    def resolveAttributes(self, next_op):
        if self.touched:
            self.generatingAttrs.update(self.lop.leftChild.outRelation)
            self.generatingAttrs[self.lop.tid.id] = self.lop.tid
            for attrId, attr in self.generatingAttrs.items():
                if attrId == self.tid.id:
                    exp = self.tid.id_name
                else:
                    exp = f"temp{self.opId}_{attr.name}[{self.tid.id_name}]"
                self.pipe.originExpr[attrId] = exp
        else:
            self.usingAttrs.update(self.lop.leftChild.outRelation)
        super().resolveAttributes(next_op)
        
    def resolveDatastructures(self):
        lc = self.lop.leftChild
        self.dss.add(TempTable(f'temp{self.opId}', lc.tupleNum, lc.outRelation))

    def genLocalVar(self, declaredAttrs):
        c = Code()
        if self.touched: # probe
            c.add(f'Range local{self.opId}_range;')
        else: # build
            c.add("int wp, writeMask, numProj;")
            c.add("writeMask = __ballot_sync(ALL_LANES, active);")
            c.add("numProj = __popc(writeMask);")
            c.add("if (thread_id == 0) { ")
            c.add(f"wp = atomicAdd(nout_temp{self.opId}, numProj);")
            c.add("}")
            c.add("wp = __shfl_sync(ALL_LANES, wp, 0);")
            c.add("wp = wp + __popc(writeMask & prefixlanes);")
        return c

    def genProbe(self):
        c = Code()
        tableName = f'temp{self.opId}'
        c.add(f'local{self.opId}_range.start = 0;')
        c.add(f'local{self.opId}_range.end = *nout_temp{self.opId};')
        return c

    def genBuild(self):
        c = Code()
        tableName = f'temp{self.opId}'
        for attrId, attr in self.lop.leftChild.outRelation.items():
            c.add(f"temp{self.opId}_{attr.name}[wp] = {attr.id_name};")
        return c

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        if self.touched: c.add(self.genProbe())
        else: c.add(self.genBuild())
        return c

class Aggregation(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'agg', touched)
        self.tid = lop.tid
        self.size = 1 if not self.lop.doGroup else self.lop.tupleNum * 2
        self.doGroup = self.lop.doGroup

    def resolveAttributes(self, next_op):
        if self.touched:
            self.generatingAttrs.update(self.lop.groupAttributes)
            self.generatingAttrs[self.lop.tid.id] = self.lop.tid
            cntAttr = None
            for attr_id, (attr, _, reductionType) in self.lop.aggregateTuplesCreated.items():
                self.generatingAttrs[attr_id] = attr
                if reductionType == Reduction.COUNT: cntAttr = attr
            for attrId, attr in self.generatingAttrs.items():
                if attrId == self.tid.id:
                    exp = self.tid.id_name
                elif attrId in self.lop.groupAttributes:
                    exp = f"aht{self.opId}[{self.tid.id_name}].payload.{attr.id_name}"
                else:
                    exp = f"aht{self.opId}_{attr.id_name}[{self.tid.id_name}]"
                    _, _, reductionType = self.lop.aggregateTuplesCreated[attrId]
                    if reductionType == Reduction.AVG:
                        exp = f'{exp}/aht{self.opId}_{cntAttr.id_name}[{self.tid.id_name}]'
                        
                self.pipe.originExpr[attrId] = exp
        else:
            self.usingAttrs.update(self.lop.groupAttributes)
            for attr_id, attr in self.lop.aggregateInAttributes.items():
                self.usingAttrs[attr_id] = attr
            
        super().resolveAttributes(next_op)
    
    def resolveDatastructures(self):
        self.dss.add(AggHT(self.opId, self.size, self.lop.groupAttributes, {}, self.lop.aggregateTuplesCreated))
    
    def genPipeVar(self):
        c = Code()
        if not self.touched and not self.doGroup and KernelCall.args.local_aggregation:
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                dType = langType(attr.dataType)
                if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f'{dType} local_{attr.id_name} = {CType.zeroValue[dType]};')
                elif reductionType == Reduction.MAX:
                    c.add(f'{dType} local_{attr.id_name} = {CType.minValue[dType]};')
                elif reductionType == Reduction.MIN:
                    c.add(f'{dType} local_{attr.id_name} = {CType.maxValue[dType]};')
            pass
            # TODO local aggregation
            #for attrId, (attr, _, reductionType) in self.lop.aggregateTuplesCreated.items():
            #    print(attrId, inId)
        return c
    
    def genBuildForGroupBy(self):
        c = Code()
        c.add(f'Payload{self.opId} buf_payl;')    
        c.add('uint64_t hash_key = 0;')

        for attrId, attr in self.lop.groupAttributes.items():
            c.add(f'buf_payl.{attr.id_name} = {attr.id_name};')

        for attrId, attr in self.lop.groupAttributes.items():
            if attr.dataType == Type.STRING:
                c.add(f'hash_key = hash(hash_key + stringHash({attr.id_name}));')
            else:
                c.add(f'hash_key = hash(hash_key + ((uint64_t) {attr.id_name}));')
                
        c.add('int bucketFound = 0;')
        c.add('int numLookups = 0;')
        c.add('int bucketId = -1;')
        
        c.add('while (!bucketFound) {')
        c.add('bucketId = -1;')
        c.add('bool done = false;')
        
        c.add('while (!done) {')
        c.add(f'bucketId = (hash_key + numLookups) % {int(self.size)};')
        c.add(f'agg_ht<Payload{self.opId}>& entry = aht{self.opId}[bucketId];')
        c.add('numLookups++;')
        
        c.add('if (entry.lock.enter()) {')
        c.add('entry.payload = buf_payl;')
        c.add('entry.hash = hash_key;')
        c.add('entry.lock.done();')
        c.add('break;')
        c.add('} else {')
        c.add('entry.lock.wait();')
        c.add('done = (entry.hash == hash_key);')
        c.add('}')
        
        c.add('}')

        c.add(f"Payload{self.opId} entry = aht{self.opId}[bucketId].payload;")
        c.add('bucketFound = 1;')
        for attrId, attr in self.lop.groupAttributes.items():
            if attr.dataType == Type.STRING:
                c.add(f"bucketFound &= stringEquals(entry.{attr.id_name}, {attr.id_name});")
            else:
                c.add(f"bucketFound &= entry.{attr.id_name} == {attr.id_name};")
        
        c.add('}')

        for attrId, (inId, _) in self.lop.aggregateTuples.items():
            attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
            inAttr = self.lop.aggregateInAttributes.get(inId, None)
            dst = f"aht{self.opId}_{attr.id_name}[bucketId]"
            if reductionType == Reduction.COUNT:
                c.add(f"atomicAdd(&{dst},1);")
            elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                c.add(f"atomicAdd(&{dst},{inAttr.id_name});")
            elif reductionType == Reduction.MAX:
                c.add(f"atomicMax(&{dst},{inAttr.id_name});")
            elif reductionType == Reduction.MIN:
                c.add(f"atomicMin(&{dst},{inAttr.id_name});")

        return c
        
    def genBuildForAgg(self):
        c = Code()
        # TODO Local aggregation
        
        if not self.touched and not self.doGroup and KernelCall.args.local_aggregation:
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                inAttr = self.lop.aggregateInAttributes.get(inId, None)
                if reductionType == Reduction.COUNT:
                    c.add(f'local_{attr.id_name} += 1;')
                elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f'local_{attr.id_name} += {inAttr.id_name};')
                elif reductionType == Reduction.MAX:
                    c.add(f'local_{attr.id_name} = local_{attr.id_name} < {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
                elif reductionType == Reduction.MIN:
                    c.add(f'local_{attr.id_name} = local_{attr.id_name} > {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
        else:        
            for attrId, (inId, _) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                inAttr = self.lop.aggregateInAttributes.get(inId, None)
                dst = f"aht{self.opId}_{attr.id_name}[0]"
                if reductionType == Reduction.COUNT:
                    c.add(f"atomicAdd(&{dst},1);")
                elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f"atomicAdd(&{dst},{inAttr.id_name});")
                elif reductionType == Reduction.MAX:
                    c.add(f"atomicMax(&{dst},{inAttr.id_name});")
                elif reductionType == Reduction.MIN:
                    c.add(f"atomicMin(&{dst},{inAttr.id_name});")
        return c
    
    def genPipePostCode(self):
        c = Code()
        if not self.touched and not self.doGroup and KernelCall.args.local_aggregation:
            num_warps_per_block = int(KernelCall.defaultBlockSize / 32)
                
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType) # 
                c.add(f"__shared__ {dType} shared_{local_loc_dst}[{num_warps_per_block}];")
            
            c.add("for (int offset = 16; offset > 0; offset /= 2) {")
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType)
                if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f"{local_loc_dst} += __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")                        
                elif reductionType == Reduction.MAX:
                    c.add("{")
                    c.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                    c.add(f"{local_loc_dst} = {local_loc_dst} < v ? v : {local_loc_dst};")
                    c.add("}")
                elif reductionType == Reduction.MIN:
                    c.add("{")
                    c.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                    c.add(f"{local_loc_dst} = {local_loc_dst} > v ? v : {local_loc_dst};")
                    c.add("}")
            c.add("}")                
            c.add("if (thread_id == 0) {")
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType)
                c.add(f"shared_{local_loc_dst}[warp_id % {num_warps_per_block}] = {local_loc_dst};")
            c.add("}")
            c.add("__syncthreads();")
            c.add(f"if (warp_id % {num_warps_per_block} == 0) ")
            c.add("{")
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType)
                if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.zeroValue[dType]};")                     
                elif reductionType == Reduction.MAX:
                    c.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.minValue[dType]};")
                elif reductionType == Reduction.MIN:
                    c.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.maxValue[dType]};")
            c.add(f"for (int offset = {int(num_warps_per_block/2)}; offset > 0; offset /= 2)")
            c.add("{")
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType)
                if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f"{local_loc_dst} += __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")                        
                elif reductionType == Reduction.MAX:
                    c.add("{")
                    c.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                    c.add(f"{local_loc_dst} = {local_loc_dst} < v ? v : {local_loc_dst}")
                    c.add("}")
                elif reductionType == Reduction.MIN:
                    c.add("{")
                    c.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                    c.add(f"{local_loc_dst} = {local_loc_dst} > v ? v : {local_loc_dst}")
                    c.add("}")
            c.add("}")
            c.add("}")
                
            c.add("if (threadIdx.x == 0) {")
            for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
                attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
                loc_dst = f"aht{self.opId}_{attr.id_name}[0]"
                local_loc_dst = 'local_{}'.format(attr.id_name)
                dType = langType(attr.dataType)
                if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                    c.add(f"atomicAdd(&{loc_dst},{local_loc_dst});")
                elif reductionType == Reduction.MAX:
                    c.add(f"atomicMax(&{loc_dst},{local_loc_dst});")
                elif reductionType == Reduction.MIN:
                    c.add(f"atomicMin(&{loc_dst},{local_loc_dst});")
            c.add("}")
        return c

    def genBuild(self):
        c = Code()
        if self.doGroup: return self.genBuildForGroupBy()
        else: return self.genBuildForAgg()
    
    def genProbe(self):
        c = Code()
        tid = "0"
        if self.doGroup: 
            tid = self.tid.id_name
            #c.add(f"active = aht{self.opId}[{tid}].lock.lock == OnceLock::LOCK_DONE;")
        else:
            pass
        # TODO
        return c

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        if self.touched: c.add(self.genProbe())
        else: c.add(self.genBuild())
        return c
    
    def genTableSize(self):
        return str(self.size)


class Materialize(PhysicalOperator):
    
    def __init__(self, lop, touched):
        super().__init__(lop, 'materialize', touched)
        self.tableName = "result" if self.lop.isResult else self.lop.table['name']
        
    def resolveAttributes(self, next_op):
        self.usingAttrs.update(self.lop.inRelation)
        super().resolveAttributes(next_op)

    def resolveDatastructures(self):
        if self.lop.isResult:
            self.dss.add(ResultTable(self.tableName, self.lop.tupleNum, self.lop.inRelation))
        else:
            self.dss.add(TempTable(self.tableName, self.lop.tupleNum, self.lop.inRelation))
        pass
    
    def genLocalVar(self, declaredAttrs):
        c = Code()
        c.add("int wp, writeMask, numProj;")
        c.add("writeMask = __ballot_sync(ALL_LANES, active);")
        c.add("numProj = __popc(writeMask);")
        c.add(f"if (thread_id == 0) wp = atomicAdd(nout_{self.tableName}, numProj);")
        c.add("wp = __shfl_sync(ALL_LANES, wp, 0);")
        c.add("wp = wp + __popc(writeMask & prefixlanes);")
        return c

    def genOperation(self):
        c = Code()
        c.add('// ' + str(self))
        for attrId, attr in self.lop.inRelation.items():
            if self.lop.isResult:
                c.add(f"{self.tableName}_{attr.id_name}[wp] = {attr.id_name};")
            else:
                c.add(f"{self.tableName}_{attr.name}[wp] = {attr.id_name};")
        return c
    
    #def resolveDatastructures(self):
    #    self.dss.add(AggHT(self.opId, self.size, self.lop.groupAttributes, {}, self.lop.aggregateTuplesCreated))