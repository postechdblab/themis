from dogqc.kernel import KernelCall

from themis.operator import Scan, IndexJoin, MWayIndexJoin, Exist, Selection, Map, MultiMap, InnerJoin, SemiJoin, OuterJoin, AntiJoin, CrossJoin, Aggregation, AggSelection, Materialize
from themis.datastructure import Datastructures
from themis.code import Code

class Subpipe:
    
    def __init__(self, subpipe_id, pipe, operators):
        self.id = subpipe_id
        self.pipe = pipe
        self.operators = operators
        self.inputType = -1 # 0: scan, 1: multi, 2: filter
        self.outputType = -1 # 0: materialize, 1: multi, 2: filter
        self.doLBforInput = False
        self.doLBforOutput = False
        self.inAttrs = operators[0].inAttrs
        self.outAttrs = operators[-1].outAttrs
        for op in self.operators:
            op.pipe = pipe
            op.subpipe = self
        
    def __str__(self):        
        inAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.inAttrs.values()))) + ']'
        outAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.outAttrs.values()))) + ']'
        return f"spid: {self.id}, inType: {self.inputType}/{self.doLBforInput}, outType: {self.outputType}/{self.doLBforOutput}, attrs: {inAttrs}/{outAttrs}" # ops: {self.operators}"
    
    def resolveInputOutputType(self):
        if self.id == 0:
            self.inputType = 0
        else:
            prev_subpipe = self.pipe.subpipes[self.id-1]
            self.inputType = prev_subpipe.outputType
            self.doLBforInput = prev_subpipe.doLBforOutput
            
        if self.id + 1 == len(self.pipe.subpipes):
            self.outputType = 0
        else:
            if self.operators[-1].lop.outputCard == 'm':
                self.outputType = 1
            else:
                self.outputType = 2
            self.doLBforOutput = self.operators[-1].lop.doLB

    def resolveAttributes(self, next_op):
        for idx in range(len(self.operators)):
            op = self.operators[len(self.operators)-1-idx]
            op.initAttrs()
            op.resolveAttributes(next_op)
            next_op = op
        self.inAttrs = self.operators[0].inAttrs
        self.outAttrs = self.operators[-1].outAttrs
        self.usingAttrs = {}
        self.generatingAttrs = {}
        for op in self.operators:
            self.usingAttrs.update(op.usingAttrs)
            self.generatingAttrs.update(op.generatingAttrs)

    def resolveDatastructures(self):
        self.dss = Datastructures()
        for op in self.operators:
            op.resolveDatastructures()
            self.dss.update(op.dss)

    def getUsingAttrs(self):
        attrs = {}
        for op in self.operators:
            attrs.update(op.usingAttrs)
        return attrs

    def getGeneratingAttrs(self):
        attrs = {}
        for op in self.operators:
            attrs.update(op.generatingAttrs)
        return attrs
    
    def getMappedAttrs(self):
        attrs = {}
        for op in self.operators:
            attrs.update(op.mappedAttrs)
        return attrs        

    def genCode(self):
        codesVariableLoad = [ op.genVariableLoad() for op in self.operators ]
        codesOperation = [ op.genOperation() for op in self.operators ]
        return codeVariableLoad, codesOperation
    
    def genLocalVar(self, declaredAttrs):
        c = Code()
        for op in self.operators:
            c.add(op.genLocalVar(declaredAttrs))
        return c
    
    def genOperation(self):
        codes = []
        for op in self.operators:
            codes.append(op.genOperation())
        return codes
    
    def genPipeVar(self):
        c = Code()
        for op in self.operators:
            c.add(op.genPipeVar())
        return c

    def genCodeBeforeExecution(self):
        c = Code()
        for op in self.operators:
            c.add(op.genCodeBeforeExecution())
        return c
    
    def genCodeAfterExecution(self):
        c = Code()
        for op in self.operators:
            c.add(op.genCodeAfterExecution())
        return c

class SubpipeSequence:
    
    def __init__(self, pipe, id):
        self.id = id
        self.subpipes = []
        self.pipe = pipe
        self.inputType = None
        self.outputType = None
        self.doLBforInput = None
        self.doLBforOutput = None
        self.inAttrs = None
        self.outAttrs = None
        self.inBoundaryAttrs = {}
        self.outBoundaryAttrs = {}

    def __str__(self):
        inAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.inAttrs.values()))) + ']'
        outAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.outAttrs.values()))) + ']'
        inBAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.inBoundaryAttrs.values()))) + ']'
        outBAttrs = '[' + ','.join(list(map(lambda x: str(x.id), self.outBoundaryAttrs.values()))) + ']'

        return f"inType: {self.inputType}/{self.doLBforInput}, outType: {self.outputType}/{self.doLBforOutput}, attrs: {inAttrs}/{outAttrs}, battrs: {inBAttrs}/{outBAttrs}"

    def resolveInputOutputType(self):
        self.inputType = self.subpipes[0].inputType
        self.outputType = self.subpipes[-1].outputType
        self.doLBforInput = self.subpipes[0].doLBforInput
        self.doLBforOutput = self.subpipes[-1].doLBforOutput        
        self.inAttrs = self.subpipes[0].operators[0].inAttrs
        self.outAttrs = self.subpipes[-1].operators[-1].outAttrs

    def getTid(self):
        if self.inputType == 0:
            return self.subpipes[0].operators[0].tid
        elif self.inputType == 1:
            return self.pipe.subpipeSeqs[self.id-1].subpipes[-1].operators[-1].tid
        else:
            return None
        
    def convertTid(self, var):
        prevOp = self.pipe.subpipeSeqs[self.id-1].subpipes[-1].operators[-1]
        if isinstance(prevOp, IndexJoin):
            return prevOp.genConvertTid(var)
        return var            

    def genOperation(self):
        codes = []
        for sp in self.subpipes:
            codes.append(sp.genOperation())
        return codes
    
    def resolveBoundaryAttrs(self):
        self.inAttrs = self.subpipes[0].operators[0].inAttrs
        self.outAttrs = self.subpipes[-1].operators[-1].outAttrs
        
        if KernelCall.args.lazy_materialization:
            for attrId, attr in self.outAttrs.items():
                if attrId in self.pipe.attr2tid:
                    tid = self.pipe.tids[self.pipe.attr2tid[attrId]]
                    self.outBoundaryAttrs[tid.id] = tid
                else:
                    self.outBoundaryAttrs[attrId] = attr
            for attrId, attr in self.inAttrs.items():
                if attrId in self.pipe.attr2tid:
                    tid = self.pipe.tids[self.pipe.attr2tid[attrId]]
                    self.inBoundaryAttrs[tid.id] = tid
                else:
                    self.inBoundaryAttrs[attrId] = attr               
        else:
            self.inBoundaryAttrs.update(self.inAttrs)
            self.outBoundaryAttrs.update(self.outAttrs)
            

class Pipe:
    
    def __init__(self, pipe_id):
        self.id = pipe_id
        self.subpipes = []
        self.stringConstantsDic = {}
        self.dss = Datastructures()
        self.subpipeSeqs = []
        self.attr2tid = {}
        self.tids = {}
        self.originExpr = {}
        
        
    def resolveSubpipeSeqs(self):
        spSeqId = 0
        self.subpipeSeqs.append(SubpipeSequence(self, spSeqId))
        spSeqId += 1
        for idx, sp in enumerate(self.subpipes):
            self.subpipeSeqs[-1].subpipes.append(sp)            
            if (idx + 1) < len(self.subpipes) and (sp.outputType == 1 or sp.doLBforOutput):
                self.subpipeSeqs.append(SubpipeSequence(self, spSeqId))
                spSeqId += 1
    
        for subpipeSeq in self.subpipeSeqs:
            subpipeSeq.resolveInputOutputType()    
    
    def resolveInputOutputTypeOfSubpipes(self):
        for subpipe in self.subpipes:
            subpipe.resolveInputOutputType()
            
    def resolveAttributes(self):
        next_op = None
        for idx in range(len(self.subpipes)):
            subpipe = self.subpipes[len(self.subpipes)-1-idx]
            subpipe.resolveAttributes(next_op)
            next_op = subpipe.operators[0]
                
    def resolveDatastructures(self):
        for sp in self.subpipes:
            sp.resolveDatastructures()
            self.dss.update(sp.dss)

    def resolveBoundaryAttrs(self):
        for spSeq in self.subpipeSeqs:
            spSeq.resolveBoundaryAttrs()

    def stringConstants(self, token):
        if token not in self.stringConstantsDic:
            self.stringConstantsDic[token] = len(self.stringConstantsDic)
        return "string_constant{}".format(self.stringConstantsDic[token])

    def genStringConstants(self):
        c = Code()
        for token, token_id in self.stringConstantsDic.items():
            c.add(f'str_t string_constant{token_id} = stringConstant("{token}",{len(token)});')
        return c

    def genOperation(self):
        codes = []
        for spSeq in self.subpipeSeqs:
            codes.append(spSeq.genOperation())
        return codes
    
    def genPipeVar(self):
        c = Code()
        for sp in self.subpipes:
            c.add(sp.genPipeVar())
        return c
    
    def genArgsForKernelDeclation(self):
        return self.dss.genArgsForKernelDeclation()
    
    def genArgsForKernelCall(self):
        return self.dss.genArgsForKernelCall()
    
    def genCodeBeforeExecution(self):
        c = Code()
        for sp in self.subpipes:
            c.add(sp.genCodeBeforeExecution())
        return c
    
    def genCodeAfterExecution(self):
        c = Code()
        for sp in self.subpipes:
            c.add(sp.genCodeAfterExecution())
        return c
    
    def isLastOperatorMultiHashJoin(self):
        lastOp = self.subpipes[-1].operators[-1]
        if isinstance(lastOp, InnerJoin) and lastOp.multimatch:
            return True
        return False

operators_dict = {
    'indexjoin': IndexJoin,
    'mwayindexjoin': MWayIndexJoin,
    'exist': Exist,
    'scan': Scan,
    'selection': Selection,
    'map': Map,
    'multimap': MultiMap,
    'equijoin': InnerJoin,
    'semijoin': SemiJoin,
    'antijoin': AntiJoin,
    'outerjoin': OuterJoin,
    'crossjoin': CrossJoin,
    'aggregation': Aggregation,
    'aggselection': AggSelection,
    'materialize': Materialize
}

def genPipesForPlan(plan):

    info = {}
    info['pipelines'] = []
    for node in plan:
        info['pipelines'].append([[]])
        node.genPipes(info)
    p_pipes = []
    op_ids = set([])
    for pid, pipe in enumerate(info['pipelines']):
        p_pipes.append([])
        prev_pop = None
        for spid, subpipe in enumerate(pipe):
            p_pipes[-1].append([])
            for opid, lop in enumerate(subpipe):
                pop = None
                touched = lop.opId in op_ids
                op_ids.add(lop.opId)
                pop = operators_dict[lop.opType](lop, touched)
                p_pipes[-1][-1].append(pop)
                pop.prev = prev_pop
                prev_pop = pop
    
    dss = Datastructures()
    pipes = []
    for pid, subpipes in enumerate(p_pipes):
        p = Pipe(pid)
        for spid, operators in enumerate(subpipes):
            p.subpipes.append(Subpipe(spid, p, operators))
        p.resolveInputOutputTypeOfSubpipes()
        p.resolveDatastructures()
        p.resolveSubpipeSeqs()
        dss.update(p.dss)
        pipes.append(p)          
    return dss, pipes
