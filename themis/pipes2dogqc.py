import sys
import subprocess
import time

from dogqc.kernel import KernelCall
from dogqc.types import Type
from dogqc.cudalang import CType

from themis.datastructure import langType
from themis.code import Code
from themis.operator import Selection, Aggregation, AggSelection, HashJoin

from themis.pipes2naive import codeGenerator as naiveCodeGenerator

class codeGenerator(naiveCodeGenerator):
    
    def __init__(self, dss, pipes):
        super().__init__(dss, pipes)
        self.doIntraWarpLB = True
        self.doCnt = True
            
    def genPipeIntialization(self, pipe):
        c = Code()
        c.add('int thread_id = threadIdx.x % 32;')
        c.add('int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;')
        c.add('unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));')
        c.add('int active = 0;')
        c.add(f'int bufferBase = (threadIdx.x / 32) * 32;')
        
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            if spSeq.inputType == 1:
                c.add(f'Range ts_{spSeqId}_range;')
                c.add(f'Range ts_{spSeqId}_range_cached;')
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name};')
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name}_cached;') 
            elif spSeq.inputType == 2:
                c.add(f'int buffercount{spSeqId} = 0;')
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    c.add(f'__shared__ {langType(attr.dataType)} ts_{spSeqId}_{attr.id_name}_flushed[{KernelCall.defaultBlockSize}];')
        return c
    
    def genInitialDistribution(self, pipe):
        c = Code()
        scan = pipe.subpipes[0].operators[0]
        c.add(f'Range ts_0_range_cached;')
        c.add(f'ts_0_range_cached.start = blockIdx.x * blockDim.x + threadIdx.x;')
        c.add(f'ts_0_range_cached.end = {scan.genTableSize()};')
        return c
    
    def genInputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        stepSize = KernelCall.defaultBlockSize * KernelCall.defaultGridSize
        c.add(f'unsigned flushPipeline0 = 0;')
        c.add('while (!flushPipeline0) {')
        c.add(f'int loop{spSeq.id} = ts_{spSeq.id}_range_cached.start;')
        c.add(f'active = loop{spSeq.id} < ts_{spSeq.id}_range_cached.end;')
        c.add(f'ts_{spSeq.id}_range_cached.start += {stepSize};')
        #c.add(f'active_mask = __ballot_sync(ALL_LANES, active);')
        #c.add(f'if (active_mask == 0) break;')
        tid = spSeq.getTid()
        c.add(f'int {tid.id_name} = loop{spSeq.id};')
        c.add(f'flushPipeline0 = !(__ballot_sync(ALL_LANES, ts_{spSeq.id}_range_cached.start < ts_{spSeq.id}_range_cached.end));')
        return c
    
    def genInputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        tid = spSeq.getTid()
        lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
        c = Code()
        
        # if len(spSeq.inBoundaryAttrs) > 0:
        #     c.add('if (active) {')
        #     for attrId, attr in spSeq.inBoundaryAttrs.items():
        #         if attrId == tid.id: continue
        #         if attrId in lastOp.generatingAttrs: continue
        #         c.add(f'ts_{spSeq.id}_{attr.id_name} = {attr.id_name};')
        #     c.add('}')
        
        c.add(f'int probeActive{spSeq.id} = active;')
        c.add(f'unsigned activeProbe{spSeq.id} = __ballot_sync(ALL_LANES, probeActive{spSeq.id});')
        c.add(f'int num{spSeq.id} = ts_{spSeq.id}_range.end - ts_{spSeq.id}_range.start;')
        c.add(f'while (activeProbe{spSeq.id} > 0) ' + '{')
        c.add(f'unsigned tupleLane = __ffs(activeProbe{spSeq.id})-1;')
        c.add(f'ts_{spSeq.id}_range_cached.start = __shfl_sync(ALL_LANES, ts_{spSeq.id}_range.start, tupleLane) + thread_id;')
        c.add(f'ts_{spSeq.id}_range_cached.end = __shfl_sync(ALL_LANES, ts_{spSeq.id}_range.end, tupleLane);')
        c.add(f'activeProbe{spSeq.id} -= (1 << tupleLane);')
        c.add(f'probeActive{spSeq.id} = ts_{spSeq.id}_range_cached.start < ts_{spSeq.id}_range_cached.end;')

        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId in lastOp.generatingAttrs: continue
            c.add('{')
            name = f'ts_{spSeq.id}_{attr.id_name}'
            if attr.dataType == Type.STRING:
                c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, tupleLane);')
                c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, tupleLane);')
                c.add(f'{name}_cached.start = start;')
                c.add(f'{name}_cached.end = end;')
            else:
                c.add(f'{name}_cached = __shfl_sync(ALL_LANES, {name}, tupleLane);')
            c.add('}')

        c.add(f'while(__any_sync(ALL_LANES, probeActive{spSeq.id}))' + '{')
        c.add(f'active = probeActive{spSeq.id};')
        c.add(f'int loopvar{spSeq.id} = ts_{spSeq.id}_range_cached.start;')
        c.add(f'ts_{spSeq.id}_range_cached.start += 32;')
        c.add(f'probeActive{spSeq.id} = probeActive{spSeq.id} && (ts_{spSeq.id}_range_cached.start < ts_{spSeq.id}_range_cached.end);')
        c.add(f'unsigned flushPipeline{spSeq.id} = !__ballot_sync(ALL_LANES, probeActive{spSeq.id});')
                
        # Declare in-boundary attrs
        c.add(f'//{list(attrsToDeclareAndMaterialize)}')
        c.add(f'int {tid.id_name};')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            c.add(f'{langType(attr.dataType)} {attr.id_name};')
            
        
        _, attrsToLoadBeforeExec, _, _ = attrsToDeclareAndMaterialize[0]
        c.add(self.genAttrDeclaration(spSeq.pipe, attrsToLoadBeforeExec))
        
        c.add('if (active) {')

        # Load in-boundary attrs
        c.add(f'// last op generating Attrs: {lastOp.generatingAttrs}')
        var = f'loopvar{spSeq.id}'
        c.add(f'{tid.id_name} = {spSeq.convertTid(var)};')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId in lastOp.generatingAttrs:
                c.add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
            else:
                c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_cached;')
            
        c.add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

        c.add('}')
        return c
    
    def genInputCodeForType2(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        
        c.add(f'unsigned activemask{spSeq.id} = __ballot_sync(ALL_LANES, active);')
        c.add(f'int numactive{spSeq.id} = __popc(activemask{spSeq.id});')
        c.add(f'int scan{spSeq.id}, remaining{spSeq.id}, bufIdx{spSeq.id};')
        
        loopLvl = self.findLowerLoopLvl(spSeq)
        c.add(f'int minTuplesInFlight{spSeq.id} = flushPipeline{loopLvl} ? 0 : 31;')
        
        c.add(f'while((buffercount{spSeq.id} + numactive{spSeq.id}) > minTuplesInFlight{spSeq.id})' + '{')
        
        c.add(f'if ((numactive{spSeq.id} < 32) && buffercount{spSeq.id})' + '{')
        c.add(f'remaining{spSeq.id} = max(((buffercount{spSeq.id} + numactive{spSeq.id}) - 32), 0);')
        c.add(f'scan{spSeq.id} = __popc((~(activemask{spSeq.id}) & prefixlanes));')
        c.add(f'if((!(active) && (scan{spSeq.id} < buffercount{spSeq.id})))' + '{')
        c.add(f'bufIdx{spSeq.id} = (remaining{spSeq.id} + (scan{spSeq.id} + bufferBase));')
        
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_flushed[bufIdx{spSeq.id}];') 
        c.add('active = 1;')
        c.add('}')
        c.add(f'buffercount{spSeq.id} = remaining{spSeq.id};')
        c.add('}')
        return c

    def genOutputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        lastOp = spSeq.subpipes[-1].operators[-1]
        opId = lastOp.opId
        tid = lastOp.tid
        #attrsGeneratedByLastOp = lastOp.generatingAttrs
        
        currentMaterializedAttrs = attrsToDeclareAndMaterialize[-1][-1]

        attrsToDeclare = {}
        for attrId, attr in spSeq.outBoundaryAttrs.items():
            if attrId in lastOp.generatingAttrs: continue
            if attrId in currentMaterializedAttrs: continue
            attrsToDeclare[attrId] = attr
        
        c.add(self.genAttrDeclaration(spSeq.pipe, attrsToDeclare))
        c.add(f'//{currentMaterializedAttrs}')
        c.add('if (active) {')
        c.add(f'ts_{spSeq.id+1}_range.set(local{opId}_range);')
        
        for attrId, attr in spSeq.outBoundaryAttrs.items():
            if attrId in lastOp.generatingAttrs: continue
            if attrId in currentMaterializedAttrs:
                c.add(f'ts_{spSeq.id+1}_{attr.id_name} = {attr.id_name};')
            elif attrId in spSeq.inBoundaryAttrs:
                c.add(f'ts_{spSeq.id+1}_{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
            else:
                c.add(f'ts_{spSeq.id+1}_{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
        c.add('}')
        return c
    
    def genOutputCodeForType2(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        
        lastOp = spSeq.subpipes[-1].operators[-1]        
        if len(spSeq.outBoundaryAttrs) > 0:
            currentMaterializedAttrs = attrsToDeclareAndMaterialize[-1][-1]
            c.add(f'// {currentMaterializedAttrs}')
            attrs = {}
            for attrId, attr in spSeq.outBoundaryAttrs.items():
                if attrId in currentMaterializedAttrs: continue
                #if isinstance(lastOp, HashJoin) and attrId == lastOp.tid.id: continue 
                attrs[attrId] = attr            
            c.add(self.genAttrDeclaration(spSeq.pipe, attrs))
            if len(attrs) > 0:
                c.add('if (active) {')
                for attrId, attr in attrs.items():
                    if attrId in spSeq.inBoundaryAttrs:
                        c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
                    else:
                        c.add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
                c.add('}')
        return c

    def genOutputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        return c

    def genKernelCodeForSpSeq(self, spSeq, attrsToDeclareAndMaterialize, innerCode):
        c = Code()
        c.add(self.genInputCode(spSeq, attrsToDeclareAndMaterialize))
        if spSeq.inputType != 0:
            c.add("// Materialize attrs: [" + ",".join(list(map(lambda x: str(x), spSeq.inBoundaryAttrs.keys()))) + "]")

        spCodes = spSeq.genOperation()
        c.add(f"//subpipe seq {spSeq.id} {spSeq}")        
        for spId, subpipe in enumerate(spSeq.subpipes):
            c.add(f"// sp {subpipe}")
            opCodes = spCodes[spId]
            
            attrsToDeclare, attrsToLoadBeforeExec, attrsToMaterialize, _ = attrsToDeclareAndMaterialize[spId]
            
            # Declare attrs
            if (spSeq.inputType == 1) and (spId == 0):
                # Exclude attrs we declared already when we load inBoundary attributes
                attrs = {}
                for attrId, attr in attrsToDeclare.items():
                    if attrId not in attrsToLoadBeforeExec:
                        attrs[attrId] = attr
                attrsToDeclare = attrs
                
            c.add('//Declare attributesq')
            c.add(self.genAttrDeclaration(spSeq.pipe, attrsToDeclare))
            
            # Declare local variables
            c.add('//gen local var')
            c.add(subpipe.genLocalVar(attrsToDeclare))
            c.add('if (active) {')
            
            # Load attrs which we can load before the execution of this subpipe
            if (spSeq.inputType == 1) and (spId == 0): pass
            else: c.add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

            for opId, op in enumerate(subpipe.operators):
                c.add(self.genAttrLoad(spSeq.pipe, attrsToMaterialize[opId]))
                c.add(opCodes[opId])
            c.add('} // end of active')
        c.add(self.genOutputCode(spSeq, attrsToDeclareAndMaterialize))
        c.add(innerCode)
        if spSeq.inputType == 0:
            c.add('}')            
        elif spSeq.inputType == 1:
            c.add('}\n} // End of loop level')
        elif spSeq.inputType == 2:
            # TODO: 
            c.add('active = 0;')
            c.add(f'activemask{spSeq.id} = numactive{spSeq.id} = 0;')
            c.add('}')
            c.add(f'if (numactive{spSeq.id} > 0)' + '{')
            c.add(f'scan{spSeq.id} = __popc(activemask{spSeq.id} & prefixlanes) + buffercount{spSeq.id};')
            c.add(f'bufIdx{spSeq.id} = bufferBase + scan{spSeq.id};')
            c.add('if (active) {')
            for attrId, attr in spSeq.inBoundaryAttrs.items():
                c.add(f'ts_{spSeq.id}_{attr.id_name}_flushed[bufIdx{spSeq.id}] = {attr.id_name};') 
            c.add('}')
            c.add('__syncwarp();')
            c.add(f'buffercount{spSeq.id} += numactive{spSeq.id};')
            c.add('active = 0;')
            c.add('} // End of if level')
            #c.add('} // End of if level')
        return c


def compileCode( filename, code, compileOption, arch="sm_75", debug=False):
    codeGen = codeGenerator(None, None)
    codeGen.compileCode( filename, code, compileOption, arch, debug)

def genCode(dss, pipes):
    codeGen = codeGenerator(dss, pipes)
    return codeGen.genCode()