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
        
        num_warps_per_block = int(KernelCall.defaultBlockSize / 32)
            
        self.doGridShuffle = KernelCall.args.pyper_grid_threshold > 0
        self.gridShuffleThreshold = KernelCall.args.pyper_grid_threshold
        
        self.doBlockShuffle = num_warps_per_block > 1
        if num_warps_per_block > 2:
            self.blockShuffleThreshold = 24
            self.blockShuffleNumWarps = 2
        else:
            self.blockShuffleThreshold = 16
            self.blockShuffleNumWarps = 1

    def genArgsForKernelDeclarationForPipe(self, pipe_name, pipe):
        c = super().genArgsForKernelDeclarationForPipe(pipe_name, pipe)
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            c.add(f'int* global_{spSeqId}_lock,int* global_{spSeqId}_num, // {len(spSeq.inBoundaryAttrs)}')
            tid = None
            if spSeq.inputType == 0: tid = spSeq.subpipes[0].operators[0].tid
            elif spSeq.inputType == 1: tid = pipe.subpipeSeqs[spSeqId-1].subpipes[-1].operators[-1].tid
            if tid is not None: c.add(f'volatile int* global_{spSeqId}_{tid.id_name},')
            for attrId, attr in spSeq.inBoundaryAttrs.items():
                if tid is not None and tid.id == attrId: continue
                c.add(f'volatile {langType(attr.dataType)}* global_{spSeqId}_{attr.id_name},')
        c.add('bool isLast,')
        return c
    
    def genArgsForKernelCallForPipe(self, pipe_name, pipe):
        c = super().genArgsForKernelCallForPipe(pipe_name, pipe)
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            c.add(f'global{pipe_name}_{spSeqId}_lock,global{pipe_name}_{spSeqId}_num,')
            tid = None
            if spSeq.inputType == 0: tid = spSeq.subpipes[0].operators[0].tid
            elif spSeq.inputType == 1: tid = pipe.subpipeSeqs[spSeqId-1].subpipes[-1].operators[-1].tid
            if tid is not None: c.add(f'global{pipe_name}_{spSeqId}_{tid.id_name},')
            for attrId, attr in spSeq.inBoundaryAttrs.items():
                if tid is not None and tid.id == attrId: continue
                c.add(f'global{pipe_name}_{spSeqId}_{attr.id_name},')
        c.add('isLast,')
        return c
    
    def genBufferDeclarationInMainForPipe(self, pipe_name, pipe):
        c = Code()
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            name = f'global{pipe_name}_{spSeqId}'
            c.add(f'int* {name}_lock;')
            c.add(f'cudaMalloc((void**) &{name}_lock, sizeof(int));')
            c.add(f'cudaMemset(&{name}_lock, 0, sizeof(int));')    
            c.add(f'int* {name}_num;')
            c.add(f'cudaMalloc((void**) &{name}_num, sizeof(int));')
            c.add(f'cudaMemset(&{name}_num, 0, sizeof(int));')     

            tid = None
            if spSeq.inputType == 0: tid = spSeq.subpipes[0].operators[0].tid
            elif spSeq.inputType == 1: tid = pipe.subpipeSeqs[spSeqId-1].subpipes[-1].operators[-1].tid
            if tid is not None: 
                c.add(f'int* {name}_{tid.id_name};')
                c.add(f'cudaMalloc((void**) &{name}_{tid.id_name}, sizeof(int) * {KernelCall.defaultBlockSize});')

            for attrId, attr in spSeq.inBoundaryAttrs.items():
                if tid is not None and tid.id == attrId: continue
                dtype = langType(attr.dataType)
                if attr.dataType == Type.STRING: dtype = 'str_t'
                c.add(f'{dtype}* {name}_{attr.id_name};')
                c.add(f'cudaMalloc((void**) &{name}_{attr.id_name}, sizeof({dtype}) * {KernelCall.defaultBlockSize});')
        return c

    def genDeclarationInMain(self):
        c = super().genDeclarationInMain()
        pid = 0
        for pipe in self.pipes:
            c.add(self.genBufferDeclarationInMainForPipe(pid, pipe))
            pid += 1
            if pipe.isLastOperatorMultiHashJoin():
                c.add(self.genBufferDeclarationInMainForPipe(pid, pipe))
                pid += 1
        return c

    def genPipeIntialization(self, pipe):
        c = Code()
        c.add('__shared__ int shmActThreadsB;')
        c.add('__shared__ int shmNumWarps;')
        c.add('__shared__ int shmNumInBuffer;')
        numWarpsPerBlock = int(KernelCall.defaultBlockSize / 32)
        c.add(f'__shared__ unsigned shmWarpActive[{numWarpsPerBlock}];')

        c.add('int thread_id = threadIdx.x % 32;')
        c.add('int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;')
        c.add('unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));')
        c.add('int active = isLast;')
        c.add('bool keepGoing = isLast;')
        return c
    
    def genInitialDistribution(self, pipe):
        c = Code()
        scan = pipe.subpipes[0].operators[0]
        c.add(f'int scan_offset = blockIdx.x * blockDim.x;')
        c.add(f'int scan_table_size = isLast ? 0 : {scan.genTableSize()};')
        
        return c
    
    def genInputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        stepSize = KernelCall.defaultBlockSize * KernelCall.defaultGridSize
        tid = spSeq.getTid()
        c = Code()
        c.add('while (true) {')
        c.add(f'if ((isLast && !keepGoing) ||  (!isLast && scan_offset >= scan_table_size)) break;')
        c.add(f'int {tid.id_name} = scan_offset + threadIdx.x;')
        c.add(f'active = !isLast && ({tid.id_name} < scan_table_size);')
        c.add(f'scan_offset += {stepSize};')
        return c
    
    def genGridShuffleCode(self, spSeq, attrsToShuffle):
        c = Code()
        c.add('if (thread_id == 0) shmWarpActive[threadIdx.x / 32] = warpActive;')
        c.add('__syncthreads();')
        c.add('if (active) {')
        c.add('int dst = 0;')
        c.add('int subwarp_id = threadIdx.x / 32;')
        c.add('for (int i = 0; i < subwarp_id; ++i) dst += __popc(shmWarpActive[i]);')
        c.add('dst += __popc(warpActive & prefixlanes);')
        
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst].start = {attr.id_name}.start;')
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst].end = {attr.id_name}.end;')
            else: # attr.dataType == Type.INT:
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst] = {attr.id_name};')
        c.add('} // active')
        
        c.add('// Get the lock of the global memory buffer')
        c.add('if (threadIdx.x == 0) {')
        c.add('if (!isLast) {')
        c.add(f'while (0 != atomicCAS(global_{spSeq.id}_lock, 0, warp_id+1));')
        c.add(f'shmNumInBuffer = atomicCAS(global_{spSeq.id}_num, -1, -1);')
        c.add('} else {')
        c.add(f'shmNumInBuffer = *global_{spSeq.id}_num;')
        c.add('}')
        c.add('}')
        c.add('__syncthreads();')
        
        c.add('int numInBuffer = shmNumInBuffer;')
        c.add('int newNumInBuffer;')
        c.add(f'if ((!isLast) && (numInBuffer + actThreadsB < {KernelCall.defaultBlockSize})) ' + '{ // push tuples')
        c.add('if (threadIdx.x < actThreadsB) {')
        c.add('int dst = numInBuffer+threadIdx.x;')
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'global_{spSeq.id}_{attr.id_name}[dst].start = shm{spSeq.id}_{attr.id_name}[threadIdx.x].start;')
                c.add(f'global_{spSeq.id}_{attr.id_name}[dst].end = shm{spSeq.id}_{attr.id_name}[threadIdx.x].end;')
            else: # attr.dataType == Type.INT:
                c.add(f'global_{spSeq.id}_{attr.id_name}[dst] = shm{spSeq.id}_{attr.id_name}[threadIdx.x];')
        c.add('__threadfence();')
        c.add('}')        
        c.add('newNumInBuffer = numInBuffer + actThreadsB;')
        c.add('active = false;')
        c.add('} else { // pull tuples')
        c.add('if (numInBuffer > 0) {')
        c.add(f'int num_to_pull = ({KernelCall.defaultBlockSize} - actThreadsB) < numInBuffer ? ({KernelCall.defaultBlockSize} - actThreadsB) : numInBuffer;')
        c.add('if (threadIdx.x < actThreadsB) {')
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'{attr.id_name}.start = shm{spSeq.id}_{attr.id_name}[threadIdx.x].start;')
                c.add(f'{attr.id_name}.end = shm{spSeq.id}_{attr.id_name}[threadIdx.x].end;')
            else: # attr.dataType == Type.INT:
                c.add(f'{attr.id_name} = shm{spSeq.id}_{attr.id_name}[threadIdx.x];')
        c.add('} else if (threadIdx.x < (actThreadsB + numInBuffer)) {')
        c.add('int src = numInBuffer - num_to_pull + threadIdx.x - actThreadsB;')
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'{attr.id_name}.start = global_{spSeq.id}_{attr.id_name}[src].start;')
                c.add(f'{attr.id_name}.end = global_{spSeq.id}_{attr.id_name}[src].end;')
            else: # attr.dataType == Type.INT:
                c.add(f'{attr.id_name} = global_{spSeq.id}_{attr.id_name}[src];')
        c.add('}')
        c.add('active = threadIdx.x < (actThreadsB + numInBuffer);')
        c.add('newNumInBuffer = numInBuffer - num_to_pull;')
        c.add('keepGoing = false;')
        c.add('} else {')
        c.add('newNumInBuffer = numInBuffer;')
        c.add('}')
        c.add('} // push or pull tuples')
        
        c.add('if (threadIdx.x == 0) {')
        c.add('if (!isLast) {')
        c.add(f'atomicCAS(global_{spSeq.id}_num, numInBuffer, newNumInBuffer);')
        c.add(f'atomicCAS(global_{spSeq.id}_lock, warp_id+1, 0);')
        c.add('} else {')
        c.add(f'*global_{spSeq.id}_num = newNumInBuffer;')
        c.add('}')
        c.add('}')
        return c
    
    def genBlockShuffleCode(self, spSeq, attrsToShuffle):
        c = Code()
        c.add('if (thread_id == 0) shmWarpActive[threadIdx.x / 32] = warpActive;')
        c.add('__syncthreads();')
        c.add('if (active) {')
        c.add('int dst = 0;')
        c.add('int subwarp_id = threadIdx.x / 32;')
        c.add('for (int i = 0; i < subwarp_id; ++i) dst += __popc(shmWarpActive[i]);')
        c.add('dst += __popc(warpActive & prefixlanes);')
        
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst].start = {attr.id_name}.start;')
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst].end = {attr.id_name}.end;')
            else: # attr.dataType == Type.INT:
                c.add(f'shm{spSeq.id}_{attr.id_name}[dst] = {attr.id_name};')
        c.add('} // active')
        c.add('__syncthreads();')
        c.add('active = threadIdx.x < actThreadsB;')
        c.add('if (active) {')
        for attrId, attr in attrsToShuffle.items():
            if attr.dataType == Type.STRING:
                c.add(f'{attr.id_name}.start = shm{spSeq.id}_{attr.id_name}[threadIdx.x].start;')
                c.add(f'{attr.id_name}.end = shm{spSeq.id}_{attr.id_name}[threadIdx.x].end;')
            else: # attr.dataType == Type.INT:
                c.add(f'{attr.id_name} = shm{spSeq.id}_{attr.id_name}[threadIdx.x];')        
        c.add('} // active')
        return c
    
    def genShuffleCode(self, spSeq, attrsToShuffle):
        c = Code()
        if not self.doBlockShuffle and not self.doGridShuffle: return c

        c.add('{ // Shuffle')

        for attrId, attr in attrsToShuffle.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = 'str_t'
            c.add(f'__shared__ {dtype} shm{spSeq.id}_{attr.id_name}[{KernelCall.defaultBlockSize}];')

        c.add('if (threadIdx.x == 0) shmNumWarps = 0;')
        c.add('int actThreadsB = __syncthreads_count(active);')
        
        c.add(f'if (((isLast && blockIdx.x == {spSeq.id-1}) || (!isLast && actThreadsB > 0))) ' + '{ // Vote')

        c.add(f'unsigned warpActive = __ballot_sync(ALL_LANES, active);')
        c.add(f'int actThreadsW = __popc(warpActive);')
        c.add('int numWarps = 0;')

        if self.doGridShuffle:
            c.add(f'if ((!isLast && actThreadsB <= {self.gridShuffleThreshold}) || (isLast && blockIdx.x == {spSeq.id-1})) ' + '{ // GridShuffle')        
            c.add(self.genGridShuffleCode(spSeq, attrsToShuffle))
            c.add('} // GridShuffle')

        if self.doBlockShuffle:
            if self.doGridShuffle:
                c.add(f'else if (!isLast && actThreadsW <= {self.blockShuffleThreshold} && thread_id == 0)' + '{')
            else:
                c.add(f'if (!isLast && actThreadsW <= {self.blockShuffleThreshold} && thread_id == 0)' + '{')
            c.add('atomicAdd(&shmNumWarps, 1);')
            c.add('}')
            c.add('__syncthreads();')
            c.add('numWarps = shmNumWarps;')
            c.add(f'if (numWarps > {self.blockShuffleNumWarps}) ' + '{')
            c.add(self.genBlockShuffleCode(spSeq, attrsToShuffle))
            c.add('} // BlockShuffle')
        
        c.add('} // Vote')
        c.add('} // Shuffle')
        return c
    
    
    def genInputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        tid = spSeq.getTid()
        lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
        c = Code()
        
        c.add(f'while (__syncthreads_count(local{lastOp.opId}_range.start < local{lastOp.opId}_range.end || keepGoing))' + '{')
        c.add(f'active = (!keepGoing) && (local{lastOp.opId}_range.start < local{lastOp.opId}_range.end);')
        c.add(f'int loopvar{spSeq.id} = local{lastOp.opId}_range.start++;')
        
        # Declare in-boundary attrs
        c.add(f'//{list(attrsToDeclareAndMaterialize)}')
        c.add(f'int {tid.id_name};')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            c.add(f'{langType(attr.dataType)} {attr.id_name};')
        
        _, attrsToLoadBeforeExec, _, _ = attrsToDeclareAndMaterialize[0]
        c.add(self.genAttrDeclaration(spSeq.pipe, attrsToLoadBeforeExec))
        
        c.add('if (active) {')
        var = f'loopvar{spSeq.id}'
        c.add(f'{tid.id_name} = {spSeq.convertTid(var)};')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId in lastOp.generatingAttrs: continue
            c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
        c.add('}')
        
        attrsToShuffle = {}
        attrsToShuffle[tid.id] = tid        
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId in lastOp.generatingAttrs: continue
            attrsToShuffle[attrId] = attr
        c.add(self.genShuffleCode(spSeq, attrsToShuffle))
        
        c.add('if (active) {')
        # Load in-boundary attrs
        c.add(f'// last op generating Attrs: {lastOp.generatingAttrs}')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId not in lastOp.generatingAttrs: continue
            c.add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
            
        c.add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

        c.add('}')

        return c
    
    def genInputCodeForType2(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        attrsToShuffle = {}
        attrsToShuffle.update(spSeq.inBoundaryAttrs)
        c.add(self.genShuffleCode(spSeq, attrsToShuffle))
        c.add('{')
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
        for attrId, attr in spSeq.outBoundaryAttrs.items():
            if attrId in lastOp.generatingAttrs: continue
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING: dtype = 'str_t'
            c.add(f'{dtype} ts_{spSeq.id+1}_{attr.id_name};')
                
        c.add('if (active) {')
        
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
        if spSeq.inputType == 0 or spSeq.inputType == 1:
            c.add('keepGoing = false;')
        c.add('}')
            
        return c
    

    def genKernelCallCodeForPipe(self, pipe_name, pipe):
        c = Code()
        c.add(pipe.genCodeBeforeExecution())
        c.add('cudaDeviceSynchronize();')
        c.add('bool isLast = false;')
        c.add(f'krnl_{pipe_name}<<<{KernelCall.defaultGridSize},{KernelCall.defaultBlockSize}>>>(')
        c.add(self.genArgsForKernelCallForPipe(pipe_name, pipe))
        c.add('cnts')
        c.add(');')
        c.add('cudaDeviceSynchronize();')
        c.add('isLast = true;')
        c.add(f'krnl_{pipe_name}<<<{len(pipe.subpipeSeqs)},{KernelCall.defaultBlockSize}>>>(')
        c.add(self.genArgsForKernelCallForPipe(pipe_name, pipe))
        c.add('cnts')
        c.add(');')
        c.add('cudaDeviceSynchronize();')
        c.add(pipe.genCodeAfterExecution())
        return c


def compileCode( filename, code, compileOption, arch="sm_75", debug=False):
    codeGen = codeGenerator(None, None)
    codeGen.compileCode( filename, code, compileOption, arch, debug)

def genCode(dss, pipes):
    codeGen = codeGenerator(dss, pipes)
    return codeGen.genCode()