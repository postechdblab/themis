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
        self.doCnt = False
        self.tsWidth = 32
        self.doInterWarpLB = KernelCall.args.inter_warp_lb
        self.maxInterval = KernelCall.args.inter_warp_lb_interval
        self.interWarpLbMethod = KernelCall.args.inter_warp_lb_method # 'aws'
        self.idleWarpDetectionType = KernelCall.args.inter_warp_lb_detection_method # 'twolvlbitmaps' # 'twolvlbitmaps'
        self.worksharingThreshold = KernelCall.args.inter_warp_lb_ws_threshold # 512 #16384 #1024
        self.nonEmptyBufferDetectionType = KernelCall.args.inter_warp_lb_detection_method # 'twolvlbitmaps' #'idqueue' #'twolvlbitmaps'
        self.doWorkoadSizeTracking = True
    
    def genArgsForKernelDeclarationForPipe(self, pipe_name, pipe):
        c = super().genArgsForKernelDeclarationForPipe(pipe_name, pipe)
        if not self.doInterWarpLB: return c
        c.add('unsigned int* global_num_idle_warps, int* global_scan_offset,')
        if self.interWarpLbMethod == 'aws':
            c.add('Themis::PushedParts::PushedPartsStack* gts, size_t size_of_stack_per_warp,')
            c.add('Themis::StatisticsPerLvl* global_stats_per_lvl,')
            if self.idleWarpDetectionType == 'twolvlbitmaps':
                c.add('unsigned long long* global_bit1, unsigned long long* global_bit2,')
            elif self.idleWarpDetectionType == 'idqueue':
                c.add('Themis::Detection::Stack::IdStack* global_id_stack,')
        elif self.interWarpLbMethod == 'ws':
            c.add('WorkSharing::TaskBook* taskbook, WorkSharing::TaskStack* taskstack,')
            if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
                c.add('unsigned long long* num_buffered_inodes, unsigned long long* global_bit1, unsigned long long* global_bit2,')
        return c
    
    def genArgsForKernelCallForPipe(self, pipe_name, pipe):
        c = super().genArgsForKernelCallForPipe(pipe_name, pipe)
        if not self.doInterWarpLB: return c
        c.add('global_num_idle_warps, global_scan_offset,')
        if self.interWarpLbMethod == 'aws':
            c.add('gts, size_of_stack_per_warp,')
            c.add('global_stats_per_lvl,')
            if self.idleWarpDetectionType == 'twolvlbitmaps':
                c.add('global_bit1, global_bit2,')
            elif self.idleWarpDetectionType == 'idqueue':
                c.add('global_id_stack,')
        elif self.interWarpLbMethod == 'ws':
            c.add('taskbook, taskstack,')
            if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
                c.add('num_buffered_inodes, global_bit1, global_bit2,')
        return c
    
    def genDeclarationInMain(self):
        c = super().genDeclarationInMain()
        if not self.doInterWarpLB: return c

        num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
        min_num_warps = min(num_warps, KernelCall.args.min_num_warps)

        # Declare the basic variables for inter warp load balancing
        c.add('unsigned int* global_info;')
        c.add('cudaMalloc((void**)&global_info, sizeof(unsigned int) * 2 * 32);')
        c.add('unsigned int* global_num_idle_warps = global_info;')
        c.add('int* global_scan_offset = (int*)(global_info + 32);')
        
        if self.interWarpLbMethod == 'ws':
            c.add('WorkSharing::TaskBook* taskbook;')
            c.add('WorkSharing::TaskStack* taskstack;')
            c.add('cudaMalloc((void**) &taskbook, 1 * 1024 * 1024 * 1024);')
            c.add('cudaMalloc((void**) &taskstack, 256 * 1024 * 1024);')
            if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
                c.add('unsigned long long* num_buffered_inodes = NULL;')
                c.add('cudaMalloc((void**)&num_buffered_inodes, sizeof(unsigned long long));')
                bitmapsize = (int((num_warps-1) / 64) + 1)
                c.add(f'//bitmapsize: {bitmapsize} for {num_warps}')
                c.add('unsigned long long* global_bit1;')
                c.add(f'cudaMalloc((void**)&global_bit1, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
                c.add('unsigned long long* global_bit2 = global_bit1 + 16;')
                c.add(f'cudaMemset(global_bit1, 0, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
        elif self.interWarpLbMethod == 'aws':
            c.add('Themis::StatisticsPerLvl* global_stats_per_lvl = NULL;')
            c.add(f'Themis::InitStatisticsPerLvl(global_stats_per_lvl, {num_warps});')
            c.add(f'Themis::PushedParts::PushedPartsStack* gts;')
            c.add(f'size_t size_of_stack_per_warp;')
            c.add(f'Themis::PushedParts::InitPushedPartsStack(gts, size_of_stack_per_warp, 1 << 31, {num_warps});')
            if self.idleWarpDetectionType == 'twolvlbitmaps':
                bitmapsize = (int((num_warps-1) / 64) + 1)
                c.add(f'//bitmapsize: {bitmapsize} for {num_warps}')
                c.add('unsigned long long* global_bit1;')
                c.add(f'cudaMalloc((void**)&global_bit1, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
                c.add('unsigned long long* global_bit2 = global_bit1 + 16;')
                c.add('cudaMemset(global_bit1, 0, sizeof(unsigned long long) * 16);')
                c.add(f'cudaMemset(global_bit2, 0, sizeof(unsigned long long) * {bitmapsize * 16});')
            elif self.idleWarpDetectionType == 'idqueue':
                c.add('int* global_id_stack_buf = NULL;')
                c.add(f'cudaMalloc((void**)&global_id_stack_buf, sizeof(int) * {num_warps+4});')
                c.add(f'cudaMemset(global_id_stack_buf, 0, sizeof(int) * {num_warps+4});')
                c.add('Themis::Detection::Stack::IdStack* global_id_stack = (Themis::Detection::Stack::IdStack*) global_id_stack_buf;')
        return c
    
    def genKernelCallCodeForPipe(self, pipe_name, pipe):
        c = Code()
        c.add(pipe.genCodeBeforeExecution())
        if self.doInterWarpLB:
            num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
            c.add('cudaMemset(global_info, 0, 64 * sizeof(unsigned int));')
            if self.interWarpLbMethod == 'ws':
                c.add('cudaMemset(taskbook, 0, 128);')
                c.add('cudaMemset(taskstack, 0, 128);')
                if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
                    c.add(f'cudaMemset(num_buffered_inodes, 0, sizeof(unsigned long long));')
                    bitmapsize = (int((num_warps-1) / 64) + 1)
                    c.add(f'cudaMemset(global_bit1, 0, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
            elif self.interWarpLbMethod == 'aws':
                c.add(f'\tcudaMemset(global_stats_per_lvl, 0, sizeof(Themis::StatisticsPerLvl) * {len(pipe.subpipeSeqs)});')
                tableSize = pipe.subpipes[0].operators[0].genTableSize()
                if tableSize[0] == '*': 
                    tableSize = tableSize[1:]                
                    c.add(f'Themis::InitStatisticsPerLvlPtr(global_stats_per_lvl, {num_warps}, {tableSize}, {len(pipe.subpipeSeqs)});')
                else:
                    c.add(f'Themis::InitStatisticsPerLvl(global_stats_per_lvl, {num_warps}, {tableSize}, {len(pipe.subpipeSeqs)});')

                if self.idleWarpDetectionType == 'twolvlbitmaps':
                    bitmapsize = (int((num_warps-1) / 64) + 1)
                    c.add(f'cudaMemset(global_bit1, 0, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
                elif self.idleWarpDetectionType == 'idqueue':
                    c.add(f'cudaMemset(global_id_stack, 0, sizeof(Themis::Detection::Stack::IdStack));')
            c.add('cudaDeviceSynchronize();')

        c.add(f'krnl_{pipe_name}<<<{KernelCall.defaultGridSize},{KernelCall.defaultBlockSize}>>>(')
        c.add(self.genArgsForKernelCallForPipe(pipe_name, pipe))
        c.add('cnts')
        c.add(');')
        c.add('cudaDeviceSynchronize();')

        c.add(pipe.genCodeAfterExecution())
        return c
            
    def genPipeIntialization(self, pipe):
        c = Code()
        c.add(f'__shared__ int active_thread_ids[{KernelCall.defaultBlockSize}];')
        
        if self.doInterWarpLB:
            c.add(f'if (blockIdx.x > {int(1024 / KernelCall.defaultBlockSize) * 82}) return; // Maximum number of warps a GPU can execute concurrently')
            c.add('int gpart_id = -1;')
            c.add('Themis::WarpsStatus* warp_status = (Themis::WarpsStatus*) global_num_idle_warps;')
            c.add('if (threadIdx.x == 0) {')
            c.add('if (warp_status->isTerminated()) active_thread_ids[0] = -1;')
            c.add(f'else active_thread_ids[0] = warp_status->addTotalWarpNum({int(KernelCall.defaultBlockSize/32)});')
            c.add('}')
            c.add('__syncthreads();')
            c.add('gpart_id = active_thread_ids[0];')
            c.add('__syncthreads();')
            c.add('if (gpart_id == -1) return;')
            c.add('gpart_id = gpart_id + threadIdx.x / 32;')
        
        
        c.add('int thread_id = threadIdx.x % 32;')
        c.add('int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;')
        c.add('unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));')
        c.add('int active = 0;')
        
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            if spSeq.inputType == 1:
                c.add(f'Range ts_{spSeqId}_range;')
                c.add(f'Range ts_{spSeqId}_range_cached;')
                tid = spSeq.getTid()
                lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    if attrId == tid.id: continue
                    if attrId in lastOp.generatingAttrs: continue
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name};')
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name}_cached;') 
            elif spSeq.inputType == 2:
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name};')
                    c.add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name}_flushed;')                 
        return c
    
    def genInitialDistribution(self, pipe):
        c = Code()
        scan = pipe.subpipes[0].operators[0]
        
        c.add(f'int inodes_cnts = 0; // the number of nodes per level')
        c.add(f'Range ts_0_range_cached;')
        c.add(f'ts_0_range_cached.end = {scan.genTableSize()};')
        
        if self.doInterWarpLB:
            num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
            c.add('int local_scan_offset = 0;')
            c.add('int global_scan_end = ts_0_range_cached.end;')
            c.add(f'Themis::PullINodesAtZeroLvlDynamically<{num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
            if self.interWarpLbMethod == 'aws':            
                c.add('Themis::LocalLevelAndOrderInfo local_info;')
                if self.doWorkoadSizeTracking:
                    c.add('Themis::WorkloadTracking::InitLocalWorkloadSizeAtZeroLvl(inodes_cnts, local_info, global_stats_per_lvl);')
                c.add('unsigned interval = 1;')
                c.add(f'unsigned loop = {self.maxInterval} - 1;')
        else:
            c.add(f'Themis::PullINodesAtZeroLvlStatically(thread_id, ts_0_range_cached, inodes_cnts);')

        c.add(f'unsigned mask_32 = 0; // a bit mask to indicate levels where more than 32 INodes exist')
        c.add(f'unsigned mask_1 = 0; // a bit mask to indicate levels where INodes exist')
        c.add(f'int lvl = -1;')
        c.add(f'Themis::UpdateMaskAtZeroLvl(0, thread_id, ts_0_range_cached, mask_32, mask_1);')
        c.add(f'Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        return c
    
    def genInputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        stepSize = KernelCall.defaultBlockSize * KernelCall.defaultGridSize
        tid = spSeq.getTid()
        c = Code()
        if self.doInterWarpLB and self.interWarpLbMethod == 'aws':
            c.add(f'while (lvl >= 0 && loop < {self.maxInterval}) ' + '{')
        else:
            c.add('while (lvl >= 0) {')
        c.add('__syncwarp();')
        c.add('if (lvl == 0) {')
        c.add(f'int {tid.id_name};')
        c.add(f'Themis::FillIPartAtZeroLvl(lvl, thread_id, active, {tid.id_name}, ts_0_range_cached, mask_32, mask_1, {stepSize});')
        
        if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
            c.add('Themis::WorkloadTracking::UpdateWorkloadSizeAtZeroLvl(thread_id, ++loop, local_info, global_stats_per_lvl);')
        return c
    
    def genInputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        tid = spSeq.getTid()
        lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
        c = Code()
        if self.doInterWarpLB and self.interWarpLbMethod == 'aws':
            c.add(f'while (lvl >= {spSeq.id} && loop < {self.maxInterval}) ' + '{')
        else:
            c.add(f'while (lvl >= {spSeq.id}) ' + '{')
        c.add('__syncwarp();')
        c.add(f'if (lvl == {spSeq.id}) ' + '{')
        c.add(f'int loopvar{spSeq.id};')
        c.add(f'Themis::FillIPartAtLoopLvl({spSeq.id}, thread_id, active, loopvar{spSeq.id}, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
        
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
        c.add('int ts_src = 32;')
        c.add(f'bool is_updated = Themis::DistributeFromPartToDPart(thread_id, {spSeq.id}, ts_src, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
        if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
            c.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtLoopLvl(thread_id, {spSeq.id}, ++loop, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_1, local_info, global_stats_per_lvl);')
        c.add('if (is_updated) {')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            if attrId == tid.id: continue
            if attrId in lastOp.generatingAttrs: continue
            c.add('{')
            name = f'ts_{spSeq.id}_{attr.id_name}'
            if attr.dataType == Type.STRING:
                c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
                c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
                c.add('if (ts_src < 32) {')
                c.add(f'{name}_cached.start = start;')
                c.add(f'{name}_cached.end = end;')
                c.add('}')
            else:
                c.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
                c.add(f'if (ts_src < 32) {name}_cached = cache;')
            c.add('}')
        c.add('}')
        return c
    
    def genInputCodeForType2(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        c.add(f'if (lvl == {spSeq.id}) ' + '{')
        if len(spSeq.inBoundaryAttrs) > 0:
            c.add(f'if (!(mask_32 & (0x1u << {spSeq.id}))) ' + '{')
            for attrId, attr in spSeq.inBoundaryAttrs.items():
                c.add(f'ts_{spSeq.id}_{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_flushed;')    
            c.add('}')
        c.add(f'Themis::FillIPartAtIfLvl({spSeq.id}, thread_id, inodes_cnts, active, mask_32, mask_1);')
        if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
            c.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtIfLvl(thread_id, {spSeq.id}, loop, inodes_cnts, mask_1, local_info, global_stats_per_lvl);')
        
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            c.add(f'{langType(attr.dataType)} {attr.id_name};')
        c.add('if (active) {')
        for attrId, attr in spSeq.inBoundaryAttrs.items():
            c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
        c.add('}')
        return c


    def genWorkSharingPushCode(self, spSeq, attrs):
        c = Code()
        if not self.doInterWarpLB or not self.interWarpLbMethod == 'ws':
            return code
        
        speculated_size = 0
        for atttId, attr in attrs.items():
            speculated_size += CType.size[langType(attr.dataType)]
        
        lastOp = spSeq.subpipes[-1].operators[-1]
        c.add('if (push_active_mask) {')
        if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
            c.add(f'if ((local{lastOp.opId}_range.end - local{lastOp.opId}_range.start) > {self.worksharingThreshold})' + '{')
            c.add(f'atomicAdd(num_buffered_inodes, local{lastOp.opId}_range.end - local{lastOp.opId}_range.start);')
            c.add('}')
        c.add('WorkSharing::Task* task = NULL;')
        c.add(f'if (active) task = taskbook->AllocTask({spSeq.id+1},local{lastOp.opId}_range.start,local{lastOp.opId}_range.end,{speculated_size},{self.worksharingThreshold});')                
        c.add('if (task != NULL) {')
        if len(attrs) > 0:
            c.add('char* attr = task->GetAttrPtr();')
            for attrId, attr in attrs.items():
                c.add('{')
                if attr.dataType == Type.STRING:
                    c.add(f'*((str_t*) attr) = {attr.id_name};')
                    c.add(f'attr += sizeof(str_t);')
                else:
                    c.add(f'*(({langType(attr.dataType)}*)attr) = {attr.id_name};')
                    c.add(f'attr += {CType.size[langType(attr.dataType)]};')
                c.add('}')
        
        c.add('} // task != NULL')
        if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
            c.add('WorkSharing::TaskStack* my_taskstack = (WorkSharing::TaskStack*) (((char*) taskstack) + gpart_id * 1024 * 64);;')
            c.add('if (my_taskstack->Push(task) && thread_id == 0) Themis::Detection::TwoLvlBitmaps::Set(gpart_id, global_bit1, global_bit2);')
        else:
            c.add('taskstack->Push(task);')
        c.add(f'push_active_mask = __ballot_sync(ALL_LANES, active && (local{lastOp.opId}_range.start < local{lastOp.opId}_range.end));')
        c.add('} // push_active_mask')
        
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
        
        c.add('unsigned push_active_mask = __ballot_sync(ALL_LANES, active);')
        
        if self.doInterWarpLB and self.interWarpLbMethod == 'ws':
            # Find attributes to push
            attrs = {}
            for attrId, attr in spSeq.outBoundaryAttrs.items():
                if attrId == tid.id: continue
                if attrId in lastOp.generatingAttrs: continue
                attrs[attrId] = attr
            c.add(self.genWorkSharingPushCode(spSeq, attrs))
        
        c.add('if (push_active_mask) {')
        # generated by the last Op?
        
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
        c.add('int ts_src = 32;')
        c.add(f'Themis::DistributeFromPartToDPart(thread_id, {spSeq.id+1}, ts_src, ts_{spSeq.id+1}_range, ts_{spSeq.id+1}_range_cached);')
        c.add(f'Themis::UpdateMaskAtLoopLvl({spSeq.id+1}, ts_{spSeq.id+1}_range_cached, mask_32, mask_1);')
        for attrId, attr in spSeq.outBoundaryAttrs.items():
            if attrId in lastOp.generatingAttrs: continue
            c.add('{')
            name = f'ts_{spSeq.id+1}_{attr.id_name}'
            if attr.dataType == Type.STRING:
                c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
                c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
                c.add('if (ts_src < 32) {')
                c.add(f'{name}_cached.start = start;')
                c.add(f'{name}_cached.end = end;')
                c.add('}')
            else:
                c.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
                c.add(f'if (ts_src < 32) {name}_cached = cache;')
            c.add('}')
        c.add('}')
        
        # Find the outermost loop level
        loop_lvl = self.findLowerLoopLvl(spSeq)
        c.add(f'if (!(mask_32 & (0x1 << {spSeq.id+1})))' + '{')
        c.add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
        c.add('else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        c.add('continue;')
        c.add('}')
        c.add(f'lvl = {spSeq.id+1};')
        
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

        c.add('unsigned push_active_mask = __ballot_sync(ALL_LANES, active);')
        c.add('if (push_active_mask) {')
        c.add(f'int old_ts_cnt = __shfl_sync(ALL_LANES, inodes_cnts, {spSeq.id+1});')
        c.add('int ts_cnt = old_ts_cnt + __popc(push_active_mask);')
        c.add(f'if (thread_id == {spSeq.id+1}) inodes_cnts = ts_cnt;')
        c.add(f'Themis::UpdateMaskAtIfLvlAfterPush({spSeq.id+1}, ts_cnt, mask_32, mask_1);')
        if len(spSeq.outBoundaryAttrs) > 0:
            c.add('if (ts_cnt >=32) {')
            for attrId, attr in spSeq.outBoundaryAttrs.items():
                c.add(f'ts_{spSeq.id+1}_{attr.id_name} = {attr.id_name};')
            c.add('if (ts_cnt - old_ts_cnt < 32) {')
            c.add('unsigned ts_src = 32;')
            c.add('if (!active) ts_src = old_ts_cnt - __popc((~push_active_mask) & prefixlanes) - 1;')
            for attrId, attr in spSeq.outBoundaryAttrs.items():
                name = f'ts_{spSeq.id+1}_{attr.id_name}'
                c.add('{')
                if attr.dataType == Type.STRING:
                    c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}_flushed.start, ts_src);')
                    c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}_flushed.end, ts_src);')
                    c.add('if (ts_src < 32) {')
                    c.add(f'{name}.start = start;')
                    c.add(f'{name}.end = end;')
                    c.add('}')
                else:
                    c.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}_flushed, ts_src);')
                    c.add(f'if (ts_src < 32) {name} = cache;')
                c.add('}')
            c.add('}')            
            c.add('} else {')
            c.add('active_thread_ids[threadIdx.x] = 32;')
            c.add('int* src_thread_ids = active_thread_ids + ((threadIdx.x >> 5) << 5);')
            c.add('if (active) src_thread_ids[__popc(push_active_mask & prefixlanes)] = thread_id;')
            c.add('unsigned ts_src = thread_id >= old_ts_cnt && thread_id < ts_cnt ? src_thread_ids[thread_id - old_ts_cnt] : 32;')        
            for attrId, attr in spSeq.outBoundaryAttrs.items():
                name = f'ts_{spSeq.id+1}_{attr.id_name}'
                c.add('{')
                if attr.dataType == Type.STRING:
                    c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.start, ts_src);')
                    c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.end, ts_src);')
                    c.add('if (ts_src < 32) {')
                    c.add(f'{name}_flushed.start = start;')
                    c.add(f'{name}_flushed.end = end;')
                    c.add('}')
                else:
                    c.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {attr.id_name}, ts_src);')
                    c.add(f'if (ts_src < 32) {name}_flushed = cache;')
                c.add('}')
            c.add('}')
        c.add('} // push active mask')
        # Find the outermost loop level
        loop_lvl = self.findLowerLoopLvl(spSeq)
        c.add(f'if (!(mask_32 & (0x1 << {spSeq.id+1})))' + '{')
        c.add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
        c.add('else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        c.add('continue;')
        c.add('}')
        c.add(f'lvl = {spSeq.id+1};')
        
        c.add('}')
        return c

    def genOutputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        # Find the outermost loop level
        loop_lvl = self.findLowerLoopLvl(spSeq)
        c.add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
        c.add('else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
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
            c.add('} //' +  f'{spSeq.id}')
        return c


    def genCopyCodeForAdaptiveWorkSharingPull(self, pipe):
        c = Code()
        tsWidth = self.tsWidth
        c.add('switch (lowest_lvl) {')
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            c.add(f'case {spSeqId}: ' + '{')
            if spSeq.inputType == 0:
                c.add('Themis::PushedParts::PushedPartsAtZeroLvl* src_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
                c.add('Themis::PullINodesFromPPartAtZeroLvl(thread_id, src_pparts, ts_0_range_cached, inodes_cnts);')
                c.add('if (thread_id == 0) stack->PopPartsAtZeroLvl();')
            elif spSeq.inputType == 1:
                c.add('Themis::PushedParts::PushedPartsAtLoopLvl* src_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
                c.add(f'Themis::PullINodesFromPPartAtLoopLvl(thread_id, {spSeqId}, src_pparts, ts_{spSeqId}_range_cached, ts_{spSeqId}_range, inodes_cnts);')
                
                attrs = {}
                tid = spSeq.getTid()
                lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    if attrId == tid.id: continue
                    if attrId in lastOp.generatingAttrs: continue
                    attrs[attrId] = attr
                
                speculated_size = 0
                if len(attrs) > 0:
                    c.add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')
                    
                    for attrId, attr in attrs.items():
                        if attr.dataType == Type.STRING:
                            c.add(f'Themis::PullStrAttributesAtLoopLvl(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
                        else:
                            c.add(f'Themis::PullAttributesAtLoopLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));')
                        speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                c.add(f'if (thread_id == 0) stack->PopPartsAtLoopLvl({speculated_size});')
            elif spSeq.inputType == 2:
                c.add('Themis::PushedParts::PushedPartsAtIfLvl* src_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
                c.add(f'Themis::PullINodesFromPPartAtIfLvl(thread_id, {spSeqId}, src_pparts, inodes_cnts);')
                if len(spSeq.inBoundaryAttrs) > 0:
                    c.add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')
                    speculated_size = 0
                    for attrId, attr in spSeq.inBoundaryAttrs.items():
                        if attr.dataType == Type.STRING:
                            c.add(f'Themis::PullStrAttributesAtIfLvl(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
                        else:
                            c.add(f'Themis::PullAttributesAtIfLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));')
                        speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                    c.add(f'if (thread_id == 0) stack->PopPartsAtIfLvl({speculated_size});')
            c.add('} break;')
        c.add('} // switch')
        return c


    def genCodeForAdaptiveWorkSharingPull(self, pipe):
        num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
        min_num_warps = min(num_warps, KernelCall.args.min_num_warps)
        c = Code()

        if KernelCall.args.mode == 'stats':
            c.add('unsigned long long current_tp = clock64();')
            c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
            c.add('tp = current_tp;')
            c.add('current_status = TYPE_STATS_WAITING;')
            c.add('stat_counters[TYPE_STATS_NUM_IDLE] += 1;')
            
        c.add('if (thread_id == 0 && local_info.locally_lowest_lvl != -1) atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_warps, 1);')
        c.add('inodes_cnts = 0;')
        c.add('mask_32 = mask_1 = 0;')
        c.add('unsigned int num_idle_warps = 0;')
        c.add('int src_warp_id = -1;')
        c.add('int lowest_lvl = -1;')
        c.add(f'bool is_successful = Themis::PullINodesAtZeroLvlDynamically<{num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
        c.add('if (is_successful) {')
        c.add('loop = 0;')
        c.add('lowest_lvl = 0;')
        c.add('} else { // adaptive work-sharing for all levels')    
        # Adaptive work-sharing for all levels
        c.add(f'Themis::Wait<{num_warps}, {min_num_warps}>(')
        c.add('gpart_id, src_warp_id, warp_id, thread_id, lowest_lvl, warp_status, num_idle_warps, global_stats_per_lvl, gts, size_of_stack_per_warp')
        if self.idleWarpDetectionType == 'twolvlbitmaps':
            c.add(',global_bit1, global_bit2')
        elif self.idleWarpDetectionType == 'idqueue':
            c.add(',global_id_stack')
        c.add(');')

        # Terminiation code    
        c.add('if (src_warp_id == -2) {')
        if KernelCall.args.mode == 'stats':
            c.add(f"stat_counters[TYPE_STATS_WAITING] += (clock64() - tp);")
        c.add('if (blockIdx.x == 0 && threadIdx.x == 0) warp_status->terminate();')
        c.add('break;')
        c.add('}')
        # ~Termination code
        
        c.add('Themis::PushedParts::PushedPartsStack* stack = Themis::PushedParts::GetIthStack(gts, size_of_stack_per_warp, (size_t) src_warp_id);')
        
        c.add(self.genCopyCodeForAdaptiveWorkSharingPull(pipe))
        c.add('if (thread_id == 0) {')
        c.add('__threadfence();')
        c.add('stack->FreeLock();')
        c.add('}')
        c.add(f'loop = {self.maxInterval} - 1;')
        # ~ Adaptive work-sharing for all levels
        c.add('} // ~ adaptive work-sharing for all levels')
        
        if self.doWorkoadSizeTracking:
            c.add('Themis::WorkloadTracking::InitLocalWorkloadSize(lowest_lvl, inodes_cnts, local_info, global_stats_per_lvl);')
        c.add('lvl = lowest_lvl;')
        c.add('if (thread_id == lvl) mask_32 = inodes_cnts >= 32 ? 0x1 << lvl : 0;')
        c.add('mask_1 = 0x1 << lvl;')
        c.add('mask_32 = __shfl_sync(ALL_LANES, mask_32, lvl);')
        
        return c

    def genCopyCodeForAdaptiveWorkSharingPush(self, pipe):
        tsWidth = self.tsWidth
        stepSize = KernelCall.defaultBlockSize * KernelCall.defaultGridSize
        
        c = Code()
        c.add('switch(lvl_to_push) {')
        
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            c.add(f'case {spSeqId}: ' + '{')
            if spSeq.inputType == 0:
                c.add(f'if (thread_id == 0) stack->PushPartsAtZeroLvl();')
                c.add('Themis::PushedParts::PushedPartsAtZeroLvl* target_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
                c.add(f'num_to_push = Themis::PushINodesToPPartAtZeroLvl(thread_id, target_pparts, ts_0_range_cached, {stepSize});')
                c.add(f'num_remaining = num_nodes - num_to_push;')
                c.add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
                c.add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
                
            elif spSeq.inputType == 1:

                attrs = {}
                tid = spSeq.getTid()
                lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    if attrId == tid.id: continue
                    if attrId in lastOp.generatingAttrs: continue
                    attrs[attrId] = attr
                
                speculated_size = 0
                for attrId, attr in attrs.items():
                    speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                c.add(f'if (thread_id == 0) stack->PushPartsAtLoopLvl({spSeqId}, {speculated_size});')
                c.add('Themis::PushedParts::PushedPartsAtLoopLvl* target_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
                c.add(f'num_to_push = Themis::PushINodesToPPartAtLoopLvl(thread_id, {spSeqId}, target_pparts, ts_{spSeqId}_range_cached, ts_{spSeqId}_range);')
                c.add(f'num_remaining = num_nodes - num_to_push;')
                c.add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
                c.add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
                c.add(f'int ts_src = 32;')
                c.add(f'Themis::DistributeFromPartToDPart(thread_id, {spSeqId}, ts_src, ts_{spSeqId}_range, ts_{spSeqId}_range_cached, mask_32, mask_1);')
                if len(attrs) > 0:
                    speculated_size = 0
                    c.add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
                    for attrId, attr in attrs.items():
                        name = f'ts_{spSeqId}_{attr.id_name}'
                        c.add('{')
                        if attr.dataType == Type.STRING:
                            c.add(f'Themis::PushStrAttributesAtLoopLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
                            c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
                            c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
                            c.add(f'if (ts_src < 32) {name}_cached.start = start;')
                            c.add(f'if (ts_src < 32) {name}_cached.end = end;')
                        else:
                            dtype = langType(attr.dataType)
                            c.add(f'Themis::PushAttributesAtLoopLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
                            c.add(f'{dtype} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
                            c.add(f'if (ts_src < 32) {name}_cached = cache;')
                        c.add('}')
                        speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
            else:
                speculated_size = 0
                for attrId, attr in spSeq.inBoundaryAttrs.items():
                    speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                c.add(f'if (thread_id == 0) stack->PushPartsAtIfLvl({spSeqId}, {speculated_size});')
                c.add('Themis::PushedParts::PushedPartsAtIfLvl* target_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
                c.add(f'num_to_push = Themis::PushINodesToPPartAtIfLvl(thread_id, {spSeqId}, target_pparts, inodes_cnts);')
                c.add(f'mask_32 = ((~m) & mask_32);')
                c.add(f'mask_1 = ((~m) & mask_1);')
                if len(spSeq.inBoundaryAttrs) > 0:
                    c.add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
                    speculated_size = 0
                    for attrId, attr in spSeq.inBoundaryAttrs.items():
                        name = f'ts_{spSeqId}_{attr.id_name}'
                        c.add('{')
                        if attr.dataType == Type.STRING:
                            c.add(f'Themis::PushStrAttributesAtIfLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
                        else:
                            dtype = langType(attr.dataType)
                            c.add(f'Themis::PushAttributesAtIfLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
                        c.add('}')
                        speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
            c.add('} break;')
        c.add('}')
        num_warps_per_block = int(KernelCall.defaultBlockSize / 32)
        c.add(f'if ((target_warp_id / {num_warps_per_block}) == (gpart_id / {num_warps_per_block})) __threadfence_block();')
        c.add('else __threadfence();')
        return c

    def genCodeForAdaptiveWorkSharingPush(self, pipe):
        
        num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
        min_num_warps = min(num_warps, KernelCall.args.min_num_warps)

        c = Code()
        
        if KernelCall.args.mode == 'stats':
            c.add('unsigned long long current_tp = clock64();')
            c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
            c.add('tp = current_tp;')
            c.add('current_status = TYPE_STATS_PUSHING;')
            
        c.add('int target_warp_id = -1;')
        c.add('unsigned int num_idle_warps = 0;')
        c.add('unsigned int num_warps = 0;')
        
        if self.doWorkoadSizeTracking:
            c.add('bool is_allowed = Themis::isPushingAllowed(thread_id, warp_status, num_idle_warps, num_warps, local_info, global_stats_per_lvl);')
        else:
            c.add('bool is_allowed = false;')
        c.add('if (is_allowed) {')
        
        c.add(f'Themis::FindIdleWarp<{len(pipe.subpipeSeqs)},{num_warps},{min_num_warps}>(')
        c.add('target_warp_id, warp_id, thread_id, warp_status, num_idle_warps, num_warps,gts, size_of_stack_per_warp')
        if self.idleWarpDetectionType == 'twolvlbitmaps':
            c.add(', global_bit1, global_bit2')
        elif self.idleWarpDetectionType == 'idqueue':
            c.add(', global_id_stack')        
        c.add(');')
        c.add('}')
        
        if KernelCall.args.mode == 'sample':
            c.add('if (tried) sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_TRY_PUSHING);')
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_DETECTING);')

        c.add('if (target_warp_id >= 0) {')
        if KernelCall.args.mode == 'sample':
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_PUSHING);')
            
        c.add('Themis::PushedParts::PushedPartsStack* stack = Themis::PushedParts::GetIthStack(gts, size_of_stack_per_warp, target_warp_id);')      
        
        c.add('int lvl_to_push = local_info.locally_lowest_lvl;')
        
        c.add('int num_to_push = 0;')
        c.add('int num_remaining = 0;')
        c.add('int num_nodes = local_info.num_nodes_at_locally_lowest_lvl > 0 ? local_info.num_nodes_at_locally_lowest_lvl : 0;')
        
        c.add(f'unsigned m = 0x1u << lvl_to_push;')
        c.add(self.genCopyCodeForAdaptiveWorkSharingPush(pipe))
        c.add("Themis::WorkloadTracking::UpdateWorkloadSizeOfIdleWarpAfterPush(thread_id, lvl_to_push, num_to_push, global_stats_per_lvl);")
        c.add('if (thread_id == 0) stack->FreeLock();')
        
        # Recalculate current workload size of this busy warp
        c.add('// Calculate the workload size of this busy warp')
        c.add('int new_num_nodes_at_locally_lowest_lvl = num_remaining;')
        c.add('int8_t new_local_max_order = Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);')
        
        c.add('int new_local_lowest_lvl = new_num_nodes_at_locally_lowest_lvl > 0 ? lvl_to_push : -1;')
        # Find the new lowet level
        c.add('if (new_num_nodes_at_locally_lowest_lvl == 0 && mask_1 != 0) {')
        c.add('new_local_lowest_lvl = __ffs(mask_1) - 1;')
        if len(pipe.subpipeSeqs) > 1:
            c.add('switch (new_local_lowest_lvl) {')
            for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
                if spSeqId == 0: continue
                c.add(f'case {spSeqId}:' + '{')
                if spSeq.inputType == 1:
                    c.add(f'Themis::CountINodesAtLoopLvl(thread_id, {spSeqId}, ts_{spSeqId}_range_cached, ts_{spSeqId}_range, inodes_cnts);')
                    c.add(f'new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {spSeqId});')
                    c.add(f'new_local_max_order = Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);')
                else:
                    c.add(f'new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {spSeqId});')
                    c.add(f'new_local_max_order = 0;')
                c.add('} break;')
            c.add('}')      
        c.add('}')
        c.add('Themis::WorkloadTracking::UpdateWorkloadSizeOfBusyWarpAfterPush(thread_id, mask_1, new_num_nodes_at_locally_lowest_lvl, new_local_lowest_lvl, new_local_max_order, local_info, global_stats_per_lvl);')
        c.add('interval = 0;')
        c.add('Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        c.add('} else {')
        c.add(f'Themis::chooseNextIntervalAfterPush(interval, local_info, num_warps, num_idle_warps, is_allowed, {self.maxInterval});')
        c.add('} // else')
        c.add(f'loop = {self.maxInterval} - interval;')
        return c

    def genCodeForAdaptiveWorkSharing(self, pipe):
        c = Code()
        c.add('loop = 0;')
        c.add('if (lvl == -1) {')
        c.add(self.genCodeForAdaptiveWorkSharingPull(pipe))
        c.add('} else { // if (lvl == -1)')
        c.add(self.genCodeForAdaptiveWorkSharingPush(pipe))
        c.add('} // end of adaptive work sharing ')
        return c
        
    def genCodeForWorkSharingPull(self, pipe):
        num_warps = int(KernelCall.defaultBlockSize / 32) * KernelCall.defaultGridSize
        min_num_warps = min(num_warps, KernelCall.args.min_num_warps)
        
        c = Code()

        if KernelCall.args.mode == 'stats':
            c.add('unsigned long long current_tp = clock64();')
            c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
            c.add('tp = current_tp;')
            c.add('current_status = TYPE_STATS_WAITING;')
            c.add('stat_counters[TYPE_STATS_NUM_IDLE] += 1;')
            
        c.add('inodes_cnts = 0;')
        c.add('mask_32 = mask_1 = 0;')
        c.add(f'bool is_successful = Themis::PullINodesAtZeroLvlDynamically<{num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
        c.add('if (is_successful) {')
        c.add('lvl = 0;')
        c.add('} else { // work-sharing for all levels')    
        # Adaptivork-sharing for all levels
        c.add('int start = 0;')
        c.add('int end = 0;')
        if self.nonEmptyBufferDetectionType == 'twolvlbitmaps':
            c.add('WorkSharing::TaskStack* target_stack = NULL;')
            c.add(f'char* attr = WorkSharing::Wait<{min_num_warps}>(warp_id, thread_id, target_stack, lvl, start, end, {self.worksharingThreshold}, taskbook, taskstack, num_buffered_inodes, global_bit1, global_bit2, warp_status);')
        else:
            c.add(f'char* attr = WorkSharing::Wait(thread_id, lvl, start, end, {self.worksharingThreshold}, taskbook, taskstack, warp_status);')
        # TODO two lvl bitmap and dedicated stack        
        # Terminiation code    
        c.add('if (lvl == -2) {')
        if KernelCall.args.mode == 'stats': c.add(f"stat_counters[TYPE_STATS_WAITING] += (clock64() - tp);")
        c.add('if (blockIdx.x == 0 && threadIdx.x == 0) warp_status->terminate();')
        c.add('break;')
        c.add('}')
        # ~Termination code
        c.add(f'if ((end - start) >= 32) mask_32 = 0x1 << lvl;')
        c.add(f'mask_1 = 0x1 << lvl;')
        c.add('switch (lvl) {')
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            if spSeq.inputType != 1: continue
            c.add(f'case {spSeqId}:' + '{')
            c.add(f'if (thread_id == {spSeqId}) inodes_cnts = (end - start);')
            c.add(f'ts_{spSeqId}_range_cached.start = start + thread_id;')
            c.add(f'ts_{spSeqId}_range_cached.end = end;')
            lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
            tid = spSeq.getTid()
            
            for attrId, attr in spSeq.inBoundaryAttrs.items():
                if attrId == tid.id: continue
                if attrId in lastOp.generatingAttrs: continue
                name = f'ts_{spSeqId}_{attr.id_name}'
                if attr.dataType == Type.STRING:
                    c.add(f'{name}_cached.start =  *((char**) attr);')
                    c.add(f'{name}_cached.end =  *((char**) (attr + 8));')
                else:
                    c.add(f'{name}_cached = *(({langType(attr.dataType)}*) attr);')
                c.add(f'attr += {CType.size[langType(attr.dataType)]};')  
            c.add('} break;')
        c.add('} // switch')
        c.add('} // work-sharing for all levels')
        return c

    def genKernelCodeForPipe(self, pipe_name, pipe):
        pipe.resolveAttributes()
        pipe.resolveBoundaryAttrs()
        
        c = Code()
        c.add('__global__ void')
        if self.doInterWarpLB:
            c.add('__launch_bounds__(128, 8)')
        c.add(f'krnl_{pipe_name} (')
        c.add(self.genArgsForKernelDeclarationForPipe(pipe_name, pipe))
        c.add(f'int* cnt')
        c.add(') {')
        
        attrsToDeclareAndMaterialize = self.resolveAttrsToDeclareAndMaterialize(pipe)
        innerCode = Code()
        for i in range(len(pipe.subpipeSeqs)):
            spSeq = pipe.subpipeSeqs[len(pipe.subpipeSeqs)-i-1]
            innerCode = self.genKernelCodeForSpSeq(spSeq, attrsToDeclareAndMaterialize[len(pipe.subpipeSeqs)-i-1], innerCode)
        
        pipeVarCode = pipe.genPipeVar()
        stringConstantsCode = pipe.genStringConstants()
        c.add("//===================================")
        c.add(f"//pipe {pipe.id}")
        
        pipeInitializationCode = self.genPipeIntialization(pipe)
        initialDistributionCode = self.genInitialDistribution(pipe)

        c.add(pipeInitializationCode)
        c.add(initialDistributionCode)        
        c.add(stringConstantsCode)
        c.add(pipeVarCode)
        if self.doInterWarpLB:
            c.add('do {')
            c.add(innerCode)
            if self.interWarpLbMethod == 'aws':
                c.add(self.genCodeForAdaptiveWorkSharing(pipe))
            elif self.interWarpLbMethod == 'ws':
                c.add(self.genCodeForWorkSharingPull(pipe))
            c.add('} while (true); // while loop')
        else:
            c.add(innerCode)
        c.add(self.genPipePostCode(pipe_name, pipe))
        c.add('} // End of pipe ' + f'krnl_{pipe_name}')
        
        
        return c

def compileCode( filename, code, compileOption, arch="sm_75", debug=False):
    codeGen = codeGenerator(None, None)
    codeGen.compileCode( filename, code, compileOption, arch, debug)

def genCode(dss, pipes):
    codeGen = codeGenerator(dss, pipes)
    return codeGen.genCode()