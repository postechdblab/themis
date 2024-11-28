import sys
import subprocess
import time

from dogqc.kernel import KernelCall
from dogqc.types import Type
from dogqc.cudalang import CType

from themis.datastructure import langType
from themis.code import Code
from themis.operator import Selection, Aggregation

class codeGenerator():
    
    def __init__(self, dss, pipes):
        self.dss = dss
        self.pipes = pipes
        self.doIntraWarpLB = False
        self.doCnt = False
        self.num_result = 5
        
    def writeCodeFile ( self, code, filename ):
        with open(filename, 'w') as f:
            f.write( code )
        cmd = "astyle --indent-col1-comments " + filename
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)

    def compileCode( self, filename, code, compileOption, arch, debug ):
        print("compilation...")
        sys.stdout.flush()

        cuFilename = filename + ".cu"

        self.writeCodeFile(str(code), cuFilename)

        nvccFlags = compileOption + " -I ../../ -std=c++11 -g -arch=" + arch + " "
        hostFlags = "-pthread "
        if debug:
            nvccFlags += "-g -G "
            hostFlags += "-rdynamic "
        cmd = "nvcc " + cuFilename + " -o " + filename + " " + nvccFlags + " -Xcompiler=\"" + hostFlags + "\" "
        print(cmd)
        start = time.time()
        if debug:
            subprocess.run(cmd, shell=True)
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        end = time.time()
        print ( "compilation time: %.1f ms" % ((end-start)*1000) )
    
    def genPipeIntialization(self, pipe):
        c = Code()
        c.add('int thread_id = threadIdx.x % 32;')
        c.add('int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;')
        c.add('unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));')
        c.add('unsigned active_mask = 0;')
        c.add('int active = 0;')
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
        c.add('while (true) {')
        c.add(f'int loop{spSeq.id} = ts_{spSeq.id}_range_cached.start;')
        c.add(f'active = loop{spSeq.id} < ts_{spSeq.id}_range_cached.end;')
        c.add(f'ts_{spSeq.id}_range_cached.start += {stepSize};')
        c.add(f'active_mask = __ballot_sync(ALL_LANES, active);')
        c.add(f'if (active_mask == 0) break;')
        tid = spSeq.getTid()
        c.add(f'int {tid.id_name} = loop{spSeq.id};')

        return c
    
    def genInputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        c.add('while (true) {')
        c.add(f'int loop{spSeq.id} = ts_{spSeq.id}_range_cached.start;')
        c.add(f'active = loop{spSeq.id} < ts_{spSeq.id}_range_cached.end;')
        c.add(f'ts_{spSeq.id}_range_cached.start += {1};')
        c.add(f'active_mask = __ballot_sync(ALL_LANES, active);')
        c.add(f'if (active_mask == 0) break;')
        tid = spSeq.getTid()
        var = f'loop{spSeq.id}'
        c.add(f'int {tid.id_name};')
        c.add(f'if (active) {tid.id_name} = {spSeq.convertTid(var)};')
        return c
    
    def genInputCodeForType2(self, sqSeq, attrsToDeclareAndMaterialize):
        c = Code()
        c.add('active_mask = __ballot_sync(ALL_LANES, active);')
        c.add('if (active_mask == 0) continue;')
        return c
    
    def genInputCode(self, spSeq, attrsToDeclareAndMaterialize):
        if spSeq.inputType == 0: return self.genInputCodeForType0(spSeq, attrsToDeclareAndMaterialize)
        elif spSeq.inputType == 1: return self.genInputCodeForType1(spSeq, attrsToDeclareAndMaterialize)
        elif spSeq.inputType == 2: return self.genInputCodeForType2(spSeq, attrsToDeclareAndMaterialize)
    
    def genOutputCodeForType0(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        return c
    
    def genOutputCodeForType1(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        opId = spSeq.subpipes[-1].operators[-1].opId
        c.add(f'Range ts_{spSeq.id+1}_range_cached;')
        c.add(f'ts_{spSeq.id+1}_range_cached.set(local{opId}_range);')
        return c
    
    def genOutputCodeForType2(self, spSeq, attrsToDeclareAndMaterialize):
        c = Code()
        #lastOp = spSeq.subpipes[-1].operators[-1]
        
        # if lastOp.tid is not None and not isinstance(lastOp, Selection) and not isinstance(lastOp, Aggregation):
        #     c.add(f'// {lastOp}')
        #     c.add(f'int {lastOp.tid.id_name} = bucketId{lastOp.opId};') 
        return c
    
    def genOutputCode(self, spSeq, attrsToDeclareAndMaterialize):
        if spSeq.outputType == 0: return self.genOutputCodeForType0(spSeq, attrsToDeclareAndMaterialize)
        elif spSeq.outputType == 1: return self.genOutputCodeForType1(spSeq, attrsToDeclareAndMaterialize)
        elif spSeq.outputType == 2: return self.genOutputCodeForType2(spSeq, attrsToDeclareAndMaterialize)
    
    def resolveAttrsToMaterialize(self, materializedAttrs, subpipe):
        attrs = {}
        for attrId, attr in subpipe.getUsingAttrs().items():
            if attrId not in materializedAttrs:
                attrs[attrId] = attr

        for attrId, attr in subpipe.getMappedAttrs().items():
            if attrId not in materializedAttrs:
                attrs[attrId] = attr
        return attrs
    
    def genAttrLoad(self, pipe, attrs):
        c = Code()
        for attrId, attr in attrs.items():
            if attr.id_name == pipe.originExpr[attrId]:
                continue
            if attr.dataType == Type.STRING:
                c.add(f"{attr.id_name} = {pipe.originExpr[attrId]};")
            else:
                c.add(f"{attr.id_name} = {pipe.originExpr[attrId]};")
        return c

    def resolveAttrsToDeclare(self, materializedAttrs, subpipe):
        attrsToDeclare = {}
        for op in subpipe.operators:
            candidateAttrsToMaterialize = op.usingAttrs if KernelCall.args.lazy_materialization else op.inAttrs
            for attrId, attr in candidateAttrsToMaterialize.items():
                if attrId not in materializedAttrs:
                    attrsToDeclare[attrId] = attr
            for attrId, attr in op.mappedAttrs.items():
                if attrId not in materializedAttrs:
                    attrsToDeclare[attrId] = attr
        return attrsToDeclare

    def genAttrDeclaration(self, pipe, attrs):
        c = Code()
        for attrId, attr in attrs.items():
            if attr.dataType == Type.STRING:
                c.add(f"str_t {attr.id_name};")
            else:
                c.add(f"{langType(attr.dataType)} {attr.id_name};")
        return c
    
    def resolveMaterializableAttrsBeforeExecution(self, materializedAttrs, subpipe):
        attrs = {}
        for op in subpipe.operators:
            for attrId, attr in op.usingAttrs.items():
                if attrId in subpipe.inAttrs and attrId not in materializedAttrs:
                    attrs[attrId] = attr
        materializedAttrs.update(attrs)
        return attrs

    def resolveAttrsToMaterialize(self, materializedAttrs, op):
        candidateAttrsToMaterialize = op.usingAttrs if KernelCall.args.lazy_materialization else op.inAttrs
        attrs = {}
        for attrId, attr in candidateAttrsToMaterialize.items():
            if attrId not in materializedAttrs:
                attrs[attrId] = attr            
        materializedAttrs.update(attrs)
        return attrs

    def resolveAttrsToDeclareAndMaterialize(self, pipe):
        first_op_tid = pipe.subpipes[0].operators[0].tid
        materializedAttrs = {}
        materializedAttrs[first_op_tid.id] = first_op_tid
        infoPerSpSeqs = []
        
        print(pipe.id)
        for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
            print('\t', spSeqId)
            if self.doIntraWarpLB and spSeq.doLBforInput: materializedAttrs = {}
            print('\t',  self.doIntraWarpLB, spSeq.doLBforInput, materializedAttrs)
            materializedAttrs.update(spSeq.inBoundaryAttrs)
            print('\t', spSeq.inBoundaryAttrs)
            infoPerSubpipes = []
            for spId, subpipe in enumerate(spSeq.subpipes):
                attrsToDeclare = self.resolveAttrsToDeclare(materializedAttrs, subpipe)
                attrsToLoadBeforeExec = self.resolveMaterializableAttrsBeforeExecution(materializedAttrs, subpipe)
                attrsToMaterialize = []                
                for opId, op in enumerate(subpipe.operators):
                    attrsToMaterialize.append(self.resolveAttrsToMaterialize(materializedAttrs, op))            
                    if op.tid is not None:
                        materializedAttrs[op.tid.id] = op.tid
                    for attrId, attr in op.mappedAttrs.items():
                        materializedAttrs[attrId] = attr
                    
                currentMaterializedAttrs = {}
                currentMaterializedAttrs.update(materializedAttrs)
                infoPerSubpipes.append([attrsToDeclare, attrsToLoadBeforeExec, attrsToMaterialize, currentMaterializedAttrs])
                print('\t\t', spId)
                print('\t\t', attrsToDeclare, attrsToLoadBeforeExec, attrsToMaterialize)
            infoPerSpSeqs.append(infoPerSubpipes)
            
        return infoPerSpSeqs


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
            c.add(self.genAttrDeclaration(spSeq.pipe, attrsToDeclare))
            
            # Declare local variables
            c.add(subpipe.genLocalVar(attrsToDeclare))
            c.add('if (active) {')
            
            # Load attrs which we can load before the execution of this subpipe
            c.add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

            for opId, op in enumerate(subpipe.operators):
                c.add(self.genAttrLoad(spSeq.pipe, attrsToMaterialize[opId]))
                c.add(opCodes[opId])
            c.add('} // end of active')
        c.add(self.genOutputCode(spSeq, attrsToDeclareAndMaterialize))
        c.add(innerCode)
        if spSeq.inputType == 0 or spSeq.inputType == 1:
            c.add('}')
        return c
    
    
    def genPipePostCode(self, pipe_name, pipe):
        c = Code()
        for subpipe in pipe.subpipes:
            for op in subpipe.operators:
                c.add(op.genPipePostCode())
        return c
    
    def genKernelCodeForPipe(self, pipe_name, pipe):
        pipe.resolveAttributes()
        pipe.resolveBoundaryAttrs()
        
        c = Code()
        c.add('__global__ void')
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
        c.add(innerCode)
        c.add(self.genPipePostCode(pipe_name, pipe))
        c.add('} // End of pipe ' + f'krnl_{pipe_name}')
        
        return c

    def genHeaders(self):
        c = Code()
        c.add("#include <chrono>")
        c.add("#include <list>")
        c.add("#include <unordered_map>")
        c.add("#include <vector>")
        c.add("#include <iostream>")
        c.add("#include <ctime>")
        c.add("#include <limits.h>")
        c.add("#include <float.h>")
        c.add('#include "dogqc/include/csv.h"')
        c.add('#include "dogqc/include/mappedmalloc.h"')
        c.add('#include "themis/include/range.cuh"')
        c.add('#include "themis/include/themis.cuh"')
        c.add('#include "themis/include/work_sharing.cuh"')
        c.add('#include "themis/include/adaptive_work_sharing.cuh"')
        return c

    def genArgsForKernelDeclarationForPipe(self, pipe_name, pipe):
        return pipe.dss.genArgsForKernelDeclaration()
        
    def genArgsForKernelCallForPipe(self, pipe_name, pipe):
        return pipe.genArgsForKernelCall()

    def genKernelCallCodeForPipe(self, pipe_name, pipe):
        c = Code()
        c.add(pipe.genCodeBeforeExecution())
        c.add('cudaDeviceSynchronize();')
        c.add(f'krnl_{pipe_name}<<<{KernelCall.defaultGridSize},{KernelCall.defaultBlockSize}>>>(')
        c.add(self.genArgsForKernelCallForPipe(pipe_name, pipe))
        c.add('cnts')
        c.add(');')
        c.add('cudaDeviceSynchronize();')
        c.add(pipe.genCodeAfterExecution())
        return c
        
    def genResultPrintcode(self, pipes):
        c = Code()
        lastOp = pipes[-1].subpipes[-1].operators[-1]
        c.add('std::clock_t start_copyTime = std::clock();')
        c.add('int cpu_nout_result;')
        c.add('cudaMemcpy(&cpu_nout_result, nout_result, sizeof(int), cudaMemcpyDeviceToHost);')
        for attrId, attr in lastOp.usingAttrs.items():
            c.add(f'std::vector<{langType(attr.dataType)}> cpu_result_{attr.id_name}(cpu_nout_result);')
            c.add(f'cudaMemcpy(cpu_result_{attr.id_name}.data(), result_{attr.id_name}, sizeof({langType(attr.dataType)}) * cpu_nout_result, cudaMemcpyDeviceToHost);')
        c.add('cudaDeviceSynchronize();')
        c.add('std::clock_t stop_copyTime = std::clock();')
        c.add('printf ( "copyTime: %6.1f ms\\n", (stop_copyTime - start_copyTime) / (double) (CLOCKS_PER_SEC / 1000) );')
        c.add('printf("cout_nout_result: %d\\n", cpu_nout_result);')
        c.add(f'for (int pv = 0; pv < {self.num_result} && pv < cpu_nout_result; ++pv) ' + '{')
        for attrId, attr in lastOp.usingAttrs.items():
            c.add(f'printf("{attr.id_name}: ");')
            if attr.dataType != Type.STRING:
                c.add(f'printf("{CType.printFormat[langType(attr.dataType)]}, ", cpu_result_{attr.id_name}[pv]);')
            else:
                c.add(f'printf(", ");')
                #c.add(f'stringPrint(cpu_result_{attr.id_name}[pv]);')
        c.add('printf("\\n");')
        c.add('}')
        return c 

    def genElapsedTimePrintCode(self, num_kernel_calls):
        c = Code()
        for i in range(num_kernel_calls):
            #c.add('{')
            c.add(f'std::chrono::duration<double, std::micro> elapsed{i} = end_timepoint{i} - start_timepoint{i};') 
            c.add(f'printf ( "%32s: %6.1f ms\\n", "KernelTime{i}", ((float) elapsed{i}.count()) / 1000 );')
            #c.add('}')
        #c.add('{')
        c.add(f'std::chrono::duration<double, std::micro> elapsed = end_timepoint - start_timepoint;') 
        c.add(f'printf ( "%32s: %6.1f ms\\n", "totalKernelTime0", ((float) elapsed.count()) / 1000 );')
        #c.add('}')
        return c

    def findLowerLoopLvl(self, spSeq):
        loop_lvl = spSeq.id
        while True:
            inputType = spSeq.pipe.subpipeSeqs[loop_lvl].inputType
            if inputType == 0 or inputType == 1:
                return loop_lvl
            loop_lvl -= 1

    def genDeclarationInMain(self):
        return self.dss.genDeclarationInMain()
    
    def genDeclarationInGlobal(self):
        return self.dss.genDeclarationInGlobal()


    def genCode(self):
        c = Code()
        c.add(self.genHeaders())
        codeDeclarationInGlobal = self.genDeclarationInGlobal()
        c.add("//====================================")
        c.add("//Global declaration")
        c.add("//------------------------------------")
        c.add(str(codeDeclarationInGlobal))
        c.add("//====================================")
        
        # Generate codes for kernels and kernel calls
        kernelCodes, kernelCallCodes = [], []
        pid = 0
        for pipe in self.pipes:
            pipe_name = str(pid)
            kernelCodes.append(self.genKernelCodeForPipe(pipe_name, pipe))
            kernelCallCodes.append(self.genKernelCallCodeForPipe(pipe_name, pipe))
            pid += 1
            if pipe.isLastOperatorMultiHashJoin():
                pipe_name = str(pid)
                print('pipe_name', pid)
                kernelCodes.append(self.genKernelCodeForPipe(pipe_name, pipe))
                kernelCallCodes.append(self.genKernelCallCodeForPipe(pipe_name, pipe))
                pid +=1
        
        c.add("//====================================")
        c.add("//Kernel codes")
        c.add("//------------------------------------")
        for kernelCode in kernelCodes:
            c.add(kernelCode)        
        c.add("//====================================")
        
        c.add("int main() {")
        codeDeclarationInMain = self.genDeclarationInMain()
        c.add("//====================================")
        c.add("//Main declaration")
        c.add("//------------------------------------")
        c.add(str(codeDeclarationInMain))
        c.add("//====================================")
        
        c.add("int* cnts = NULL;")
        c.add("int cpu_cnts[32];")
        if self.doCnt:
            c.add("cudaMalloc((void**)&cnts, 32 * sizeof(int));")
        
        c.add('cudaDeviceSynchronize();')
        c.add("std::chrono::steady_clock::time_point start_timepoint, end_timepoint;")
        c.add('start_timepoint = std::chrono::steady_clock::now();')
        for kid, kernelCallCode in enumerate(kernelCallCodes):
            if self.doCnt:
                c.add('cudaMemset(cnts, 0, 32 * sizeof(int));')
            c.add(f"std::chrono::steady_clock::time_point start_timepoint{kid}, end_timepoint{kid};")
            c.add('{')
            c.add(f"start_timepoint{kid} = std::chrono::steady_clock::now();")
            c.add(kernelCallCode)
            c.add(f"end_timepoint{kid} = std::chrono::steady_clock::now();")
            c.add('cudaDeviceSynchronize();')
            if self.doCnt:
                c.add('cudaMemcpy(&cpu_cnts, cnts, 32 * sizeof(int), cudaMemcpyDeviceToHost);')
                c.add('for (int i = 0; i < 32; ++i) printf("%d ", cpu_cnts[i]);')
                c.add('printf("\\n");')
            c.add('}')
                
        c.add(f"end_timepoint = end_timepoint{len(kernelCallCodes)-1};")
        
        # Print result
        resultPrintCode = self.genResultPrintcode(self.pipes)
        c.add(resultPrintCode)
        elapsedTimePrintCode = self.genElapsedTimePrintCode(len(kernelCallCodes))
        c.add(elapsedTimePrintCode)
        c.add("return 0;")
        c.add("}")
                
        f_code = open('./code.txt', 'w')
        f_code.write(str(c))
        f_code.close()
        return c

def compileCode( filename, code, compileOption, arch="sm_75", debug=False):
    codeGen = codeGenerator(None, None)
    codeGen.compileCode( filename, code, compileOption, arch, debug)

def genCode(dss, pipes):
    codeGen = codeGenerator(dss, pipes)
    return codeGen.genCode()