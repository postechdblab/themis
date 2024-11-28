import subprocess
import os
import sys
import copy
import time

import dogqc.identifier as ident
import dogqc.querylib as qlib
from dogqc.cudalang import *
from dogqc.variable import Variable
from dogqc.code import Code
from dogqc.code import Timestamp
from dogqc.gpuio import GpuIO
from dogqc.kernel import Kernel, KernelCall
from dogqc.types import Type
from dogqc.cudalang import CType


class CodeGenerator ( object ):

    def __init__( self, decimalRepresentation ):
        self.read = Code()
        self.types = Code()
        self.kernels = []
        self.currentKernel = None
        self.kernelCalls = []

        self.declare = Code()
        self.finish = Code()
        self.mirrorKernel = None

        self.gpumem = GpuIO ( )
        self.constCounter = 0

        self.decimalType = decimalRepresentation
   
    def langType ( self, relDataType ):
        internalTypeMap = {}
        internalTypeMap [ Type.INT ] = CType.INT
        internalTypeMap [ Type.ULL ] = CType.ULL
        internalTypeMap [ Type.DATE ] = CType.UINT
        internalTypeMap [ Type.CHAR ] = CType.CHAR
        internalTypeMap [ Type.FLOAT ] = self.decimalType
        internalTypeMap [ Type.DOUBLE ] = self.decimalType
        internalTypeMap [ Type.STRING ] = CType.STR_TYPE
        return internalTypeMap [ relDataType ]

    def stringConstant ( self, token ):
        self.constCounter += 1
        c = Variable.val ( CType.STR_TYPE, "c" + str ( self.constCounter ) ) 
        emit ( assign ( declare ( c ), call ( "stringConstant", [ "\"" + token + "\"", len(token) ] ) ), self.init() )
        return c

    def openKernel ( self, kernel ):
        self.kernels.append ( kernel )
        self.currentKernel = kernel
        self.kernelCalls.append ( KernelCall.generated ( kernel ) )
        return kernel
    
    # used for multiple passes e.g. (multi) hash build
    def openMirrorKernel ( self, suffix ):
        kernel = copy.deepcopy ( self.currentKernel )
        kernel.kernelName = self.currentKernel.kernelName + suffix
        self.kernels.append ( kernel )
        self.mirrorKernel = kernel
        self.kernelCalls.append ( KernelCall.generated ( kernel ) )
        return kernel
    
    def closeKernel ( self ):
        self.currentKernel = None

        if self.mirrorKernel:
            self.mirrorKernel = None

    def add ( self, string ):
        self.currentKernel.add ( string )
        if self.mirrorKernel:
            self.mirrorKernel.add ( string )
 
    def init ( self ):
        return self.currentKernel.init
        
    def warplane( self ):
        try:
            return self.currentKernel.warplane
        except AttributeError:
            self.currentKernel.warplane = Variable.val ( CType.UINT, "warplane" )
            emit ( assign ( declare ( self.currentKernel.warplane ), modulo ( threadIdx_x(), intConst(32) ) ), self.init() )
            return self.currentKernel.warplane

    def warpid( self ):
        try:
            return self.currentKernel.warpid
        except AttributeError:
            self.currentKernel.warpid = Variable.val ( CType.UINT, "warpid" )
            emit ( assign ( declare ( self.currentKernel.warpid ), div ( threadIdx_x(), intConst(32) ) ), self.init() )
            return self.currentKernel.warpid

    def newStatisticsCounter ( self, varname, text ):
        counter = Variable.val ( CType.UINT, varname ) 
        counter.declareAssign ( intConst(0), self.declare )
        self.gpumem.mapForWrite ( counter ) 
        self.gpumem.initVar ( counter, "0u" ) 
        self.currentKernel.addVar ( counter )
        emit ( printf ( text+"%i\\n", [ counter ]), self.finish )  
        return counter

    def prefixlanes( self ):
        try:
            return self.currentKernel.prefixlanes
        except AttributeError:
            self.currentKernel.prefixlanes = Variable.val ( CType.UINT, "prefixlanes" )
            emit ( assign ( declare ( self.currentKernel.prefixlanes ), 
                shiftRight ( bitmask32f(), sub ( intConst(32), self.warplane() ) ) ), self.init() )    
            return self.currentKernel.prefixlanes
    
    def addDatabaseAccess ( self, context, accessor ):
        self.read.add( accessor.getCodeAccessDatabase ( context.inputAttributes ) )
        self.accessor = accessor

    # build complete code file from generated pieces and add time measurements
    def composeCode( self, useCuda=True ):
        code = Code()
        code.add ( qlib.getIncludes () )
        if useCuda:
            code.add ( qlib.getCudaIncludes () )
        code.addFragment ( self.types )
        for k in self.kernels: 
            code.add(k.getKernelCode())
        code.add( "int main() {" )
        if useCuda:
            code.add ("int* cnt; cudaMalloc((void**) &cnt, sizeof(int));")
            code.add ("#ifdef MODE_PROFILE")
            code.add ("std::vector<unsigned long long> cpu_counters({});"
                .format(int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)))
            code.add ("unsigned long long* active_clocks;")
            code.add ("cudaMalloc((void**) &active_clocks, sizeof(unsigned long long) * {});"
                .format(int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)))
            code.add ("unsigned long long* active_lanes_nums;")
            code.add ("cudaMalloc((void**) &active_lanes_nums, sizeof(unsigned long long) * {});"
                .format(int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)))
            code.add ("unsigned long long* oracle_active_lanes_nums;")
            code.add ("cudaMalloc((void**) &oracle_active_lanes_nums, sizeof(unsigned long long) * {});"
                .format(int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)))
            code.add ("#endif")

            code.add('#ifdef MODE_SAMPLE')
            code.add('unsigned long long* samples;')
            code.add('cudaMalloc((void**)&samples, sizeof(unsigned long long) * {});'.format(2*int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)))
            code.add('cudaDeviceSynchronize();')
            code.add('unsigned long long* sample_start;')
            code.add('cudaMalloc((void**)&sample_start, sizeof(unsigned long long));')
            
            code.add('#endif')

        code.addUntimedFragment ( self.read, "import" )
        code.addUntimedFragment ( self.declare, "declare" )
        if self.gpumem.cudaMalloc.hasCode:
            wakeup = Code()
            comment ( "wake up gpu", wakeup ) 
            code.addUntimedCudaFragment ( wakeup, "wake up gpu" )
        code.addUntimedCudaFragment ( self.gpumem.cudaMalloc, "cuda malloc" )
        if useCuda:
            printMemoryFootprint ( code )
        code.addUntimedCudaFragment ( self.gpumem.cudaMallocHT, "cuda mallocHT" )
        if useCuda:
            printMemoryFootprint ( code )
        code.addUntimedCudaFragment ( self.gpumem.cudaMemcpyIn, "cuda memcpy in" )
        tsKernels = Timestamp ( "totalKernelTime", code )
        for call in self.kernelCalls: 
            code.addCudaFragment ( call.get(), call.kernelName, call.getAnnotations() )
            code.add('{')
            code.add('int numBlocks = 0;')
            code.add(f'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, {call.kernelName}, {KernelCall.defaultBlockSize}, 0);')
            code.add(f'printf("{call.kernelName} # blocks: %d\\n", numBlocks);')
            code.add('cudaFuncAttributes attr;')
            code.add(f'cudaFuncGetAttributes(&attr, {call.kernelName});')
            code.add('printf("Number of registers per thread: %d\\n", attr.numRegs);')
            code.add('}')
        tsKernels.stop()
        code.addUntimedCudaFragment ( self.gpumem.cudaMemcpyOut, "cuda memcpy out" )
        code.addUntimedCudaFragment ( self.gpumem.cudaFree, "cuda free" )
        code.addTimedFragment ( self.finish, "finish" )
        if useCuda:
            code.timestamps.append ( tsKernels )

        emit ( printf ( "<timing>\\n" ), code )  
        for ts in code.timestamps: 
            ts.printTime() 
        emit ( printf ( "</timing>\\n" ), code )  

        code.add("}")
        return code.content

    def writeCodeFile ( self, code, filename ):
        print(filename)
        with open(filename, 'w') as f:
            f.write( code )
        
        # format sourcecode
        cmd = "astyle --indent-col1-comments " + filename
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
   

    def compile_( self, filename, compileOption, arch="sm_52", debug=False ):
        print("compilation...")
        sys.stdout.flush()
        self.filename = filename
        cuFilename = filename + ".cu"

        self.writeCodeFile ( self.composeCode(), cuFilename )

        # compile
        nvccFlags = "-I ../../ -std=c++11 -g -arch=" + arch + " " + compileOption
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
        

    def compileCpu ( self, filename, debug=False ):
        self.filename = filename
        cppFilename = filename + ".cpp"

        self.writeCodeFile ( self.composeCode(False), cppFilename )

        # compile
        flags = "-std=c++11  -pthread -I ../../ -I ../../code/dogqc"
        if debug:
            flags += " -g"
        cmd = "g++ " + cppFilename + " -o " + filename + " " + flags
        print(cmd)
        output = subprocess.check_output(cmd, shell=True)

    def execute( self, deviceid=None, timeout=None ):  
        print("\nexecution...")
        sys.stdout.flush()
        cmd = "./" + self.filename
        #output = subprocess.check_output(cmd, shell=True, timeout=timeout).decode('utf-8')
        try:
            if deviceid != None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceid)
            proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate(timeout=timeout)
            out, err = out.decode('utf-8'), err.decode('utf-8')
            print(out)
            print(err)
            with open(self.filename + ".log", "w") as log_file:
                print(out, file=log_file)
            return (out)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            print('cmd: {}'.format(e.cmd))
            print('output:\n{}'.format(e.output))
            print('timeout: {}'.format(e.timeout))
            print('totalKernelTime: {}'.format(timeout))
            with open(self.filename + ".log", "w") as log_file:
                print(e.output, file=log_file)
                print('totalKernelTime: {}'.format(timeout*1000), file=log_file)
            time.sleep(3)
