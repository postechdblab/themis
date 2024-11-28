from dogqc.cudalang import *
from dogqc.code import Code
import dogqc.identifier as ident




class KernelCall ( object ):

    defaultGridSize = 1024
    defaultBlockSize = 128

    def __init__ ( self, gridSize, blockSize ):
        self.blockSize = KernelCall.defaultBlockSize
        self.gridSize = KernelCall.defaultGridSize
        if blockSize != None:
            self.blockSize = blockSize
        if gridSize != None:
            self.gridSize = gridSize

    def generated ( kernel, gridSize=None, blockSize=None ):
        call = KernelCall ( gridSize, blockSize )
        call.kernel = kernel
        call.kernelName = kernel.kernelName
        return call 
    
    def library ( kernelName, parameters, templateParameters="", gridSize=None, blockSize=None ):
        call = KernelCall ( gridSize, blockSize )
        call.kernel = None
        call.kernelName = kernelName
        call.parameters = parameters
        call.templateParameters = templateParameters
        return call 

    def get ( self ):
        if self.kernel != None:
            return KernelCall.generic ( self.kernel.kernelName, self.kernel.getParameters(), self.gridSize, self.blockSize )
        else:
            return KernelCall.generic ( self.kernelName, self.parameters, self.gridSize, self.blockSize )

    def getAnnotations ( self ):
        if self.kernel != None and len(self.kernel.annotations) > 0:
            return " ".join(self.kernel.annotations)
        else:
            return ""


    def generic ( kernelName, parameters, gridSize=1024, blockSize=128, templateParams="" ):
        # kernel invocation parameters
        code = Code()
    
        templatedKernel = kernelName
        if templateParams != "":
            templatedKernel += "<" + templateParams + ">"
    
        with Scope ( code ):
            emit ( "int gridsize=" + str(gridSize), code )
            emit ( "int blocksize=" + str(blockSize), code )
            #emit ( "int* cnt; cudaMalloc((void**)&cnt, sizeof(int));", code )
            emit ( "cudaMemset(cnt, 0, sizeof(int))", code )
            
            if kernelName[:5] == "krnl_":
                code.add('#ifdef MODE_SAMPLE')
                code.add('\tkrnl_sample_start<<<1,32>>>(sample_start);')
                code.add('cudaDeviceSynchronize();')
                code.add('#endif')
                call = templatedKernel + "<<<gridsize, blocksize>>>(cnt,\n#ifdef MODE_PROFILE\nactive_clocks,\nactive_lanes_nums,\noracle_active_lanes_nums,\n#endif\n"
                call += "#ifdef MODE_SAMPLE\nsamples,sample_start,\n#endif\n"
            else:
                call = templatedKernel + "<<<gridsize, blocksize>>>("
            # add parameters: input attributes, output attributes and additional variables (output number)
            comma = False
            for a in parameters:
                if not comma:
                    comma = True
                else:
                    call += ", " 
                call += str(a)
            call += ")"
            emit ( call, code )
            if kernelName[:5] == "krnl_":
                code.add('#ifdef MODE_PROFILE')

                for counter in ['active_clocks', 'active_lanes_nums', 'oracle_active_lanes_nums']:

                    emit ( "cudaMemcpy(cpu_counters.data(), {}, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost)"
                        .format(counter, int(KernelCall.defaultBlockSize * KernelCall.defaultGridSize / 32)), code )
                    emit ( 'printf("{} {}:")'.format(kernelName, counter), code )
                    code.add ( 'for (int i = 0; i < {}; ++i)'.format(int(KernelCall.defaultBlockSize * KernelCall.defaultGridSize / 32)) + ' {' )
                    code.add ( 'printf(" %lld", cpu_counters[i]);' )
                    code.add ( '}' )
                    emit ( 'printf("\\n")', code )
                code.add('#endif')
                code.add('#ifdef MODE_SAMPLE')        
                size = int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)
                code.add('std::vector<unsigned long long> cpu_samples({});'.format(2 * size))
                code.add('cudaMemcpy(cpu_samples.data(), samples, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(2 * size))
                code.add('for (int w = 0; w < {}; ++w) '.format(size) + ' {')
                code.add('\tprintf("sample {} %d: ", w);'.format(kernelName))
                code.add('\tunsigned long long* d = (&cpu_samples[w * 2]);')
                code.add('\tfor (int e = 0; e < 2; ++e) {')
                code.add('\t\tprintf(" %lld/%lld/%d/%lld", d[e] & 0x000FFFFFFFFFFFFF, d[e] >> 52 & 0xFF, 0, d[e] >> 60);')
                code.add('\t}')
                code.add('\tprintf("\\n");')
                code.add('}')
                code.add('#endif')

                emit ( "int cpu_cnt = 0; cudaMemcpy(&cpu_cnt, cnt, sizeof(int), cudaMemcpyDeviceToHost)", code )
                emit ( 'std::cout << "{} result:" << cpu_cnt << std::endl;'.format(kernelName), code )
        return code

    





class Kernel ( object ):
    
    def __init__ ( self, name ):
        self.init = Code()
        self.body = Code()
        self.inputColumns = {}
        self.outputAttributes = []
        self.variables = []
        self.kernelName = name
        self.annotations = []

    def add ( self, code ):
        self.body.add( code )
    
    def addVar ( self, c ):
        # resolve multiply added columns
        self.inputColumns [ c.get() ] = c

    def getParameters ( self ):
        params = []
        for name, c in self.inputColumns.items():
            params.append ( c.getGPU() )
        for a in self.outputAttributes:
            params.append ( ident.gpuResultColumn( a ) )
        for v in self.variables:
            params.append( v.getGPU() )
        return params

    def getKernelCode( self ):
        kernel = Code()
        
        # open kernel frame
        if self.kernelName[:5] == "krnl_":
            kernel.add("__global__ void ")
            #if KernelCall.defaultBlockSize > 4:
            #    kernel.add("__launch_bounds__({}, 8) ".format(KernelCall.defaultBlockSize))
            kernel.add(self.kernelName + "(int* cnt,")
            kernel.add("#ifdef MODE_PROFILE")
            kernel.add("unsigned long long* active_clocks,")
            kernel.add("unsigned long long* active_lanes_nums, unsigned long long *oracle_active_lanes_nums,")
            kernel.add("#endif")
            kernel.add('#ifdef MODE_SAMPLE')
            kernel.add('unsigned long long* samples, unsigned long long* sample_start,')
            kernel.add('#endif')
        else:
            kernel.add("__global__ void " + self.kernelName + "(")
        comma = False
        params = ""
        for name, c in self.inputColumns.items():
            if not comma:
                comma = True
            else:
                params += ", " 
            params += c.dataType + "* " + c.get()
        for a in self.outputAttributes:
            params += ", " 
            params += a.dataType + "* " + ident.resultColumn( a )
        for v in self.variables:
            params += ", " 
            params += v.dataType + "* " + v.get()
        kernel.add( params + ") {")

        if self.kernelName[:5] == "krnl_":
            kernel.add("#ifdef MODE_PROFILE")
            kernel.add("unsigned long long active_clock = clock64();")
            kernel.add("unsigned long long active_lanes_num = 0;")
            kernel.add("unsigned long long oracle_active_lanes_num = 0;")

            kernel.add("#endif")

            kernel.add('#ifdef MODE_SAMPLE')
            kernel.add('if (threadIdx.x % 32 == 0) {')
            kernel.add('uint32_t smid32;')
            kernel.add('asm volatile("mov.u32 %0, %%smid;" : "=r"(smid32));')
            kernel.add('uint64_t smid = (uint64_t) smid32;')
            kernel.add('unsigned long long t = 1;')
            kernel.add('unsigned long long cck = clock64();')
            kernel.add('samples += (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 2;')
            kernel.add('samples[0] = (cck > (*sample_start) ? cck - (*sample_start) : 0) | (t << 60) | (smid << 52);')
            kernel.add('}')
            kernel.add('#endif')

        # add code generated by operator tree
        kernel.add(self.init.content)
        
        # add code generated by operator tree
        kernel.add(self.body.content)
        
        # close kernel frame
        if self.kernelName[:5] == "krnl_":
            kernel.add("#ifdef MODE_PROFILE")
            kernel.add("if (threadIdx.x % 32 == 0) {")
            kernel.add("active_clocks[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = clock64() - active_clock;")
            kernel.add("active_lanes_nums[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = active_lanes_num;")
            kernel.add("oracle_active_lanes_nums[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = oracle_active_lanes_num;")
            kernel.add("}")
            kernel.add("#endif")

            kernel.add('#ifdef MODE_SAMPLE')
            kernel.add('if (threadIdx.x % 32 == 0) {')
            kernel.add('uint32_t smid32;')
            kernel.add('asm volatile("mov.u32 %0, %%smid;" : "=r"(smid32));')
            kernel.add('uint64_t smid = (uint64_t) smid32;')
            kernel.add('unsigned long long t = 1;')
            kernel.add('unsigned long long cck = clock64();')
            kernel.add('samples[1] = (cck > (*sample_start) ? cck - (*sample_start) : 0) | (t << 60) | (smid << 52);')
            kernel.add('}')
            kernel.add('#endif')

            
        kernel.add("}") 
        return kernel.content

    def annotate ( self, msg ):
        self.annotations.append(msg)


