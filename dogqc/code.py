from dogqc.cudalang import *


class Timestamp ( object ):
  
    id=0
    
    def __init__ ( self, name, code, annotations="" ):
        
        self.label = name
        self.name = name.replace(" ", "_") + str(Timestamp.id)
        
        print(self.name)
        
        self.startTime = "start_" + self.name
        self.stopTime = "stop_" + self.name
        self.elapsedTime = "elapsed_time_" + self.name
        self.code = code
        #self.code.add("cudaDeviceSynchronize();")
        #self.code.add(f"cudaEvent_t {self.startTime}, {self.stopTime};")
        #self.code.add(f"float {self.elapsedTime};")
        #self.code.add(f"cudaEventCreate(&{self.startTime});")
        #self.code.add(f"cudaEventCreate(&{self.stopTime});")
        #self.code.add(f"cudaEventRecord({self.startTime});")
        self.code.add("std::clock_t " + self.startTime + " = std::clock();")
        
        self.annotations = annotations
        Timestamp.id += 1

    def stop ( self ):
        #self.code.add(f"cudaEventRecord({self.stopTime});")
        #self.code.add(f"cudaEventSynchronize({self.stopTime});")
        #self.code.add(f"cudaEventElapsedTime(&{self.elapsedTime}, {self.startTime}, {self.stopTime});")
        self.code.add("std::clock_t " + self.stopTime + " = std::clock();")
    
    def printTime(self ):
        if self.annotations == "":
            msg = self.label
        else:
            msg = self.label + " " + self.annotations
        #self.code.add ( "printf ( \"%32s: %6.1f ms\\n\", \"" + msg + "\"," + self.elapsedTime + ");" )
        self.code.add ( "printf ( \"%32s: %6.1f ms\\n\", \"" + msg + "\", (" + self.stopTime + " - " + self.startTime + ") / (double) (CLOCKS_PER_SEC / 1000) );" )


class Code ( object ):

    def __init__( self ):
        self.content = ""
        self.timestamps = []
        self.hasCode = False

    def add(self, line):
        self.content = self.content + str(line)
        if isinstance ( line, str ):
            self.content += "\n"
        self.hasCode = True
    
    def addFragment ( self, fragment ):
        if not fragment.hasCode:
            return
        self.add( fragment ) 
        self.add("")
    
    def addTimedFragment ( self, fragment, name ):
        if not fragment.hasCode:
            return
        ts = Timestamp ( name, self )
        self.add( fragment ) 
        ts.stop ()    
        self.add("")
        self.timestamps.append ( ts )
    
    def addUntimedFragment ( self, fragment, name ):
        if not fragment.hasCode:
            return
        self.add( fragment ) 
        self.add("")
    
    def addUntimedCudaFragment ( self, fragment, name, annotations="" ):
        if not fragment.hasCode:
            return
        self.add( fragment ) 
        emit ( deviceSynchronize(), self )
        cudaErrorCheck( name, self )
        self.add("")
         
    def addCudaFragment ( self, fragment, name, annotations="" ):
        if not fragment.hasCode:
            return
        ts = Timestamp ( name, self, annotations )
        self.add( fragment ) 
        emit ( deviceSynchronize(), self )
        ts.stop ()    
        cudaErrorCheck( name, self )
        self.add("")
        self.timestamps.append ( ts )

    def __str__ ( self ):
        return self.content



