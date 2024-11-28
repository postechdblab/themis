from dogqc.translatorBase import UnaryTranslator
from dogqc.cudalang import *
from dogqc.variable import Variable
from dogqc.kernel import KernelCall
import dogqc.querylib as qlib
import copy
from dogqc.translatorBase import AttributeLocation
from dogqc.relationalAlgebra import Join

def profile_lane_activity(ctxt):
    emit ( "#ifdef MODE_PROFILE", ctxt.codegen )
    emit ( "{", ctxt.codegen )
    emit ( "int n_active_lanes = __popc(__ballot_sync(ALL_LANES, active))", ctxt.codegen )
    emit ( "active_lanes_num += n_active_lanes", ctxt.codegen )
    emit ( "oracle_active_lanes_num += n_active_lanes > 0 ? 32 : 0;", ctxt.codegen )
    emit ( "}", ctxt.codegen )
    emit ( "#endif", ctxt.codegen )

class IndexScanTranslator ( UnaryTranslator ):

    def produce ( self, ctxt ):
        self.child.produce ( ctxt )

    def consume ( self, ctxt ):
        algExpr = self.algExpr
        opId = algExpr.opId
        commentOperator ("index scan", opId, ctxt.codegen)

        for (id, a) in self.algExpr.scanAttributes.items():
            #print("scan attribute ", a.name, a.tableName)
            if a.lastUse <= self.algExpr.opId: continue
            if a.id == self.algExpr.tid.id:
                ctxt.attFile.locFile[id] = AttributeLocation.TABLE
            else:
                ctxt.attFile.mapInputAttribute(a, self.algExpr.table )

        idx_name = algExpr.idx_name

        variables = []
        attrs = algExpr.buildKeyAttributes.values()

        for idx, attr in enumerate(attrs):
            if idx + 1 < len(attrs):
                variables.append( Variable (CType.INT, '{}_{}_offset'.format(idx_name, attr.name), algExpr.sizes[attr.name] * 2) )
            else:
                variables.append( Variable (CType.INT, '{}_{}_dir_offset'.format(idx_name, attr.name), algExpr.sizes[attr.name] * 2) )
            variables.append( Variable (CType.INT, '{}_{}_val'.format(idx_name, attr.name), algExpr.sizes[attr.name]) )

        variables.append(
            Variable (CType.INT, idx_name + "_position", algExpr.table['size'])
        )

        for a in variables :
            baseIndexFileSys = "{}/{}".format(ctxt.dbpath, a.name)
            #print(baseIndexFileSys)
            if baseIndexFileSys not in ctxt.attFile.indexColumns:
                a.declarePointer ( ctxt.codegen.read )
                emit ( assign ( a, mmapFile ( a.dataType, baseIndexFileSys ) ), ctxt.codegen.read )
                ctxt.codegen.gpumem.mapForRead ( a )
                ctxt.attFile.indexColumns [ baseIndexFileSys ] = a.getGPU()
            else:
                pass
            ctxt.codegen.currentKernel.addVar(a)


        firstSize = algExpr.sizes[list(algExpr.buildKeyAttributes.values())[0].name]
        offsetVarBuf = Variable.val ( CType.INT, "matchOffsetBuf" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
        endVarBuf = Variable.val ( CType.INT, "matchEndBuf" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
        probeActive = Variable.val ( CType.INT, "probeActive" + str ( self.algExpr.opId ), ctxt.codegen, ctxt.vars.activeVar )
        
        profile_lane_activity(ctxt)


        for idx, ((attrId_b, attr_b), (attrId_p, attr_p)) in enumerate(zip(algExpr.buildKeyAttributes.items(), algExpr.probeKeyAttributes.items())):
            with IfClause ( probeActive, ctxt.codegen ):
                var = ctxt.attFile.regFile [ attrId_p ]
                if idx == 0:
                    emit ( assign (endVarBuf, firstSize), ctxt.codegen)
                emit ( assign ( 
                    probeActive, call ( qlib.Fct.INDEX_PROBE_MULTI,
                        ['{}_{}_val'.format(algExpr.idx_name, attr_b.name), 
                        '{}_{}_{}offset'.format(algExpr.idx_name, attr_b.name, 'dir_' if idx + 1 == len(algExpr.buildKeyAttributes) else ''),
                        offsetVarBuf, endVarBuf, var])), ctxt.codegen )

        if not algExpr.multimatch:
            tid = Variable.val ( CType.INT, "tid_{}{}".format(self.algExpr.table['name'], self.algExpr.scanTableId ), ctxt.codegen, intConst(0) )
            with IfClause ( probeActive, ctxt.codegen ):
                #emit ( assign ( payl, self.htmem.payload.arrayAccess ( self.offsetVar ) ), ctxt.codegen )
                #self.payload.dematerialize ( payl, ctxt )
                #Hash.checkEquality ( ctxt.vars.activeVar, self.algExpr.buildKeyAttributes, self.algExpr.probeKeyAttributes, ctxt )
                #emit ( assignAdd ( matchFound, ctxt.vars.activeVar ), ctxt.codegen )
                emit ( assign (tid, call( qlib.Fct.INDEX_GET_TID, ["{}_position".format(algExpr.idx_name), offsetVarBuf] ) ), ctxt.codegen )
                for id, a in self.algExpr.scanAttributes.items():
                    #if a.name == 'tid': continue
                    if a.lastUse <= self.algExpr.opId: continue
                    #print("demat", a.name, a.tableName)
                    if id == self.algExpr.tid.id:
                        regVar = ctxt.attFile.declareRegister ( a )
                        emit ( assign ( regVar, "tid_{}{}".format(self.algExpr.tid.tableName, self.algExpr.scanTableId) ), ctxt.codegen )
                    else:
                        ctxt.attFile.dematerializeAttribute (a, tid, self.algExpr.scanTableId )
            self.parent.consume ( ctxt )
        else:
            endVar = Variable.val ( CType.INT, "matchEnd" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            offsetVar = Variable.val ( CType.INT, "matchOffset" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            
            bufferAtts = dict()
            bufferAtts.update ( self.algExpr.child.outRelation )
            bufferVars = []
            for id, att in bufferAtts.items():
                if att.lastUse > self.algExpr.opId:
                    print(id, att.name, att.tableName)
                    var = ctxt.attFile.regFile [ id ]
                    bufVar = copy.deepcopy ( var )
                    bufVar.name = bufVar.name + "_bcbuf" + str ( self.algExpr.opId )
                    bufVar.declare ( ctxt.codegen )
                    bufferVars.append ( (var, bufVar ) )
            
            

            activeProbes = Variable.val ( CType.UINT, "activeProbes" + str ( self.algExpr.opId ) )
            activeProbes.declareAssign ( ballotIntr ( qlib.Const.ALL_LANES, probeActive ), ctxt.codegen )

            
            numbuf = Variable.val ( CType.INT, "num" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            emit ( assign ( numbuf, sub ( endVarBuf, offsetVarBuf ) ), ctxt.codegen )      
            wideProbes = Variable.val ( CType.UINT, "wideProbes"  + str ( self.algExpr.opId ))
            wideProbes.declareAssign ( ballotIntr ( qlib.Const.ALL_LANES, largerEqual ( numbuf, intConst(32) ) ), ctxt.codegen )


            # write register state to buffer to prevent overwriting 
            for var, bufVar in bufferVars:
                emit ( assign ( bufVar, var ), ctxt.codegen )

            
            with WhileLoop ( larger ( activeProbes, intConst(0) ), ctxt.codegen ):
                
                if True:
                    tupleLane = Variable.val ( CType.UINT, "tupleLane", ctxt.codegen )
                    emit ( assign ( tupleLane, sub ( ffsIntr ( activeProbes ), 1 ) ), ctxt.codegen ) 
                    # shuffle gather offset
                    emit ( assign ( offsetVar, add ( shuffleIntr ( qlib.Const.ALL_LANES, offsetVarBuf, tupleLane ), ctxt.codegen.warplane() ) ), ctxt.codegen )
                    # shuffle gather end
                    emit ( assign ( endVar, shuffleIntr ( qlib.Const.ALL_LANES, endVarBuf, tupleLane ) ), ctxt.codegen )      
                    # shuffle other register vars
                    for var, bufVar in bufferVars:
                        emit ( assign ( var, shuffleIntr ( qlib.Const.ALL_LANES, bufVar, tupleLane ) ), ctxt.codegen )      
                    # mark lane as processed
                    emit ( assignSub ( activeProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 

                    emit ( assign ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                
                else:
                    
                    tupleLane = Variable.val ( CType.UINT, "tupleLane", ctxt.codegen )
                    broadcastLane = Variable.val ( CType.UINT, "broadcastLane", ctxt.codegen )
                    numFilled = Variable.val ( CType.INT, "numFilled", ctxt.codegen, intConst(0) )
                    num = Variable.val ( CType.INT, "num", ctxt.codegen, intConst(0) )
                    with WhileLoop ( andLogic ( smaller ( numFilled, intConst(32) ), activeProbes ), ctxt.codegen ) as l: 
                        # select leader
                        with IfClause ( larger ( wideProbes, intConst (0) ), ctxt.codegen ):
                            emit ( assign ( tupleLane, sub ( ffsIntr ( wideProbes ), 1 ) ), ctxt.codegen ) 
                            emit ( assignSub ( wideProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 
                        with ElseClause ( ctxt.codegen ):
                            emit ( assign ( tupleLane, sub ( ffsIntr ( activeProbes ), 1 ) ), ctxt.codegen ) 
                        # broadcast leader number of matches
                        emit ( assign ( num, shuffleIntr ( qlib.Const.ALL_LANES, numbuf, tupleLane ) ), ctxt.codegen )
                        with IfClause ( andLogic ( numFilled, larger ( add ( numFilled, num ), 32 ) ), ctxt.codegen ):
                            l.break_()
                        with IfClause ( largerEqual ( ctxt.codegen.warplane(), numFilled ), ctxt.codegen ):
                            emit ( assign ( broadcastLane, tupleLane ), ctxt.codegen )
                            emit ( assign ( offsetVar, sub ( ctxt.codegen.warplane(), numFilled ) ), ctxt.codegen )      
                        emit ( assignAdd ( numFilled, num ), ctxt.codegen )  
                        # mark buffered probe tuple as processed            
                        emit ( assignSub ( activeProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 

                    # shuffle gather offset
                    emit ( assignAdd ( offsetVar, shuffleIntr ( qlib.Const.ALL_LANES, offsetVarBuf, broadcastLane ) ), ctxt.codegen )
                    # shuffle gather end
                    emit ( assign ( endVar, shuffleIntr ( qlib.Const.ALL_LANES, endVarBuf, broadcastLane ) ), ctxt.codegen )      
                    # shuffle other register vars
                    for var, bufVar in bufferVars:
                        emit ( assign ( var, shuffleIntr ( qlib.Const.ALL_LANES, bufVar, broadcastLane ) ), ctxt.codegen )      

                    emit ( assign ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                
                
                ctxt.innerLoopCount += 1
                with WhileLoop ( anyIntr ( qlib.Const.ALL_LANES, probeActive ), ctxt.codegen ):
                    flushPipeline = Variable.val ( CType.UINT, "flushPipeline" + str ( self.algExpr.opId ) )
                    flushPipeline.declareAssign ( 
                        andLogic ( 
                            equals(activeProbes, intConst(0)), 
                            equals(ballotIntr ( qlib.Const.ALL_LANES, smaller(add(offsetVar,intConst(32)), endVar )), intConst(0))), 
                            ctxt.codegen )
                    ctxt.vars.divergenceFlushVar = flushPipeline

                    #emit ( assign ( ctxt.vars.activeVar, probeActive ), ctxt.codegen )
                    #ctxt.codegen.laneActivityProfile ( ctxt )
                    emit ( assign ( ctxt.vars.activeVar, intConst(0) ), ctxt.codegen )
                    #payl = Variable.val ( self.htmem.payload.dataType, "payl", ctxt.codegen )
                    tid = Variable.val ( CType.INT, "tid_{}{}".format(self.algExpr.table['name'], self.algExpr.scanTableId ), ctxt.codegen, intConst(0) )
                    with IfClause ( probeActive, ctxt.codegen ):
                        #emit ( assign ( payl, self.htmem.payload.arrayAccess ( offsetVar ) ), ctxt.codegen )
                        #self.payload.dematerialize ( payl, ctxt )
                        #for
                        emit ( assign (tid, call( qlib.Fct.INDEX_GET_TID, ["{}_position".format(algExpr.idx_name), offsetVar] ) ), ctxt.codegen )
                        for id, a in algExpr.scanAttributes.items():
                            #if a.name == 'tid': continue
                            if a.lastUse <= algExpr.opId: continue
                            #print("demat", a.name, a.tableName)
                            if id == algExpr.tid.id:
                                regVar = ctxt.attFile.declareRegister ( a )
                                emit ( assign ( regVar, "tid_{}{}".format(algExpr.tid.tableName, algExpr.scanTableId) ), ctxt.codegen )
                            else:
                                ctxt.attFile.dematerializeAttribute (a, tid, algExpr.scanTableId )

                        emit ( assign ( ctxt.vars.activeVar, intConst(1) ), ctxt.codegen )
                        emit ( assignAdd ( offsetVar, intConst(32) ), ctxt.codegen )        
                        emit ( assignAnd ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                    self.parent.consume ( ctxt )
                ctxt.innerLoopCount -= 1
            
            for var, bufVar in bufferVars:
                emit ( assign ( var, bufVar ), ctxt.codegen )



class IndexJoinTranslator ( UnaryTranslator ):

    def produce ( self, ctxt ):
        self.child.produce ( ctxt )

    def consume ( self, ctxt ):
        commentOperator ("index join", self.algExpr.opId, ctxt.codegen)

        for (id, a) in self.algExpr.scanAttributes.items():
            #print("scan attribute ", a.name, a.tableName)
            if a.lastUse <= self.algExpr.opId: continue
            if a.id == self.algExpr.tid.id:
                ctxt.attFile.locFile[id] = AttributeLocation.TABLE
            else:
                ctxt.attFile.mapInputAttribute(a, self.algExpr.table )

        cond_attr = list(self.algExpr.conditionAttributes.values())[0]
        rel_name = self.algExpr.rel_name

        variables = [ 
                Variable ( CType.INT, rel_name + "_offset", cond_attr.numElements * 2 ),
                Variable ( CType.INT, rel_name + "_position", self.algExpr.table["size"])
            ]

        for a in variables :
            baseIndexFileSys = "{}/{}".format(ctxt.dbpath, a.name)
            #print(baseIndexFileSys)
            if baseIndexFileSys not in ctxt.attFile.indexColumns:
                a.declarePointer ( ctxt.codegen.read )
                emit ( assign ( a, mmapFile ( a.dataType, baseIndexFileSys ) ), ctxt.codegen.read )
                ctxt.codegen.gpumem.mapForRead ( a )
                ctxt.attFile.indexColumns [ baseIndexFileSys ] = a.getGPU()
            else:
                pass
                #assert(False);
                #a.declarePointer (ctxt.codegen.read )
                #emit ( assign ( a, mmapFile ( a.dataType, baseIndexFileSys ) ), ctxt.codegen.read )
                #ctxt.codegen.gpumem.declare ( a )
                #emit ( assign ( a.getGPU(), ctxt.attFile.indexColumns [baseIndexFileSys] ), ctxt.codegen.gpumem.cudaMalloc )
            ctxt.codegen.currentKernel.addVar(a)
        
        if self.algExpr.unique:
            self.endVar = Variable.val ( CType.INT, "matchEnd" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            self.offsetVar = Variable.val ( CType.INT, "matchOffset" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            #self.matchStepVar = Variable.val ( CType.INT, "matchStep" + str ( self.algExpr.opId ), ctxt.codegen, intConst(1) )
            #matchFound = Variable.val ( CType.INT, "matchFound" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            probeActive = Variable.val ( CType.INT, "probeActive" + str ( self.algExpr.opId ), ctxt.codegen, ctxt.vars.activeVar )
            #ctxt.vars.buf.extend ( [ self.endVar, self.offsetVar, self.matchStepVar, matchFound, probeActive ] )
                
            if self.algExpr.joinType == Join.OUTER:
                doOuter = Variable.val ( CType.INT, "doOuter" + str ( self.algExpr.opId ), ctxt.codegen, intConst(1) )
                outerActive = Variable.val ( CType.INT, "outerActive" + str ( self.algExpr.opId ), ctxt.codegen, ctxt.vars.activeVar )
                for id, nullable in self.buildRelation.items():
                    ctxt.attFile.isNullFile [ nullable.id ] = notLogic ( matchFound )
                    
            # execute only when current thread has active elements    
            #hashVar = Variable.val ( CType.UINT64, "hash" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            with IfClause ( probeActive, ctxt.codegen ):
                var = ctxt.attFile.regFile [ cond_attr.id ]
                emit ( assign( probeActive, call( qlib.Fct.RELATIONSHIP_PROBE_MULTI,
                    ["{}_offset".format(rel_name), var, self.offsetVar, self.endVar] ) ), ctxt.codegen )
            
            #    Hash.attributes ( self.algExpr.probeKeyAttributes, hashVar, ctxt )
            #    emit ( assign ( probeActive, call( qlib.Fct.HASH_PROBE_MULTI, 
            #        [self.htmem.ht, self.htmem.numEntries, hashVar, self.offsetVar, self.endVar] ) ), ctxt.codegen )
            emit ( assign ( ctxt.vars.activeVar, probeActive ), ctxt.codegen )
            
                #payl = Variable.val ( self.htmem.payload.dataType, "payl", ctxt.codegen ) 
            profile_lane_activity(ctxt)
            tid = Variable.val ( CType.INT, "tid_{}{}".format(self.algExpr.table['name'], self.algExpr.scanTableId ), ctxt.codegen, intConst(0) )
            with IfClause ( ctxt.vars.activeVar, ctxt.codegen ):
                #emit ( assign ( payl, self.htmem.payload.arrayAccess ( self.offsetVar ) ), ctxt.codegen )
                #self.payload.dematerialize ( payl, ctxt )
                #Hash.checkEquality ( ctxt.vars.activeVar, self.algExpr.buildKeyAttributes, self.algExpr.probeKeyAttributes, ctxt )
                #emit ( assignAdd ( matchFound, ctxt.vars.activeVar ), ctxt.codegen )
                emit ( assign (tid, call( qlib.Fct.INDEX_GET_TID, ["{}_position".format(rel_name), self.offsetVar] ) ), ctxt.codegen )
                for id, a in self.algExpr.scanAttributes.items():
                    #if a.name == 'tid': continue
                    if a.lastUse <= self.algExpr.opId: continue
                    #print("demat", a.name, a.tableName)
                    if id == self.algExpr.tid.id:
                        regVar = ctxt.attFile.declareRegister ( a )
                        emit ( assign ( regVar, "tid_{}{}".format(self.algExpr.tid.tableName, self.algExpr.scanTableId) ), ctxt.codegen )
                    else:
                        ctxt.attFile.dematerializeAttribute (a, tid, self.algExpr.scanTableId )


            self.parent.consume ( ctxt )


        else:

            endVar = Variable.val ( CType.INT, "matchEnd" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            endVarBuf = Variable.val ( CType.INT, "matchEndBuf" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            offsetVar = Variable.val ( CType.INT, "matchOffset" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            offsetVarBuf = Variable.val ( CType.INT, "matchOffsetBuf" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            probeActive = Variable.val ( CType.INT, "probeActive" + str ( self.algExpr.opId ), ctxt.codegen, ctxt.vars.activeVar )
            
            profile_lane_activity(ctxt)

            bufferAtts = dict()
            bufferAtts.update ( self.algExpr.child.outRelation )
            bufferVars = []
            for id, att in bufferAtts.items():
                if att.lastUse > self.algExpr.opId:
                    print(id, att.name, att.tableName)
                    var = ctxt.attFile.regFile [ id ]
                    bufVar = copy.deepcopy ( var )
                    bufVar.name = bufVar.name + "_bcbuf" + str ( self.algExpr.opId )
                    bufVar.declare ( ctxt.codegen )
                    bufferVars.append ( (var, bufVar ) )

            with IfClause ( probeActive, ctxt.codegen ):
                var = ctxt.attFile.regFile [ cond_attr.id ]
                emit ( assign( probeActive, call( qlib.Fct.RELATIONSHIP_PROBE_MULTI,
                    ["{}_offset".format(rel_name), var, offsetVarBuf, endVarBuf] ) ), ctxt.codegen )

            activeProbes = Variable.val ( CType.UINT, "activeProbes" + str ( self.algExpr.opId ) )
            activeProbes.declareAssign ( ballotIntr ( qlib.Const.ALL_LANES, probeActive ), ctxt.codegen )
            numbuf = Variable.val ( CType.INT, "num" + str ( self.algExpr.opId ), ctxt.codegen, intConst(0) )
            emit ( assign ( numbuf, sub ( endVarBuf, offsetVarBuf ) ), ctxt.codegen )      
            wideProbes = Variable.val ( CType.UINT, "wideProbes"  + str ( self.algExpr.opId ))
            wideProbes.declareAssign ( ballotIntr ( qlib.Const.ALL_LANES, largerEqual ( numbuf, intConst(32) ) ), ctxt.codegen )


            
            

            # write register state to buffer to prevent overwriting 
            for var, bufVar in bufferVars:
                emit ( assign ( bufVar, var ), ctxt.codegen )

            
            with WhileLoop ( larger ( activeProbes, intConst(0) ), ctxt.codegen ):
                
                
                if True:
                    tupleLane = Variable.val ( CType.UINT, "tupleLane", ctxt.codegen )
                    emit ( assign ( tupleLane, sub ( ffsIntr ( activeProbes ), 1 ) ), ctxt.codegen ) 
                    # shuffle gather offset
                    emit ( assign ( offsetVar, add ( shuffleIntr ( qlib.Const.ALL_LANES, offsetVarBuf, tupleLane ), ctxt.codegen.warplane() ) ), ctxt.codegen )
                    # shuffle gather end
                    emit ( assign ( endVar, shuffleIntr ( qlib.Const.ALL_LANES, endVarBuf, tupleLane ) ), ctxt.codegen )      
                    # shuffle other register vars
                    for var, bufVar in bufferVars:
                        emit ( assign ( var, shuffleIntr ( qlib.Const.ALL_LANES, bufVar, tupleLane ) ), ctxt.codegen )      
                    # mark lane as processed
                    emit ( assignSub ( activeProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 

                    emit ( assign ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                
                else:
                
                    tupleLane = Variable.val ( CType.UINT, "tupleLane", ctxt.codegen )
                    broadcastLane = Variable.val ( CType.UINT, "broadcastLane", ctxt.codegen )
                    numFilled = Variable.val ( CType.INT, "numFilled", ctxt.codegen, intConst(0) )
                    num = Variable.val ( CType.INT, "num", ctxt.codegen, intConst(0) )

                    

                    with WhileLoop ( andLogic ( smaller ( numFilled, intConst(32) ), activeProbes ), ctxt.codegen ) as l: 
                    
                    
                        # select leader
                        with IfClause ( larger ( wideProbes, intConst (0) ), ctxt.codegen ):
                            emit ( assign ( tupleLane, sub ( ffsIntr ( wideProbes ), 1 ) ), ctxt.codegen ) 
                            emit ( assignSub ( wideProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 
                        with ElseClause ( ctxt.codegen ):
                            emit ( assign ( tupleLane, sub ( ffsIntr ( activeProbes ), 1 ) ), ctxt.codegen ) 
                        # broadcast leader number of matches
                        emit ( assign ( num, shuffleIntr ( qlib.Const.ALL_LANES, numbuf, tupleLane ) ), ctxt.codegen )
                        with IfClause ( andLogic ( numFilled, larger ( add ( numFilled, num ), 32 ) ), ctxt.codegen ):
                            l.break_()
                        with IfClause ( largerEqual ( ctxt.codegen.warplane(), numFilled ), ctxt.codegen ):
                            emit ( assign ( broadcastLane, tupleLane ), ctxt.codegen )
                            emit ( assign ( offsetVar, sub ( ctxt.codegen.warplane(), numFilled ) ), ctxt.codegen )      
                        emit ( assignAdd ( numFilled, num ), ctxt.codegen )  
                        # mark buffered probe tuple as processed            
                        emit ( assignSub ( activeProbes, ( shiftLeft ( intConst(1), tupleLane ) ) ), ctxt.codegen ) 

                    # shuffle gather offset
                    emit ( assignAdd ( offsetVar, shuffleIntr ( qlib.Const.ALL_LANES, offsetVarBuf, broadcastLane ) ), ctxt.codegen )
                    # shuffle gather end
                    emit ( assign ( endVar, shuffleIntr ( qlib.Const.ALL_LANES, endVarBuf, broadcastLane ) ), ctxt.codegen )      
                    # shuffle other register vars
                    for var, bufVar in bufferVars:
                        emit ( assign ( var, shuffleIntr ( qlib.Const.ALL_LANES, bufVar, broadcastLane ) ), ctxt.codegen )      

                    emit ( assign ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                    
                
                
                ctxt.innerLoopCount += 1
                with WhileLoop ( anyIntr ( qlib.Const.ALL_LANES, probeActive ), ctxt.codegen ):
                    
                    flushPipeline = Variable.val ( CType.UINT, "flushPipeline" + str ( self.algExpr.opId ) )
                    flushPipeline.declareAssign ( 
                        andLogic ( 
                            equals(activeProbes, intConst(0)), 
                            equals(ballotIntr ( qlib.Const.ALL_LANES, smaller(add(offsetVar,intConst(32)), endVar )), intConst(0))), 
                            ctxt.codegen )
                    ctxt.vars.divergenceFlushVar = flushPipeline

                    
                    emit ( assign ( ctxt.vars.activeVar, probeActive ), ctxt.codegen )

                    #ctxt.codegen.laneActivityProfile ( ctxt )
                    emit ( assign ( ctxt.vars.activeVar, intConst(0) ), ctxt.codegen )
                    #payl = Variable.val ( self.htmem.payload.dataType, "payl", ctxt.codegen )
                    tid = Variable.val ( CType.INT, "tid_{}{}".format(self.algExpr.table['name'], self.algExpr.scanTableId ), ctxt.codegen, intConst(0) )
                    with IfClause ( probeActive, ctxt.codegen ):
                        #emit ( assign ( payl, self.htmem.payload.arrayAccess ( offsetVar ) ), ctxt.codegen )
                        #self.payload.dematerialize ( payl, ctxt )
                        #for
                        emit ( assign (tid, call( qlib.Fct.INDEX_GET_TID, ["{}_position".format(rel_name), offsetVar] ) ), ctxt.codegen )
                        for id, a in self.algExpr.scanAttributes.items():
                            #if a.name == 'tid': continue
                            if a.lastUse <= self.algExpr.opId: continue
                            #print("demat", a.name, a.tableName)
                            if id == self.algExpr.tid.id:
                                regVar = ctxt.attFile.declareRegister ( a )
                                emit ( assign ( regVar, "tid_{}{}".format(self.algExpr.tid.tableName, self.algExpr.scanTableId) ), ctxt.codegen )
                            else:
                                ctxt.attFile.dematerializeAttribute (a, tid, self.algExpr.scanTableId )

                        emit ( assign ( ctxt.vars.activeVar, intConst(1) ), ctxt.codegen )
                        emit ( assignAdd ( offsetVar, intConst(32) ), ctxt.codegen )        
                        emit ( assignAnd ( probeActive, smaller ( offsetVar, endVar ) ), ctxt.codegen )
                    self.parent.consume ( ctxt )
                ctxt.innerLoopCount -= 1
                
            for var, bufVar in bufferVars:
                emit ( assign ( var, bufVar ), ctxt.codegen )

