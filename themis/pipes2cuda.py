import sys
import os
import subprocess
import pprint
import copy
import time
from enum import Enum
from collections import OrderedDict

sys.path.insert(0,"../../code")
sys.path.insert(0,'.')

import importlib
import dogqc.dbio as io
from schema import schema
from dogqc.util import loadScript

# algebra types
from dogqc.relationalAlgebra import Context
from dogqc.relationalAlgebra import RelationalAlgebra
from dogqc.relationalAlgebra import Join
from dogqc.relationalAlgebra import Reduction
from dogqc.cudaTranslator import CudaCompiler
from dogqc.types import Type
from dogqc.cudalang import CType
import dogqc.scalarAlgebra as scal
from dogqc.kernel import KernelCall
from dogqc.hashJoins import EquiJoinTranslator 

from dogqc.kernel import KernelCall

pp = pprint.PrettyPrinter(indent=2)


stat_counters = ['processing_clocks', 'pushing_clocks', 'waiting_clocks', 'num_idle', 'num_pushed']


def langType ( relDataType ):
    internalTypeMap = {}
    internalTypeMap [ Type.INT ] = CType.INT
    internalTypeMap [ Type.ULL ] = CType.ULL
    internalTypeMap [ Type.DATE ] = CType.UINT
    internalTypeMap [ Type.CHAR ] = CType.CHAR
    internalTypeMap [ Type.FLOAT ] = CType.FP32
    internalTypeMap [ Type.DOUBLE ] = CType.FP32
    internalTypeMap [ Type.STRING ] = CType.STR_TYPE
    internalTypeMap [ Type.PTR_INT ] = CType.PTR_INT
    return internalTypeMap [ relDataType ]


class DataStructure():

    def cuMalloc(self, name, dtype, size, gctxt):
        code = gctxt.maincode
        code.add('{}* {};'.format(dtype, name))
        #size = int((size-1)/128) * 128 + 128 
        code.add('cudaMalloc((void**) &{}, {} * sizeof({}));'
            .format(name, size, dtype))

    def mmap(self, name, dtype, size, gctxt):
        code = gctxt.maincode
        self.cuMalloc(name, dtype, size, gctxt)
        code.add('{}* mmap_{} = ({}*) map_memory_file("{}/{}");'
            .format(dtype, name, dtype, gctxt.dbpath, name))
        code.add('cudaMemcpy({}, mmap_{}, {} * sizeof({}), cudaMemcpyHostToDevice);'
            .format(name, name, size, dtype))

    def initArray(self, name, initValue, size, gctxt):
        code = gctxt.maincode
        code.add('initArray<<<{}, {}>>>({},{},{});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize, name, initValue, size))

    def declarePayload(self, opId, attributes, condAttrs, gctxt):
        c = gctxt.precode
        c.add('struct Payload{}'.format(opId))
        c.add('{')
        for attrId, attr in attributes.items():
            #if attr.name == 'tid':
            #    c.add('\tint tid{};'.format(attr.creation))
            #else:
            #    c.add('\t{} attr{}_{};'.format(langType(attr.dataType), attrId, attr.name))
            c.add('\t{} {};'.format(langType(attr.dataType), attr.id_name))
        for attrId, attr in condAttrs.items():
            if attrId in attributes: continue
            c.add('\t{} {};'.format(langType(attr.dataType), attr.id_name))
        c.add('};')        

class TrapezoidStacks(DataStructure):

    def __init__(self, ts_width, max_ts_speculated_num_ranges, max_ts_speculated_size_attributes):
        self.ts_width = ts_width
        self.max_ts_speculated_num_ranges = max_ts_speculated_num_ranges
        self.max_ts_speculated_size_attributes = max_ts_speculated_size_attributes
        self.size_per_warp = 8 * 2 * self.ts_width * self.max_ts_speculated_num_ranges + self.max_ts_speculated_size_attributes

    def declare(self, gctxt):
        #self.cuMalloc('ts', 'char', self.size_per_warp * gctxt.num_warps, gctxt)        
        c = gctxt.maincode
        

class PushedParts(DataStructure):

    def __init__(self, ts_width, max_height, max_ts_size_attributes, max_accumulated_ts_size_attributes):
        self.ts_width = ts_width
        self.max_height = max_height
        self.max_ts_size_attributes = max_ts_size_attributes
        self.size_per_warp = max_ts_size_attributes
        self.max_accumulated_ts_size_attributes = max_accumulated_ts_size_attributes
        

    def declare(self, gctxt):
        if KernelCall.args.lb:
            c = gctxt.maincode
            self.cuMalloc('global_info', 'unsigned int', 2 * 32, gctxt)
            c.add('unsigned int* global_num_idle_warps = global_info;')
            c.add('int* global_scan_offset = (int*)(global_info + 32);')
            
            if KernelCall.args.lb_type == 'ws': # Work sharing
                c.add('WorkSharing::TaskBook* taskbook;')
                c.add('WorkSharing::TaskStack* taskstack;')
                c.add('cudaMalloc((void**) &taskbook, 1 * 1024 * 1024 * 1024);')
                c.add('cudaMalloc((void**) &taskstack, 256 * 1024 * 1024);')
            else: # Work pushing
                if KernelCall.args.lb_mode == 'morsel':
                    pass
                else:
                    c.add(f'Themis::PushedParts::PushedPartsStack* gts;')
                    c.add(f'size_t size_of_stack_per_warp;')
                    c.add(f'Themis::PushedParts::InitPushedPartsStack(gts, size_of_stack_per_warp, 1 << 31, {gctxt.num_warps});')
                    if KernelCall.args.lb_detection == 'twolvlbitmaps':
                        bitmapsize = (int((gctxt.num_warps-1) / 64) + 1)
                        c.add(f'//bitmapsize: {bitmapsize} for {gctxt.num_warps}')
                        self.cuMalloc('global_bit1', 'unsigned long long',  16 + bitmapsize * 16, gctxt)
                        c.add('unsigned long long* global_bit2 = global_bit1 + 16;')
                        c.add('cudaMemset(global_bit1, 0, sizeof(unsigned long long) * 2);')
                        c.add(f'cudaMemset(global_bit2, 0, sizeof(unsigned long long) * {bitmapsize * 16});')
                    elif KernelCall.args.lb_detection == 'randomized':
                        self.cuMalloc('global_bit2', 'unsigned long long', (int((gctxt.num_warps-1) / 64) + 1) * 8, gctxt)
                        c.add(f'cudaMemset(global_bit2, 0, {(int((gctxt.num_warps-1) / 64) + 1) * 8});')
                    elif KernelCall.args.lb_detection == 'randomized':
                        pass
                    elif KernelCall.args.lb_detection == 'simple':
                        self.cuMalloc('global_id_stack_buf', 'int', gctxt.num_warps+3, gctxt)
                        c.add(f'cudaMemset(global_id_stack_buf, 0, sizeof(int) * {gctxt.num_warps+3});')
                        c.add('Themis::Detection::Stack::IdStack* global_id_stack = (Themis::Detection::Stack::IdStack*) global_id_stack_buf;')



class SingleVar(DataStructure):

    def __init__(self, typeName, name):
        self.typeName = typeName
        self.name = name

    def declare(self, gctxt):
        pass

    def toArg(self):
        return f' {self.typeName} {self.name},'

    def toCall(self):
        return f' {self.name},'

class Table(DataStructure):

    def __init__(self, name, size, attrs):
        self.name = name
        self.attrs = attrs
        self.size = size

    def declare(self, gctxt):

        self.cuMalloc('nout_{}'.format(self.name), CType.INT, 1, gctxt)
        self.initArray('nout_{}'.format(self.name), CType.zeroValue[CType.INT], 1, gctxt)
        if KernelCall.args.system == 'Pyper':
            self.cuMalloc('select_{}'.format(self.name), CType.INT, self.size, gctxt )
            self.initArray('select_{}'.format(self.name), CType.zeroValue[CType.INT], self.size, gctxt)
        for attrId, attr in self.attrs.items():
            gctxt.maincode.add('//{}'.format(attr.id_name))
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING:
                dtype = CType.STR_TYPE
            
            attrName = attr.name if self.name != 'result' and self.name[:4] != 'temp' else attr.id_name
            self.cuMalloc('{}_{}'.format(self.name, attrName), dtype, self.size, gctxt)
            
    def toArg(self):

        if KernelCall.args.system == 'Pyper':
            result = 'int* nout_{}, int* select_{}, '.format(self.name, self.name)
        else:
            result = f'int* nout_{self.name}, '
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING:
                dtype = CType.STR_TYPE

            attrName = attr.name if self.name != 'result' and self.name[:4] != 'temp' else attr.id_name
            result += ' {}* {}_{},'.format(dtype, self.name, attrName)
        return result

    def toCall(self):
        if KernelCall.args.system == 'Pyper':
            result = 'nout_{},select_{},'.format(self.name,self.name)
        else:
            result = f'nout_{self.name},'
        for attrId, attr in self.attrs.items():
            dtype = langType(attr.dataType)
            if attr.dataType == Type.STRING:
                dtype = CType.STR_TYPE
            attrName = attr.name if self.name != 'result' and self.name[:4] != 'temp' else attr.id_name
            result += ' {}_{},'.format(self.name, attrName)
        return result

class Column(DataStructure):

    def __init__(self, attr):
        self.attr = attr
        self.name = '{}_{}'.format(attr.tableName, attr.name)

    def declare(self, gctxt):
        attr = self.attr
        name = self.name
        schema = gctxt.schema[attr.tableName]
        if attr.dataType == Type.STRING:
            self.mmap(name + '_offset', CType.SIZE, schema['size']+1, gctxt)
            self.mmap(name + '_char', CType.CHAR, schema['charSizes'][attr.name], gctxt)
        else:
            self.mmap(name, langType(attr.dataType), schema['size'], gctxt)

    def toArg(self):
        if self.attr.dataType == Type.STRING:
            return 'size_t* {}_offset, char* {}_char,'.format(self.name, self.name)
        else:
            return '{}* {},'.format(langType(self.attr.dataType), self.name)

    def toCall(self):
        if self.attr.dataType == Type.STRING:
            return '{}_offset, {}_char,'.format(self.name, self.name)
        else:
            return '{},'.format(self.name)

class Trie(DataStructure):

    def __init__(self, attr):
        self.attr = attr
        self.name = attr.trieName

    def declare(self, gctxt):
        self.mmap('{}_offset'.format(self.name), CType.INT, 2, gctxt)
        
    def toArg(self):
        return 'int* {}_offset,'.format(self.name, self.name)

    def toCall(self):
        return '{}_offset,'.format(self.name, self.name)

class TrieColumn(DataStructure):

    def __init__(self, attr):
        self.name = '{}_{}'.format(attr.tid.trieName, attr.name)
        self.attr = attr

    def declare(self, gctxt):
        size = self.attr.tid.trieSize[self.attr.name]
        if self.attr.isLastKey:
            self.mmap('{}_dir_offset'.format(self.name), CType.INT, 2*size, gctxt)
        else:
            self.mmap('{}_offset'.format(self.name), CType.INT, 2*size, gctxt)
        self.mmap('{}_val'.format(self.name), CType.INT, size, gctxt)

    def toArg(self):
        if self.attr.isLastKey:
            return 'int* {}_dir_offset, int* {}_val,'.format(self.name, self.name)
        else:
            return 'int* {}_offset, int* {}_val,'.format(self.name, self.name)

    def toCall(self):
        if self.attr.isLastKey:
            return '{}_dir_offset, {}_val,'.format(self.name, self.name)
        else:
            return '{}_offset, {}_val,'.format(self.name, self.name)

class Index(DataStructure):

    def __init__(self, rel_name, from_tname, to_tname):
        self.from_tname = from_tname
        self.to_tname = to_tname
        self.name = rel_name
        
    def declare(self, gctxt):
        from_tname, to_tname = self.from_tname, self.to_tname
        rel_name = self.name
        from_tsize = gctxt.schema[from_tname]['size']
        to_tsize = gctxt.schema[to_tname]['size']
        
        self.mmap(rel_name + "_offset", CType.INT, 2*from_tsize, gctxt)
        self.mmap(rel_name + "_position", CType.INT, to_tsize, gctxt)

    def toArg(self):
        return 'int* {}_offset, int* {}_position,'.format(self.name, self.name)
    
    def toCall(self):
        return '{}_offset, {}_position,'.format(self.name, self.name)
        

class AggHT(DataStructure):

    def __init__(self, opId, size, keyAttrs, condAttrs, aggAttrs={}):
        self.opId = opId
        self.size = size
        self.keyAttrs = keyAttrs
        self.aggAttrs = aggAttrs
        self.name = 'aht{}'.format(opId)
        self.condAttrs = condAttrs
        
    def declare(self, gctxt):
        opId, size, keyAttrs, aggAttrs = self.opId, self.size, self.keyAttrs, self.aggAttrs

        self.declarePayload(opId, keyAttrs, self.condAttrs, gctxt)

        if KernelCall.args.system == 'Themis' and self.size > 1 and self.size < KernelCall.args.agg_threshold:
            size = size * (int(KernelCall.defaultBlockSize * KernelCall.defaultGridSize / 32) + 1)
        
        self.cuMalloc('aht{}'.format(opId), 'agg_ht<Payload{}>'.format(opId), size, gctxt)
        c = gctxt.maincode
        c.add('initAggHT<<<{},{}>>>(aht{},{});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize, opId, size))

        for attrId, ( attr, inputIdentifier, reductionType ) in aggAttrs.items():
            name = 'aht{}_{}'.format(opId, attr.id_name)
            dtype = langType(attr.dataType)
            self.cuMalloc(name, dtype, size, gctxt)
            if reductionType == Reduction.SUM or reductionType == Reduction.AVG or reductionType == Reduction.COUNT:
                initValue = CType.zeroValue[dtype]
            elif reductionType == Reduction.MAX:
                initValue = CType.minValue[dtype]
            elif reductionType == Reduction.MIN:
                initValue = CType.maxValue[dtype]
            else:
                assert(False)
            self.initArray(name, initValue, size, gctxt)
    
    def toArg(self):
        result = 'agg_ht<Payload{}>* aht{},'.format(self.opId, self.opId)
        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            name = 'aht{}_{}'.format(self.opId, attr.id_name)
            dtype = langType(attr.dataType) 
            result += ' {}* {},'.format(dtype, name)
        return result

    def toCall(self):
        result = 'aht{},'.format(self.opId)
        for attrId, ( attr, inputIdentifier, reductionType ) in self.aggAttrs.items():
            result += ' aht{}_{},'.format(self.opId, attr.id_name)
        return result

class MultiHT(DataStructure):

    def __init__(self, opId, size, psize, attrs):
        self.opId = opId
        self.size = size
        self.psize = psize
        self.name = 'jht{}'.format(opId)
        self.attrs = attrs

    def declare(self, gctxt):
        opId, size, psize, attrs = self.opId, self.size, self.psize, self.attrs
        
        self.declarePayload(opId, attrs, {}, gctxt)
        
        name = self.name
        self.cuMalloc(name, 'multi_ht', size, gctxt)
        c = gctxt.maincode
        c.add('initMultiHT<<<{},{}>>>({},{});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize, name, size))

        self.cuMalloc('jht{}_offset'.format(opId), CType.INT, 1, gctxt)
        self.initArray('jht{}_offset'.format(opId), CType.zeroValue[CType.INT], 1, gctxt)

        self.cuMalloc('jht{}_payload'.format(opId), 'Payload{}'.format(opId), psize, gctxt)

    def toArg(self):
        return 'multi_ht* jht{}, int* jht{}_offset, Payload{}* jht{}_payload,'.format(self.opId, self.opId, self.opId, self.opId)
        
    def toCall(self):
        return 'jht{}, jht{}_offset, jht{}_payload,'.format(self.opId, self.opId, self.opId)

class UniqueHT(DataStructure):

    def __init__(self, opId, size, attrs):
        self.opId = opId
        self.size = size
        self.name = 'ujht{}'.format(opId)
        self.attrs = attrs

    def declare(self, gctxt):
        opId, size, attrs = self.opId, self.size, self.attrs
        
        print('declare Payload', self.name, attrs)
        self.declarePayload(opId, attrs, {}, gctxt)
        
        name = self.name
        self.cuMalloc(name, 'unique_ht<Payload{}>'.format(opId), size, gctxt)
        c = gctxt.maincode
        c.add('initUniqueHT<<<{},{}>>>({},{});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize, name, size))

    def toArg(self):
        return 'unique_ht<Payload{}>* ujht{},'.format(self.opId, self.opId)

    def toCall(self):
        return 'ujht{},'.format(self.opId)



class Location:

    def __init__(self, s, tids={}):
        self.s = s
        self.tids = tids
        pass

    def __str__(self):
        return self.s #'{} {}'.format(self.s, self.tids)


    @staticmethod
    def reg(attr):
        #return Location("attr{}_{}".format(attr.id, attr.name))
        #if attr.name == "tid":
        #    return Location("tid{}".format(attr.creation))
        #else:
        return Location(attr.id_name)
        #return Location("attr{}_{}".format(attr.id, attr.name))

    @staticmethod
    def ts(attr, pos, spid):
        return Location("(({}*)(&ts_attributes[{}]))[sid{}]".format(langType(attr.dataType), pos, spid))

    @staticmethod
    def table(attr, attrTid):
        name = "{}_{}".format(attr.tableName, attr.name)
        if langType(attr.dataType) == CType.STR_TYPE:
            return Location("stringScan({}_offset,{}_char,{})".format(name, name, attrTid.id_name), {attrTid.id:attrTid})
            #return Location("stringScan({}_offset,{}_char,tid{})".format(name, name, attr.creation), {attrTid.id:attrTid})
        else:
            return Location("{}[{}]".format(name, attrTid.id_name), {attrTid.id:attrTid})
            #return Location("{}[tid{}]".format(name, attr.creation), {attrTid.id:attrTid})

    @staticmethod
    def temptable(tableName, attr, attrTid):
        #name = "temp{}_{}".format(opId, attr.id_name)
        if tableName[:4] == 'temp':
            return Location("{}_{}[{}]".format(tableName, attr.id_name, attrTid.id_name), {attrTid.id:attrTid})
        else:
            return Location("{}_{}[{}]".format(tableName, attr.name, attrTid.id_name), {attrTid.id:attrTid})
        #return Location("{}[tid{}]".format(name, opId), {attrTid.id:attrTid})

    @staticmethod
    def aht(opId, attr, attrTid):
        #if attr.name == 'tid':
        #    return Location("aht{}[tid{}].payload.tid{}".format(opId, opId, attr.creation), {attrTid.id:attrTid})
        #else:
        #return Location("aht{}[tid{}].payload.attr{}_{}".format(opId, opId, attr.id, attr.name), {attrTid.id:attrTid})
        return Location("aht{}[{}].payload.{}".format(opId, attrTid.id_name, attr.id_name), {attrTid.id:attrTid})
        

    @staticmethod
    def ujht(opId, attr, attrTid):
        #if attr.name == 'tid':
        #    return Location("ujht{}[tid{}].payload.tid{}".format(opId, opId, attr.creation), {attrTid.id:attrTid})
        #else:
        #return Location("ujht{}[tid{}].payload.attr{}_{}".format(opId, opId, attr.id, attr.name), {attrTid.id:attrTid})
        return Location("ujht{}[{}].payload.{}".format(opId, attrTid.id_name, attr.id_name), {attrTid.id:attrTid})

    @staticmethod
    def apayl(opId, attr, attrTid, idx=0):
        if attrTid is None:
            return Location("aht{}_{}[0]".format(opId, attr.id_name), {})
            #return Location("aht{}_attr{}_{}[0]".format(opId, attr.id, attr.name), {})
        return Location("aht{}_{}[{}]".format(opId, attr.id_name, attrTid.id_name), {attrTid.id:attrTid})
        #return Location("aht{}_attr{}_{}[tid{}]".format(opId, attr.id, attr.name, opId), {attrTid.id:attrTid})

    @staticmethod
    def jpayl(opId, attr, attrTid):
        #if attr.name == 'tid':
        #    return Location("jht{}_payload[{}].{}".format(opId, attrTid.id_name, attr.id_name), {attrTid.id:attrTid})
        #    return Location("jht{}_payload[tid{}].tid{}".format(opId, opId, attr.creation), {attrTid.id:attrTid})
        #else:
        #    return Location("jht{}_payload[tid{}].attr{}_{}".format(opId, opId, attr.id, attr.name), {attrTid.id:attrTid})
        return Location("jht{}_payload[{}].{}".format(opId, attrTid.id_name, attr.id_name), {attrTid.id:attrTid})


    @staticmethod
    def payl(attr):
        #if attr.name == "tid":
        #    return Location("payl.tid{}".format(attr.creation))
        #else:
        #    return Location("payl.attr{}_{}".format(attr.id, attr.name))
        return Location("payl.{}".format(attr.id_name))

    @staticmethod
    def divide(loc1, loc2):
        loc = Location("((float){})/((float){})".format(loc1, loc2), {})
        loc.tids.update(loc1.tids)
        loc.tids.update(loc2.tids)
        return loc

    @staticmethod
    def divide(loc1, loc2):
        loc = Location("((float){})/((float){})".format(loc1, loc2), {})
        loc.tids.update(loc1.tids)
        loc.tids.update(loc2.tids)
        return loc


class Code:

    def __init__(self):
        self.lines = []

    def add(self, line):
        self.lines.append(line)

    def addAll(self, lines):
        self.lines = self.lines + lines

    def addTabs(self, n):
        self.lines = list(map(lambda x: ('\t'* n) + x, self.lines))

    def toString(self):
        return '\n'.join(self.lines)

    def isEmpty(self):
        return len(self.lines) == 0


class PhysicalOperator:

    def __init__(self, op):
        self.algExpr = op
        self.touched = 0

    def __str__(self):
        e = self.algExpr
        return '{} {} {}'.format(e.opId, e.opType, e.outRelation.keys())

    def genScanLoop(self, spctxt):
        pass

    def genScan(self, spctxt):
        pass

    def genSingleScan(self, spctxt):
        pass

    def genMaterialize(self, spctxt):
        pass

    def genOperation(self, spctxt):
        pass

    def genPopKnownNodes(self, spctxt): # 6000000 883510 # 8646918

        algExpr = self.algExpr

        pc = spctxt.precode
        uc = spctxt.updatecode
        prev = spctxt.prev

        #pc.add("int sid{} = -1;".format(spctxt.id))
        if prev.ts_type == 0:
            spctxt.declare(algExpr.tid)
            tidLoc = spctxt.attrLoc[algExpr.tid.id]
            if KernelCall.args.lb and KernelCall.args.lb_mode == 'morsel':
                pc.add(f'Themis::FillIPartAtZeroLvlForMorsel(lvl, thread_id, inodes_cnts, sid{spctxt.id}, {tidLoc}, ts_0_range_cached);')
            else:
                if KernelCall.args.lb:
                    step = 'blockDim.x * gridDim.x' # step = '32'
                else:
                    step = 'blockDim.x * gridDim.x'
                pc.add(f"Themis::FillIPartAtZeroLvl(lvl, thread_id, active, {tidLoc}, ts_0_range_cached, mask_32, mask_1, {step});")
                
                if KernelCall.args.lb and KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
                    pc.add('Themis::WorkloadTracking::UpdateWorkloadSizeAtZeroLvl(thread_id, loop, local_info, global_stats_per_lvl);')
            #pc.add('if (active) atomicAdd(&cnt[0], 1);')
                
                
            spctxt.inReg.add(algExpr.tid.id)
            
            #for attrId, attr in algExpr.outRelation.items():
            #    if attr.id in spctxt.attrLoc:
            #        spctxt.toReg(attr)
        elif prev.ts_type == 1:
            # 1. Update Cache
            uc.add(f'int ts_src = 32;')
            uc.add(f'bool is_updated = Themis::DistributeFromPartToDPart(thread_id, {spctxt.id}, ts_src, ts_{spctxt.id}_range, ts_{spctxt.id}_range_cached, mask_32, mask_1);')
            if KernelCall.args.lb and KernelCall.args.lb_type == "wp" and KernelCall.args.lb_mode == 'glvl':
                uc.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtLoopLvl(thread_id, {spctxt.id}, loop, ts_{spctxt.id}_range, ts_{spctxt.id}_range_cached, mask_1, local_info, global_stats_per_lvl);')
            
            # 2. Update attributes
            if len(spctxt.prev.ts_list_attributes) > 0:    
                uc.add('if (is_updated) {')
                uc.add('//fuck')
                for attr in spctxt.prev.ts_list_attributes:
                    spctxt.attrLoc[attr.id] = Location('ts_{}_attr_{}_cached'.format(spctxt.id, attr.id_name), {})
                    uc.add('{')
                    if attr.dataType == Type.STRING:
                        uc.add(f'char* ts_{spctxt.id}_attr_{attr.id_name}_cached0_start = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}.start, ts_src);')
                        uc.add(f'char* ts_{spctxt.id}_attr_{attr.id_name}_cached0_end = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}.end, ts_src);')
                        uc.add('if (ts_src < 32) {')
                        uc.add(f'ts_{spctxt.id}_attr_{attr.id_name}_cached.start = ts_{spctxt.id}_attr_{attr.id_name}_cached0_start;')
                        uc.add(f'ts_{spctxt.id}_attr_{attr.id_name}_cached.end = ts_{spctxt.id}_attr_{attr.id_name}_cached0_end;')
                        uc.add('}')
                    elif attr.dataType == Type.PTR_INT:
                        uc.add(f'{langType(attr.dataType)} ts_{spctxt.id}_attr_{attr.id_name}_cached0 = (int*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}, ts_src);')
                        uc.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached = ts_{spctxt.id}_attr_{attr.id_name}_cached0;')
                    else:
                        uc.add(f'{langType(attr.dataType)} ts_{spctxt.id}_attr_{attr.id_name}_cached0 = __shfl_sync(ALL_LANES, ts_{spctxt.id}_attr_{attr.id_name}, ts_src);')
                        uc.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached = ts_{spctxt.id}_attr_{attr.id_name}_cached0;')
                    uc.add('}')
                

                for attr in spctxt.ts_list_attributes:
                    loc_src = spctxt.attrLoc[attr.id]
                    pc.add(f'// pc {attr.id_name}')
                    pc.add('{')
                    if attr.dataType == Type.STRING:
                        pc.add('char* ts_{}_attr_{}_cached0_start = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{}_attr_{}.start, ts_src);'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name, loc_src))
                        pc.add('char* ts_{}_attr_{}_cached0_end = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{}_attr_{}.end, ts_src);'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name, loc_src))
                        pc.add('ts_{}_attr_{}_cached.start =ts_{}_attr_{}_cached0_start;'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name))
                        pc.add('ts_{}_attr_{}_cached.end = ts_{}_attr_{}_cached0_end;'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name))
                    elif attr.dataType ==  Type.INT:
                        pc.add('int ts_{}_attr_{}_cached0 = __shfl_sync(ALL_LANES, ts_{}_attr_{}, ts_src);'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name, loc_src))
                        pc.add('ts_{}_attr_{}_cached = ts_{}_attr_{}_cached0;'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name))
                    elif attr.dataType == Type.DOUBLE:
                        pc.add('float ts_{}_attr_{}_cached0 = __shfl_sync(ALL_LANES, ts_{}_attr_{}, ts_src);'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name, loc_src))
                        pc.add('ts_{}_attr_{}_cached = ts_{}_attr_{}_cached0;'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name))
                    else:
                        pc.add('{} ts_{}_attr_{}_cached0 = __shfl_sync(ALL_LANES, ts_{}_attr_{}, ts_src);'
                            .format(langType(attr.dataType), spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name, loc_src))
                        pc.add('ts_{}_attr_{}_cached = ts_{}_attr_{}_cached0;'
                            .format(spctxt.id+1, attr.id_name, spctxt.id+1, attr.id_name))
                    pc.add('}')
                
                uc.add('}')
            

            #pc.add(f'lvl = {spctxt.id};')
            pc.add("int loopvar{};".format(spctxt.id))
            spctxt.declare(prev.ts_tid)
            tidLoc = spctxt.attrLoc[prev.ts_tid.id]
            pc.add(f'Themis::FillIPartAtLoopLvl({spctxt.id}, thread_id, active, loopvar{spctxt.id}, ts_{spctxt.id}_range_cached, mask_32, mask_1);')
            
            spctxt.inReg.add(prev.ts_tid.id)
            spctxt.activecode.add("{} = {};".format(tidLoc, prev.ts_tid_build))
            for attr in spctxt.prev.ts_list_attributes:
                spctxt.attrLoc[attr.id] = Location('ts_{}_attr_{}_cached'.format(spctxt.id, attr.id_name), {})
            
            for attrId, attr in algExpr.inRelation.items():
                if attr.id in spctxt.attrLoc:
                    spctxt.toReg(attr)
            for attrId, attr in algExpr.outRelation.items():
                pc.add(f"//try to load {attrId} {attr.id_name}")
                if attrId not in algExpr.inRelation and attr.id in spctxt.attrLoc:
                    spctxt.toReg(attr)
                
            spctxt.activecode.add("}")
            spctxt.activecode.addAll(uc.lines)
            spctxt.activecode.add("if (active) { // shit...")
        
        elif prev.ts_type == 2:
            
            if len(spctxt.prev.ts_list_attributes) > 0:
                pc.add(f'if (!(mask_32 & (0x1u << {spctxt.id})))' + '{')
                for attr in spctxt.prev.ts_list_attributes:
                    pc.add(f'ts_{spctxt.id}_attr_{attr.id_name} = ts_{spctxt.id}_attr_{attr.id_name}_flushed;')
                pc.add('}')
            pc.add(f"Themis::FillIPartAtIfLvl({spctxt.id}, thread_id, inodes_cnts, active, mask_32, mask_1);")
            if KernelCall.args.lb and KernelCall.args.lb_type == "wp" and KernelCall.args.lb_mode == 'glvl':
                pc.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtIfLvl(thread_id, {spctxt.id}, loop, inodes_cnts, mask_1, local_info, global_stats_per_lvl);')
            
            for attr in spctxt.prev.ts_list_attributes:
                spctxt.attrLoc[attr.id] = Location('ts_{}_attr_{}'.format(spctxt.id, attr.id_name), {})
            
        
        #pc.add("active = sid{} != -1;".format(spctxt.id))

    def genAttrToMaterialize(self, spctxt):
        algExpr = self.algExpr
        # 1. Find attributes to push
        attrToMaterialize = {}
        for attrId, attr in algExpr.outRelation.items():
            if KernelCall.args.use_pos_vec:
                #assert(False)
                if attrId in spctxt.pctxt.materialized:
                    attrToMaterialize[attrId] = attr
                else:
                    for attrTid in spctxt.attrOriginLoc[attrId].tids.values():
                        attrToMaterialize[attrTid.id] = attrTid
            else:
                attrToMaterialize[attrId] = attr

        return attrToMaterialize


    def genPushTrapzoids(self, spctxt):
        algExpr = self.algExpr

        # 1. Find attributes to push
        attrToMaterialize = self.genAttrToMaterialize(spctxt)

        if spctxt.ts_type == 1:
            spctxt.pctxt.precode.add('Themis::Range ts_{}_range_cached;'.format(spctxt.id+1))
            spctxt.pctxt.precode.add('Themis::Range ts_{}_range;'.format(spctxt.id+1))
            #spctxt.pctxt.precode.add('int  ts_{}_sid_cached = {};'.format(spctxt.id+1, 2 * KernelCall.args.ts_width))

        # 2. Calculate trapezoid info 
        spctxt.ts_size_attributes = 0        
        for attrId, attr in attrToMaterialize.items():
            if spctxt.ts_tid != None and spctxt.ts_tid.id == attrId:
                continue
            spctxt.toReg(attr)

            if spctxt.ts_type == 1:
                spctxt.pctxt.precode.add('{} ts_{}_attr_{}_cached;'.format(langType(attr.dataType), spctxt.id+1, attr.id_name))
                spctxt.pctxt.precode.add('{} ts_{}_attr_{};'.format(langType(attr.dataType), spctxt.id+1, attr.id_name))
            else:
                spctxt.pctxt.precode.add('{} ts_{}_attr_{}_flushed;'.format(langType(attr.dataType), spctxt.id+1, attr.id_name))
                spctxt.pctxt.precode.add('{} ts_{}_attr_{};'.format(langType(attr.dataType), spctxt.id+1, attr.id_name))

            spctxt.ts_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
            spctxt.ts_list_attributes.append(attr)
        spctxt.ts_speculated_size_attributes += spctxt.ts_size_attributes

        if spctxt.ts_type == 1:
            spctxt.ts_tid = algExpr.tid
            spctxt.ts_speculated_num_ranges += 1
            spctxt.ts_num_ranges = 1
        elif spctxt.ts_type == 2:
            spctxt.ts_num_ranges = 0
        else:
            assert(False)


        # 3. Gen codes that push trapezoids to a stack
        pc = spctxt.postcode
        pc.add('unsigned push_active_mask = __ballot_sync(ALL_LANES, active);')        
        
        if spctxt.ts_type == 1 and KernelCall.args.lb_type == 'ws': # work sharing
            pc.add('if (push_active_mask) {')
            ts_size_attributes = int(spctxt.ts_size_attributes / (2 * spctxt.gctxt.ts_width))
            pc.add(f'WorkSharing::Task* task = NULL;')
            
            if KernelCall.args.mode == 'stats':
                pc.add('{')
                pc.add('unsigned long long current_tp = clock64();')
                pc.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
                pc.add('tp = current_tp;')
                pc.add('current_status = TYPE_STATS_PUSHING;')
                pc.add('}')
                
            
            pc.add(f'if (active) task = taskbook->AllocTask({spctxt.id+1},local{algExpr.opId}_range.start,local{algExpr.opId}_range.end,{ts_size_attributes},{KernelCall.args.ws_rs});')                
            pc.add('if (task != NULL) {')
            if len(spctxt.ts_list_attributes) > 0:
                #pc.add('atomicAdd(&cnt[0], 1);')
                pc.add('char* attr = task->GetAttrPtr();')
                accumulated_size = 0
                for attr in spctxt.ts_list_attributes:
                    loc_src = spctxt.attrLoc[attr.id]
                    dtype = attr.dataType
                    pc.add('{')
                    if dtype == Type.STRING:
                        pc.add(f'*((str_t*) attr) = {loc_src};')
                        #pc.add(f'*((char**) attr) = {loc_src}.start;')
                        #pc.add(f'*((char**) (attr + 8)) = {loc_src}.end;')
                    elif dtype == Type.PTR_INT:
                        pc.add(f'*((int**) attr) = (int*) {loc_src};')
                    else:
                        pc.add(f'*(({langType(dtype)}*)attr) = {loc_src};') 
                    pc.add('}')
                    pc.add(f'attr += {CType.size[langType(dtype)]};')
                    accumulated_size += CType.size[langType(dtype)]
            pc.add('taskstack->Push(task);')
            pc.add('}')
            pc.add(f'push_active_mask = __ballot_sync(ALL_LANES, active & (local{algExpr.opId}_range.start < local{algExpr.opId}_range.end));')
            if KernelCall.args.mode == 'stats':
                pc.add('{')
                pc.add('unsigned long long current_tp = clock64();')
                pc.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
                pc.add('tp = current_tp;')
                pc.add('current_status = TYPE_STATS_PROCESSING;')
                pc.add('}')
            pc.add('}')
                
        
        
        
        
        pc.add('if (push_active_mask) {')


        # 3-1. Push trapezoids to a stack (update inodes_cnts...)
        #if KernelCall.args.mode == 'stats':
        #    pc.add('int inodes_cnts_before = inodes_cnts;')
        
        if spctxt.ts_type == 1:            
            pc.add(f'ts_{spctxt.id+1}_range = local{algExpr.opId}_range;')
        elif spctxt.ts_type == 2:
            pc.add(f'int old_ts_cnt =  __shfl_sync(ALL_LANES, inodes_cnts, {spctxt.id+1});')
            pc.add(f'int ts_cnt = old_ts_cnt + __popc(push_active_mask);')
            pc.add('if (thread_id == {}) inodes_cnts = ts_cnt;'.format(spctxt.id+1))
            pc.add(f'Themis::UpdateMaskAtIfLvlAfterPush({spctxt.id+1}, ts_cnt, mask_32, mask_1);')
        
        # 3-2. Push attributes to the stack
        
        if spctxt.ts_type == 1:
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                if attr.dataType == Type.STRING:
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}.start = (char*) {loc_src}.start;')
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}.end = (char*) {loc_src}.end;')
                elif attr.dataType == Type.PTR_INT:
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name} = (int*) {loc_src};')
                else:
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name} = {loc_src};')
            pc.add(f'int ts_src = 32;')
            pc.add(f'Themis::DistributeFromPartToDPart(thread_id, {spctxt.id+1}, ts_src, ts_{spctxt.id+1}_range, ts_{spctxt.id+1}_range_cached);')
            pc.add(f'Themis::UpdateMaskAtLoopLvl({spctxt.id+1}, ts_{spctxt.id+1}_range_cached, mask_32, mask_1);')
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                pc.add('{')
                if attr.dataType == Type.STRING:
                    pc.add(f'char* ts_{spctxt.id+1}_attr_{attr.id_name}_cached0_start = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}.start, ts_src);')
                    pc.add(f'char* ts_{spctxt.id+1}_attr_{attr.id_name}_cached0_end = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}.end, ts_src);')
                    pc.add('if (ts_src < 32) {')
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}_cached.start = ts_{spctxt.id+1}_attr_{attr.id_name}_cached0_start;')
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}_cached.end = ts_{spctxt.id+1}_attr_{attr.id_name}_cached0_end;')
                    pc.add('}')
                elif attr.dataType == Type.PTR_INT:
                    pc.add(f'{langType(attr.dataType)} ts_{spctxt.id+1}_attr_{attr.id_name}_cached0 = (int*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}, ts_src);')
                    pc.add(f'if (ts_src < 32) ts_{spctxt.id+1}_attr_{attr.id_name}_cached = ts_{spctxt.id+1}_attr_{attr.id_name}_cached0;')
                else:
                    pc.add(f'{langType(attr.dataType)} ts_{spctxt.id+1}_attr_{attr.id_name}_cached0 = __shfl_sync(ALL_LANES, ts_{spctxt.id+1}_attr_{attr.id_name}, ts_src);')
                    pc.add(f'if (ts_src < 32) ts_{spctxt.id+1}_attr_{attr.id_name}_cached = ts_{spctxt.id+1}_attr_{attr.id_name}_cached0;')
                pc.add('}')

        
        if spctxt.ts_type == 2 and len(spctxt.ts_list_attributes) > 0:
            pc.add('if (ts_cnt >= 32) {')            
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                # store attributes to a trapezoid stack
                if attr.dataType == Type.STRING:
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}.start = {attr.id_name}.start;')
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name}.end = {attr.id_name}.end;')
                else:
                    pc.add(f'ts_{spctxt.id+1}_attr_{attr.id_name} = {loc_src};')
            
            pc.add('if (ts_cnt - old_ts_cnt < 32) {')
            pc.add('unsigned ts_src = 32;')
            pc.add('if (!active) ts_src = old_ts_cnt - __popc((~push_active_mask) & prefixlanes) - 1;')
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                pc.add('{')
                if attr.dataType == Type.STRING:
                    pc.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}_flushed.start, ts_src);')
                    pc.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}_flushed.end, ts_src);')
                elif attr.dataType == Type.PTR_INT:
                    pc.add(f'{langType(attr.dataType)} cache = (int*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id+1}_attr_{attr.id_name}_flushed, ts_src);')
                else:
                    pc.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, ts_{spctxt.id+1}_attr_{attr.id_name}_flushed, ts_src);')
                # store attributes to a trapezoid stack
                if attr.dataType == Type.STRING:
                    pc.add('if (ts_src < 32) {')
                    pc.add('ts_{}_attr_{}.start = start;'.format(spctxt.id+1, attr.id_name))
                    pc.add('ts_{}_attr_{}.end = end;'.format(spctxt.id+1, attr.id_name))
                    pc.add('}')
                else:
                    pc.add('if (ts_src < 32) ts_{}_attr_{} = cache;'.format(spctxt.id+1, attr.id_name))
                
                
                pc.add('}')
                    
            pc.add('}')
            
            pc.add('} else {')
            pc.add('active_thread_ids[threadIdx.x] = 32;')
            pc.add('int* src_thread_ids = active_thread_ids + ((threadIdx.x >> 5) << 5);')
            pc.add('if (active) src_thread_ids[__popc(push_active_mask & prefixlanes)] = thread_id;')
            pc.add('unsigned ts_src = thread_id >= old_ts_cnt && thread_id < ts_cnt ? src_thread_ids[thread_id - old_ts_cnt] : 32;')
            
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                pc.add('{')
                if attr.dataType == Type.STRING:
                    pc.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.start, ts_src);')
                    pc.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.end, ts_src);')
                elif attr.dataType == Type.PTR_INT:
                    pc.add(f'{langType(attr.dataType)} cache = (int*) __shfl_sync(ALL_LANES, (uint64_t) {loc_src}, ts_src);')
                else:
                    pc.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {loc_src}, ts_src);')
                
                # store attributes to a trapezoid stack
                
                if attr.dataType == Type.STRING:
                    pc.add('if (ts_src < 32) {')
                    pc.add('ts_{}_attr_{}_flushed.start = start;'.format(spctxt.id+1, attr.id_name))
                    pc.add('ts_{}_attr_{}_flushed.end = end;'.format(spctxt.id+1, attr.id_name))
                    pc.add('}')
                else:
                    pc.add('if (ts_src < 32) ts_{}_attr_{}_flushed = cache;'.format(spctxt.id+1, attr.id_name))
                
                pc.add('}')
            pc.add('}')
        pc.add('}')
        
        if KernelCall.args.switch:
            pc.add('if (num_nodes_at_next_lvl < 32) break;')
        else:
            loop_lvl = spctxt.loop_lvl
            pc.add(f'if (!(mask_32 & (0x1 << {spctxt.id+1}))) ' + '{')
            pc.add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
            pc.add(f'else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
            pc.add(f'continue;')
            pc.add('}')
        
        pc.add(f'lvl = {spctxt.id+1};')
        

    def genShuffle(self, spctxt, passTid=False):
        algExpr = self.algExpr
        
        attrToMaterialize = {}
        for attrId, attr in algExpr.outRelation.items():
            if attrId in spctxt.pctxt.materialized:
                attrToMaterialize[attrId] = attr
            else:
                for attrTid in spctxt.attrOriginLoc[attrId].tids.values():
                    attrToMaterialize[attrTid.id] = attrTid

        spctxt.ts_size_attributes = 0        
        for attrId, attr in attrToMaterialize.items():
            if passTid == False and spctxt.ts_tid != None and spctxt.ts_tid.id == attrId:
                continue
            spctxt.toReg(attr)
            spctxt.ts_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
            spctxt.ts_list_attributes.append(attr)
        spctxt.ts_speculated_size_attributes += spctxt.ts_size_attributes

        pc = spctxt.postcode

        pc.add('{')
        for attr in spctxt.ts_list_attributes:
            loc_src = spctxt.attrLoc[attr.id]
            if attr.dataType == Type.STRING:
                pc.add('__shared__ str_t shm{}_{}[{}];'.format(algExpr.opId, attr.id_name, KernelCall.defaultBlockSize))
            else: # attr.dataType == Type.INT:
                pc.add('__shared__ {} shm{}_{}[{}];'.format(langType(attr.dataType), algExpr.opId, attr.id_name, KernelCall.defaultBlockSize))

        pc.add('//Initialize infos')
        pc.add('if (threadIdx.x == 0) {')
        pc.add('\tshmNumWarps = 0;')
        pc.add('}')
        pc.add('int actThreadsB = __syncthreads_count(active);')
        pc.add(f'if (((isLast && blockIdx.x == {spctxt.id}) || (!isLast && actThreadsB > 0)))'  + '{')
        pc.add('//Vote')
        pc.add('unsigned warpActive = __ballot_sync(ALL_LANES, active);')
        pc.add('int actThreadsW = __popc(warpActive);')
        
        pc.add('int numWarps = 0;')
        if int(KernelCall.args.pyper_grid_threshold) > 0:
            pc.add('if ((!isLast && actThreadsB <= {}) || (isLast && blockIdx.x == {}))'.format(KernelCall.args.pyper_grid_threshold, spctxt.id) + '{')
        else:
            pc.add('if (false) {')
        pc.add('//GridShuffle')
        
        pc.add('if (thread_id == 0) shmWarpActive[threadIdx.x / 32] = warpActive; ')
        pc.add('__syncthreads();')
        pc.add('//Push existing tuples to the shared memory buffer')
        pc.add('if (active) {')
        pc.add('int dst = 0;')
        pc.add('int subwarp_id = threadIdx.x / 32;')
        pc.add('for (int i = 0; i < subwarp_id; ++i) dst += __popc(shmWarpActive[i]);')
        pc.add('dst += __popc(warpActive & prefixlanes);')
        for attr in spctxt.ts_list_attributes:
            loc_src = spctxt.attrLoc[attr.id]
            if attr.dataType == Type.STRING:
                pc.add('shm{}_{}[dst].start = {}.start;'.format(algExpr.opId, attr.id_name, loc_src))
                pc.add('shm{}_{}[dst].end = {}.end;'.format(algExpr.opId, attr.id_name, loc_src))
            else: # attr.dataType == Type.INT:
                pc.add('shm{}_{}[dst] = {};'.format(algExpr.opId, attr.id_name, loc_src))
        pc.add('}')
        pc.add('//Get the lock of the global memory buffer')
        pc.add('if (threadIdx.x == 0) {')
        pc.add('if (!isLast) {')
        pc.add('while (0 != atomicCAS(global{}_{}_lock, 0, warp_id+1));'.format(spctxt.pctxt.id, spctxt.id))
        pc.add('shmNumInBuffer = atomicCAS(global{}_{}_num, -1, -1);'.format(spctxt.pctxt.id, spctxt.id))
        pc.add('} else {')
        pc.add(f'shmNumInBuffer = *global{spctxt.pctxt.id}_{spctxt.id}_num;')
        pc.add('}')
        pc.add('}')
        pc.add('__syncthreads();')
        pc.add('int numInBuffer = shmNumInBuffer;')
        pc.add('int newNumInBuffer;')
        pc.add('if ((!isLast) && (numInBuffer + actThreadsB < {}))'.format(KernelCall.defaultBlockSize) + '{ // push tuples')
        pc.add('//Push existing tuples in the shared memory buffer to the global memory buffer')
        pc.add('if (threadIdx.x < actThreadsB) {')
        for attr in spctxt.ts_list_attributes:
            loc_src = spctxt.attrLoc[attr.id]
            if attr.dataType == Type.STRING:
                pc.add('global{}_{}_{}[numInBuffer+threadIdx.x].start = shm{}_{}[threadIdx.x].start;'
                    .format(spctxt.pctxt.id, spctxt.id, attr.id_name, algExpr.opId, attr.id_name))
                pc.add('global{}_{}_{}[numInBuffer+threadIdx.x].end = shm{}_{}[threadIdx.x].end;'
                    .format(spctxt.pctxt.id, spctxt.id, attr.id_name, algExpr.opId, attr.id_name))
            else: # attr.dataType == Type.INT:
                pc.add('global{}_{}_{}[numInBuffer+threadIdx.x] = shm{}_{}[threadIdx.x];'
                    .format(spctxt.pctxt.id, spctxt.id, attr.id_name, algExpr.opId, attr.id_name))
        pc.add("__threadfence();")
        pc.add('}')
        pc.add('newNumInBuffer = numInBuffer + actThreadsB;')
        pc.add('active = false;')
        pc.add('} else { // pull tuples')
        pc.add('if (numInBuffer > 0) {')
        pc.add('//Pull existing tuples in the global memory buffer')
        pc.add('int num_to_pull = ({} - actThreadsB) < numInBuffer ? ({} - actThreadsB) : numInBuffer;'
            .format(KernelCall.defaultBlockSize, KernelCall.defaultBlockSize))
        pc.add('if (threadIdx.x < actThreadsB) {')
        for attr in spctxt.ts_list_attributes:
            loc_src = spctxt.attrLoc[attr.id]
            if attr.dataType == Type.STRING:
                pc.add('{}.start = shm{}_{}[threadIdx.x].start;'
                    .format(loc_src, algExpr.opId, attr.id_name))
                pc.add('{}.end = shm{}_{}[threadIdx.x].end;'
                    .format(loc_src, algExpr.opId, attr.id_name))
            else: # attr.dataType == Type.INT:
                pc.add('{} = shm{}_{}[threadIdx.x];'
                    .format(loc_src, algExpr.opId, attr.id_name))
        pc.add('} else if (threadIdx.x < (actThreadsB + numInBuffer)) {')
        for attr in spctxt.ts_list_attributes:
            loc_src = spctxt.attrLoc[attr.id]
            if attr.dataType == Type.STRING:
                pc.add('{}.start = global{}_{}_{}[numInBuffer - num_to_pull + threadIdx.x - actThreadsB].start;'
                    .format(loc_src, spctxt.pctxt.id, spctxt.id, attr.id_name))
                pc.add('{}.end = global{}_{}_{}[numInBuffer - num_to_pull + threadIdx.x - actThreadsB].end;'
                    .format(loc_src, spctxt.pctxt.id, spctxt.id, attr.id_name))
            else: # attr.dataType == Type.INT:
                pc.add('{} = global{}_{}_{}[numInBuffer - num_to_pull + threadIdx.x - actThreadsB];'
                    .format(loc_src, spctxt.pctxt.id, spctxt.id, attr.id_name))
        pc.add('}')
        
        pc.add('active = threadIdx.x < (actThreadsB + numInBuffer);')
        pc.add('newNumInBuffer = numInBuffer - num_to_pull;')
        pc.add('keepGoing = false;')
        pc.add('} else {')
        pc.add('newNumInBuffer = numInBuffer;')
        pc.add('}')
        pc.add('}')
        pc.add('if (threadIdx.x == 0) {')
        pc.add('if (!isLast) {')
        pc.add('atomicCAS(global{}_{}_num, numInBuffer, newNumInBuffer);'.format(spctxt.pctxt.id, spctxt.id))
        pc.add('atomicCAS(global{}_{}_lock, warp_id+1, 0);'.format(spctxt.pctxt.id, spctxt.id))
        pc.add('} else {')
        pc.add(f'*global{spctxt.pctxt.id}_{spctxt.id}_num = newNumInBuffer;')
        pc.add('}')
        pc.add('}')
        pc.add('}')
        
        num_warps_per_block = int(KernelCall.defaultBlockSize / 4)
        if num_warps_per_block > 1:
            intra_block_threshold = 2 # The configuration of the Pyper paper
            intra_warp_threshold = 24
            if num_warps_per_block < 3:
                intra_block_threshold = 1
                intra_warp_threshold = 24
                
            pc.add(f'else if (!isLast && actThreadsW <= {intra_warp_threshold} && thread_id == 0) ' + '{')
            pc.add('\tatomicAdd(&shmNumWarps, 1);')
            pc.add('}')
            pc.add('__syncthreads();')
            pc.add('numWarps = shmNumWarps;')
            pc.add(f'if (numWarps > {intra_block_threshold}) ' + '{')
            pc.add('if (thread_id == 0) shmWarpActive[threadIdx.x / 32] = warpActive; ')
            pc.add('__syncthreads();')
            
            pc.add('if (active) {')
            pc.add('int dst = 0;')
            pc.add('int subwarp_id = threadIdx.x / 32;')
            pc.add('for (int i = 0; i < subwarp_id; ++i) dst += __popc(shmWarpActive[i]);')
            pc.add('dst += __popc(warpActive & prefixlanes);')
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                if attr.dataType == Type.STRING:
                    pc.add('shm{}_{}[dst].start = {}.start;'.format(algExpr.opId, attr.id_name, loc_src))
                    pc.add('shm{}_{}[dst].end = {}.end;'.format(algExpr.opId, attr.id_name, loc_src))
                else: # attr.dataType == Type.INT:
                    pc.add('shm{}_{}[dst] = {};'.format(algExpr.opId, attr.id_name, loc_src))
            pc.add('}')
            pc.add('__syncthreads();')
            
            pc.add('active = threadIdx.x < actThreadsB;')
            pc.add('if (active) {')
            for attr in spctxt.ts_list_attributes:
                loc_src = spctxt.attrLoc[attr.id]
                if attr.dataType == Type.STRING:
                    pc.add('{}.start = shm{}_{}[threadIdx.x].start;'.format(loc_src, algExpr.opId, attr.id_name))
                    pc.add('{}.end = shm{}_{}[threadIdx.x].end;'.format(loc_src, algExpr.opId, attr.id_name))
                else: # attr.dataType == Type.INT:
                    pc.add('{} = shm{}_{}[threadIdx.x];'.format(loc_src, algExpr.opId, attr.id_name))
            pc.add('}')
            
            pc.add('}')

        pc.add('}')
        pc.add('}')


class Scan(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)


    def genScanLoop(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)

        if algExpr.isTempScan:
            spctxt.addVar(Table(self.algExpr.table['name'], self.algExpr.tupleNum, self.algExpr.outRelation))

        for attrId, attr in algExpr.outRelation.items():
            if attrId not in spctxt.attrLoc:
                attr.tid = algExpr.tid                
                if algExpr.isTempScan:
                    spctxt.setLoc(attrId, Location.temptable(algExpr.tableName, attr, algExpr.tid))
                else:
                    spctxt.setLoc(attrId, Location.table(attr, algExpr.tid))
                    spctxt.addVar(Column(attr))

        spctxt.pctxt.scanSize = algExpr.tupleNum

        c = spctxt.pctxt.precode
        
            
        if algExpr.isTempScan: 
            c.add('ts_0_range_cached.end = *nout_{};'.format(algExpr.tableName))
            spctxt.pctxt.num_inodes_at_lvl_zero = f'cpu_nout_{algExpr.tableName}'
        else: 
            c.add('ts_0_range_cached.end = {};'.format(algExpr.tupleNum))
            spctxt.pctxt.num_inodes_at_lvl_zero = f'{algExpr.tupleNum}'

        if KernelCall.args.lb and KernelCall.args.lb_mode == 'morsel':
            c.add('input_table_size = ts_0_range_cached.end;')
            c.add('if (thread_id == 0) {')
            c.add('ts_0_range_cached.start = 32 * 16 * warp_id;')
            c.add('ts_0_range_cached.start = ts_0_range_cached.start < input_table_size ? ts_0_range_cached.start : input_table_size;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.start + 32 * 16;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.end < input_table_size ? ts_0_range_cached.end : input_table_size;')
            c.add('inodes_cnts = ts_0_range_cached.end - ts_0_range_cached.start;')
            c.add('}')
            c.add('ts_0_range_cached.start = __shfl_sync(ALL_LANES, ts_0_range_cached.start, 0) + thread_id;')
        else:
            if KernelCall.args.lb:
                c.add(f'int global_scan_end = ts_0_range_cached.end;')
                c.add(f'Themis::PullINodesAtZeroLvlDynamically<{spctxt.gctxt.num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
            else:
                c.add(f'Themis::PullINodesAtZeroLvlStatically(thread_id, ts_0_range_cached, inodes_cnts);')
                
            #c.add('if (thread_id == 0) {')
            #c.add('\tts_0_range_cached.start = 32 * warp_id;')
            #c.add('\tinodes_cnts = (ts_0_range_cached.end % (blockDim.x * gridDim.x)) - warp_id * 32;')
            #c.add('\tinodes_cnts = inodes_cnts <= 0 ? 0 : (inodes_cnts >= 32 ? 32 : inodes_cnts % 32);')
            #c.add('\tinodes_cnts += (ts_0_range_cached.end / (blockDim.x * gridDim.x)) * 32;')
            #c.add('}')
            #if KernelCall.args.mode == 'stats':
            #    c.add('\tif (warp_id == 0 && thread_id == 0) atomicAdd(&global_stats_per_lvl[0].num_inodes, ts_0_range_cached.end - ts_0_range_cached.start);')
            
        

        
    def genScan(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        algExpr.tid.tid = algExpr.tid
        spctxt.pctxt.materialized.add(algExpr.tid.id)
        spctxt.declared.add(algExpr.tid.id)

        if algExpr.isTempScan:
            spctxt.addVar(Table(self.algExpr.table['name'], self.algExpr.tupleNum, self.algExpr.outRelation))

        for attrId, attr in algExpr.outRelation.items():
            if attrId not in spctxt.attrLoc:
                attr.tid = algExpr.tid              
                if algExpr.isTempScan:
                    spctxt.setLoc(attrId, Location.temptable(algExpr.tableName, attr, algExpr.tid))
                else:
                    spctxt.setLoc(attrId, Location.table(attr, algExpr.tid))
                    spctxt.addVar(Column(attr))
                
                if KernelCall.args.use_pos_vec == False:
                    spctxt.toReg(attr)
                    spctxt.pctxt.materialized.add(attrId)
                    spctxt.attrOriginLoc[attr.id] = Location.reg(attr)

        spctxt.pctxt.scanSize = algExpr.tupleNum

        c = spctxt.pctxt.precode
        c.add('int {} = scan_offset + threadIdx.x;'.format(algExpr.tid.id_name))        
        if algExpr.isTempScan:
            #assert(False)
            c.add('active = !isLast && ({} < *nout_{} && select_{}[{}]);'.format(algExpr.tid.id_name, algExpr.tableName,algExpr.tableName,algExpr.tid.id_name))
            c.add('if ((isLast && !keepGoing) || (!isLast && __syncthreads_count(active) == 0)) break;')
        else:
            c.add('active = !isLast && ({} < {});'.format(algExpr.tid.id_name, algExpr.tupleNum))
            
            c.add(f'if ((isLast && !keepGoing) ||  (!isLast && scan_offset >= {algExpr.tupleNum})) break;')


    def genSingleScan(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)
        spctxt.declared.add(algExpr.tid.id)

        if algExpr.isTempScan:
            spctxt.addVar(Table(self.algExpr.table['name'], self.algExpr.tupleNum, self.algExpr.outRelation))

        for attrId, attr in algExpr.outRelation.items():
            if attrId not in spctxt.attrLoc:
                attr.tid = algExpr.tid           
                if algExpr.isTempScan:
                    spctxt.setLoc(attrId, Location.temptable(algExpr.tableName, attr, algExpr.tid))
                else:
                    spctxt.setLoc(attrId, Location.table(attr, algExpr.tid))
                    spctxt.addVar(Column(attr))

        spctxt.pctxt.scanSize = algExpr.tupleNum

        c = spctxt.pctxt.precode
        c.add('if (thread_id != 0) return;')
        c.add('int {} = 0;'.format(algExpr.tid.id_name))
        if algExpr.isTempScan:
            c.add('for (; {} < *nout_{}; ++{})'
                .format(algExpr.tid.id_name, algExpr.tableName, algExpr.tid.id_name) + '{')
        else:
            c.add('for (; {} < {}; ++{})'
                .format(algExpr.tid.id_name, algExpr.tupleNum, algExpr.tid.id_name) + '{')
        c.add('active = true;')



    def genOperation(self, spctxt):
        pass


class Probe(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genOperation(self, spctxt):

        for attrId, attr in algExpr.joinAttributes.items():
            pass
            
        pass

class IntersectionSelection(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        pass

    def genOperation(self, spctxt):
        algExpr = self.algExpr

        spctxt.setLoc(algExpr.arr_val.id, Location.reg(algExpr.arr_val))
        spctxt.pctxt.materialized.add(algExpr.arr_val.id)
        loc_v = spctxt.declare(algExpr.arr_val)

        loc_tid = spctxt.toReg(algExpr.tid)
        loc_arr = spctxt.toReg(algExpr.arr)

        for attrId, attr in algExpr.outRelation.items():
            if attrId in spctxt.attrLoc:
                continue
            spctxt.setLoc(attrId, Location.table(attr, attr.tid))
            if attr.keyIdx == -1: # not key
                spctxt.addVar(Column(attr))
            else:
                pass

        ac = spctxt.activecode
        ac.add('{} = {}[{}];'.format(loc_v, loc_arr, loc_tid))
        for attrId, attr in algExpr.joinAttributes.items():
            tid = attr.tid
            loc_src = spctxt.toReg(tid)
            #spctxt.setLoc(tid.id, Location.reg(tid))
            #spctxt.pctxt.materialized.add(tid.id)
            #loc_src = spctxt.declare(tid)
            trieName, trieKeys = attr.tid.trieName, attr.tid.trieKeys
            ac.add('if (active) {')
            if attr.keyIdx == len(attr.tid.conditions):
                ac.add('\t{} = pre_{};'.format(loc_src, loc_src))
            if attr.keyIdx == 0: offset = '{}_offset'.format(trieName)
            else: offset = '{}_{}_offset'.format(trieName, trieKeys[attr.keyIdx-1])
            ac.add('int new_tid=-1;')
            ac.add('\tif (({}_{}_val + {}[{}*2]) == {})'.format(attr.tid.trieName, attr.name, offset, loc_src, loc_arr) + '{')
            ac.add('\t\tnew_tid = ({}-{}_{}_val) + {};'.format(loc_arr, attr.tid.trieName, attr.name, loc_tid))
            ac.add('} else {')
            ac.add('\t\tactive = trieSearch({}_{}_val, {}, {}, {}, new_tid);'
                .format(attr.tid.trieName, attr.name, offset, loc_v, loc_src))
            ac.add('\t}')
            ac.add('\t{} = new_tid;'.format(loc_src))
            ac.add('}')
    
    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 2
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return False

class Intersection(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genOperation(self, spctxt):
        algExpr = self.algExpr
        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)

        spctxt.setLoc(algExpr.arr.id, Location.reg(algExpr.arr))
        spctxt.pctxt.materialized.add(algExpr.arr.id)

        for attrId, attr in algExpr.outRelation.items():
            if attrId in spctxt.attrLoc:
                continue
            spctxt.setLoc(attrId, Location.table(attr, attr.tid))
            if attr.keyIdx == -1: # not key
                spctxt.addVar(Column(attr))
            else:
                pass

        # 1. do pre-processing (e.g., seaching tries for the given key values)
        c = spctxt.pctxt.precode
        for attr in algExpr.tids:
            spctxt.addVar(Trie(attr))
            c.add('int pre_{} = 0;'.format(attr.id_name))
            trieName = attr.trieName
            trieKeys = attr.trieKeys
            for idx, cond in enumerate(attr.conditions):

                if idx == 0: offset = '{}_offset'.format(trieName)
                else: offset = '{}_{}_offset'.format(trieName, trieKeys[idx-1])

                cond_attr = attr.trieKeyAttrs[idx]
                spctxt.addVar(TrieColumn(cond_attr))

                c.add('trieSearch({}_{}_val, {}, {}, pre_{}, pre_{});'
                    .format(trieName, trieKeys[idx], offset, cond, attr.id_name, attr.id_name))


        # 2. do ..
        loc_range = "local{}_range".format(algExpr.opId)
        spctxt.precode.add("Themis::Range {};".format(loc_range))

        ac = spctxt.activecode
        loc_arr = spctxt.attrLoc[algExpr.arr.id]
        ac.add('{} = NULL;'.format(loc_arr))
        ac.add('{}.start = 0;'.format(loc_range))
        for attrId, attr in algExpr.joinAttributes.items():
            spctxt.addVar(TrieColumn(attr))
            
            ac.add('{')
            if attr.keyIdx == len(attr.tid.conditions):
                spctxt.declare(attr.tid)
                loc_tid = spctxt.setLoc(attr.tid.id, Location.reg(attr.tid))
                spctxt.pctxt.materialized.add(attr.tid.id)
                ac.add('\t{} = pre_{};'.format(loc_tid, loc_tid))
            else:
                loc_tid = spctxt.toReg(attr.tid)
            
            if attr.keyIdx == 0:
                offset = '{}_offset'.format(attr.tid.trieName)
            else:
                offset = '{}_{}_offset'.format(attr.tid.trieName, attr.tid.trieKeys[attr.keyIdx-1])
            ac.add('\tint start = {}[{}*2];'.format(offset, loc_tid))
            ac.add('\tint end = {}[{}*2+1];'.format(offset, loc_tid))
            ac.add('\tif ({} == NULL || ({}.end > (end-start)))'.format(loc_arr, loc_range, loc_range) + '{')
            ac.add('\t\t{}.end = end - start;'.format(loc_range))
            ac.add('\t\t{} = {}_{}_val + start;'.format(loc_arr, attr.tid.trieName, attr.name))
            ac.add('\t}')
            ac.add('}')


    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 1
        spctxt.ts_tid = self.algExpr.tid
        spctxt.ts_tid_build = "loopvar{}".format(spctxt.id+1)
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return True

    def genTidBuild(self, spctxt):
        return "loopvar{}".format(spctxt.id+1)

    

class TrieScan(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        
    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 2
        super().genPushTrapzoids(spctxt)

    
    def doExpansion(self):
        return False
    
    def genOperation(self, spctxt):

        algExpr = self.algExpr
        ac = spctxt.activecode
        pc = spctxt.precode

        loc_arr = spctxt.attrLoc[algExpr.arr.id]
        loc_tid = spctxt.attrLoc[algExpr.tid.id]

        spctxt.setLoc(algExpr.arr_val.id, Location.reg(algExpr.arr_val))
        spctxt.pctxt.materialized.add(algExpr.arr_val.id)
        loc_v = spctxt.declare(algExpr.arr_val)

        ac.add('{} = {}[{}];'.format(loc_v, loc_arr, loc_tid))
        for attrId, attr in algExpr.joinAttributes.items():
            tid = attr.tid
            spctxt.setLoc(tid.id, Location.reg(tid))
            spctxt.pctxt.materialized.add(tid.id)
            loc_src = spctxt.declare(tid)
            
            trieName, trieKeys = attr.tid.trieName, attr.tid.trieKeys
            
            ac.add('if (active) {')
            ac.add('\t{} = pre_{};'.format(loc_src, loc_src))
            
            if attr.keyIdx == 0: offset = '{}_offset'.format(trieName)
            else: offset = '{}_{}_offset'.format(trieName, tid.trieKeys[attr.keyIdx-1])
            ac.add('int new_tid=-1;')
            ac.add('\tif (({}_{}_val + {}[{}*2]) == {})'.format(trieName, attr.name, offset, loc_src, loc_arr) + '{')
            ac.add('\t\tnew_tid = ({}-{}_{}_val) + {};'.format(loc_arr, trieName, attr.name, loc_tid))
            ac.add('} else {')
            ac.add('\t\tactive = trieSearch({}_{}_val, {}, {}, {}, new_tid);'
                .format(attr.tid.trieName, attr.name, offset, loc_v, loc_src))
            ac.add('\t}')
            ac.add('\t{} = new_tid;'.format(loc_src))
            ac.add('}')


    def genScanLoop(self, spctxt):
        algExpr = self.algExpr
        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)

        spctxt.setLoc(algExpr.arr.id, Location.reg(algExpr.arr))
        spctxt.pctxt.materialized.add(algExpr.arr.id)

        for attrId, attr in algExpr.outRelation.items():
            if attrId in spctxt.attrLoc:
                continue
            spctxt.setLoc(attrId, Location.table(attr, attr.tid))
            if attr.keyIdx == -1: # not key
                spctxt.addVar(Column(attr))
            else:
                pass
        loc_arr = spctxt.attrLoc[algExpr.arr.id]


        c = spctxt.pctxt.precode_cpu

        spctxt.pctxt.num_inodes_at_lvl_zero = f'range_end'
        spctxt.pctxt.vars.append(SingleVar('int*', spctxt.attrLoc[algExpr.arr.id]))
        spctxt.pctxt.vars.append(SingleVar('int', 'range_end'))
        c.add('int* {} = NULL;'.format(spctxt.attrLoc[algExpr.arr.id]))
        c.add('int range_end = 0;')
        for attr in algExpr.tids:
            c.add(f'int pre_{attr.id_name} = 0;')
            spctxt.pctxt.vars.append(SingleVar('int', f'pre_{attr.id_name}'))
        c.add('{')
        # 1. do pre-processing (e.g., seaching tries for the given key values)
        #c = spctxt.pctxt.precode
        for attr in algExpr.tids:
            spctxt.addVar(Trie(attr))
            trieName = attr.trieName
            trieKeys = attr.trieKeys
            for idx, cond in enumerate(attr.conditions):
                if idx == 0: offset = '{}_offset'.format(trieName)
                else: offset = '{}_{}_offset'.format(trieName, trieKeys[idx-1])
                c.add('trieSearch(mmap_{}_{}_val, mmap_{}, {}, pre_{}, pre_{});'
                    .format(trieName, trieKeys[idx], offset, cond, attr.id_name, attr.id_name))       


        for attrId, attr in algExpr.joinAttributes.items():
            loc_tid = attr.tid.id_name
            spctxt.addVar(TrieColumn(attr))
            
            trieName, trieKeys = attr.tid.trieName, attr.tid.trieKeys

            if attr.keyIdx == 0: offset = '{}_offset'.format(trieName)
            else: offset = '{}_{}_offset'.format(trieName, trieKeys[attr.keyIdx-1])
            
            c.add('{')
            c.add('\tint trie_size = trieSize(mmap_{}, pre_{});'.format(offset, loc_tid))
            c.add('\tif ({} == NULL || (range_end > trie_size)) '.format(loc_arr) + '{')
            c.add('\t\trange_end = trie_size;')
            c.add('\t\t{} = {}_{}_val + mmap_{}[2*pre_{}];'.format(loc_arr, attr.tid.trieName, attr.name, offset, loc_tid))
            c.add('\t}')
            c.add('}')
        c.add('}')

        c = spctxt.pctxt.precode
        c.add('ts_0_range_cached.end = range_end;')
        if KernelCall.args.lb and KernelCall.args.lb_mode == 'morsel':
            c.add('input_table_size = ts_0_range_cached.end;')
            c.add('if (thread_id == 0) {')
            c.add('ts_0_range_cached.start = 32 * 16 * warp_id;')
            c.add('ts_0_range_cached.start = ts_0_range_cached.start < input_table_size ? ts_0_range_cached.start : input_table_size;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.start + 32 * 16;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.end < input_table_size ? ts_0_range_cached.end : input_table_size;')
            c.add('inodes_cnts = ts_0_range_cached.end - ts_0_range_cached.start;')
            c.add('}')
        else:
            
            c.add('if (thread_id == 0) {')
            c.add('\tts_0_range_cached.start = 32 * warp_id;')
            c.add('\tinodes_cnts = (ts_0_range_cached.end % (blockDim.x * gridDim.x)) - warp_id * 32;')
            c.add('\tinodes_cnts = inodes_cnts <= 0 ? 0 : (inodes_cnts >= 32 ? 32 : inodes_cnts % 32);')
            c.add('\tinodes_cnts += (ts_0_range_cached.end / (blockDim.x * gridDim.x)) * 32;')
            #c.add('\tatomicAdd(&global_stats_per_lvl[0].num_inodes, inodes_cnts);')
            #if KernelCall.args.mode == 'stats':
            #    c.add('\tif (warp_id == 0) atomicAdd(&global_stats_per_lvl[0].num_inodes, ts_0_range_cached.end - ts_0_range_cached.start);')

            c.add('}')
        c.add('ts_0_range_cached.start = __shfl_sync(ALL_LANES, ts_0_range_cached.start, 0) + thread_id;')
        
class IndexJoin(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        self.condAttr = list(self.algExpr.conditionAttributes.values())[0]
        self.touched = 0
        
        if KernelCall.args.system == 'Pyper':
            self.doLB = op.doShuffle
            
        self.doConvert = op.doConvert


    def genOperation(self, spctxt):    
        
        algExpr = self.algExpr

        spctxt.addVar(Index(algExpr.rel_name, self.algExpr.ftable, self.algExpr.table['name']))


        if algExpr.touched % 2 == 0:        
            spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
            algExpr.tid.tid = algExpr.tid
            spctxt.pctxt.materialized.add(algExpr.tid.id)

            for attrId, attr in algExpr.outRelation.items():
                if attrId not in spctxt.attrLoc:
                    spctxt.setLoc(attrId, Location.table(attr, algExpr.tid))
                    spctxt.addVar(Column(attr))
                    attr.tid = algExpr.tid
        else:
            #pass
            for attrId, attr in algExpr.outRelation.items():
                spctxt.toReg(attr)

        ac = spctxt.activecode
        if algExpr.unique:
            spctxt.toReg(self.condAttr)
            spctxt.precode.add("Themis::Range local{}_range;".format(algExpr.opId))
            ac.add("active = indexProbeMulti ({}_offset, {}, local{}_range.start, local{}_range.end);"
                .format(algExpr.rel_name, spctxt.attrLoc[self.condAttr.id], algExpr.opId, algExpr.opId))
            spctxt.declare(algExpr.tid)
            
            if self.doConvert:
                ac.add("{} = indexGetPid({}_position, local{}_range.start);"
                    .format(spctxt.attrLoc[algExpr.tid.id],algExpr.rel_name, algExpr.opId))
            else: 
                ac.add("{} = local{}_range.start;"
                    .format(spctxt.attrLoc[algExpr.tid.id],algExpr.opId))
            spctxt.inReg.add(algExpr.tid.id)
            for attrId, attr in algExpr.outRelation.items():
                ac.add(f"//{attr.id_name}")
                spctxt.toReg(attr)
        #elif algExpr.touched % 2 == 0:
        else:
            spctxt.toReg(self.condAttr)
            for attrId, attr in algExpr.outRelation.items():
                ac.add(f"//{attr.id_name}")
                #spctxt.toReg(attr)
            ac.add("//Fuck you man")
            spctxt.precode.add("Themis::Range local{}_range;".format(algExpr.opId))
            ac.add("active = indexProbeMulti ({}_offset, {}, local{}_range.start, local{}_range.end);"
                .format(algExpr.rel_name, spctxt.attrLoc[self.condAttr.id], algExpr.opId, algExpr.opId))

    def genPushTrapzoids(self, spctxt):
        if self.algExpr.unique:
            spctxt.ts_type = 2
        else:
            spctxt.ts_type = 1
            spctxt.ts_tid = self.algExpr.tid
            if self.doConvert: 
                spctxt.ts_tid_build = "indexGetPid({}_position, loopvar{})".format(self.algExpr.rel_name, spctxt.id+1)
            else:
                spctxt.ts_tid_build = f"loopvar{spctxt.id+1}"
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        if self.algExpr.unique:
            return False
        return True

    def genTidBuild(self, spctxt):
        if self.doConvert: return "indexGetPid({}_position, loopvar{})".format(self.algExpr.rel_name, spctxt.id+1)
        else: return f"loopvar{spctxt.id+1}"


class Exist(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        self.doLB = op.doLB            

    def genOperation(self, spctxt):    
        algExpr = self.algExpr
        
        ac = spctxt.activecode 
        ac.add('//hello')
        for attr in algExpr.searchAttrs:
            spctxt.toReg(attr)
                    
        spctxt.precode.add("Themis::Range local{}_range;".format(algExpr.opId))
        
        ac = spctxt.activecode
        ac.add("// Exist check")
        
        for idx, attr in enumerate(algExpr.searchAttrs):
            if idx == 0: continue
            ac.add(f"// Binary search for {spctxt.attrLoc[attr.id]}")
            ac.add("if (active) {")
            ac.add("active = indexProbeMulti (rel_vertex_id__edge_src_offset, {}, local{}_range.start, local{}_range.end);"
                .format(spctxt.attrLoc[algExpr.searchAttrs[idx].id], algExpr.opId, algExpr.opId))            
            ac.add(f"active = active && binarySearch(edge_dst, local{algExpr.opId}_range.start, local{algExpr.opId}_range.end, {spctxt.attrLoc[algExpr.searchAttrs[0].id]});")
            ac.add("}")
        

    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 2
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return False


class Selection(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        

    def genOperation(self, spctxt):
        algExpr = self.algExpr
        cc = spctxt.activecode

        for attrId, attr in algExpr.conditionAttributes.items():
            #print(attrId, attr.name, spctxt.attrLoc[attrId])
            spctxt.toReg(attr)

        cc.add("active = {};".format(algExpr.condition.gen(spctxt)))

    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 2
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return False

class Map(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genOperation(self, spctxt):
        #print("Shit Map!!!!")
        algExpr = self.algExpr
        mapAttr = algExpr.mapAttr
        #print("Map", mapAttr.id, mapAttr)

        spctxt.pctxt.materialized.add(mapAttr.id)

        spctxt.declare(mapAttr)

        cc = spctxt.activecode

        for attrId, attr in algExpr.mappedAttributes.items():
            spctxt.toReg(attr)
        
        spctxt.setLoc(mapAttr.id, Location.reg(mapAttr))
        mapAttr.tid = None
        cc.add("{} = {};".format(
            spctxt.attrLoc[mapAttr.id], algExpr.expression.gen(spctxt)))


class MultiMap(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genOperation(self, spctxt):
        #print("Shit Map!!!!")
        algExpr = self.algExpr
        mapAttrs = algExpr.mapAttrs
        #print("Map", mapAttr.id, mapAttr)

        for attr in mapAttrs:
            spctxt.pctxt.materialized.add(attr.id)
            spctxt.declare(attr)
            attr.tid = -1

        cc = spctxt.activecode

        for attrId, attr in algExpr.mappedAttributes.items():
            spctxt.toReg(attr)
        
        for i, attr in enumerate(mapAttrs):
            spctxt.setLoc(attr.id, Location.reg(attr))
            cc.add("{} = {};".format(
                spctxt.attrLoc[attr.id], algExpr.attrs[i][1].gen(spctxt)))


class EquiJoin(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        algExpr = self.algExpr
        self.size = int(algExpr.leftChild.tupleNum * algExpr.htSizeFactor)

    def __str__(self):
        e = self.algExpr
        return '{} {} {}'.format(e.opId, e.opType, e.outRelation.keys())

    def buildHashkey(self, attrs, spctxt):

        cc = spctxt.activecode

        cc.add('uint64_t hash_key = 0;')

        for attrId, attr in attrs.items():
            spctxt.toReg(attr)
            
        for attrId, attr in attrs.items():
            loc_src = spctxt.attrLoc[attrId]
            
            if attr.dataType == Type.STRING:
                cc.add('hash_key = hash(hash_key + stringHash({}));'
                    .format(loc_src))
            else:
                cc.add('hash_key = hash(hash_key + ((uint64_t) {}));'
                    .format(loc_src))

    def buildPayload(self, attrs_dst, attrs_src, spctxt):
        
        opId = self.algExpr.opId
        cc = spctxt.activecode

        print("buildPayload", opId, attrs_dst, attrs_src)
        
        cc.add('Payload{} payl;'.format(opId))
        for (attrId_dst, attr_dst), (attrId_src, attr_src) in zip(list(attrs_dst.items()), list(attrs_src.items())):
            #print(attrId, attr.name)
            loc_dst = Location.payl(attr_dst)
            spctxt.toReg(attr_src)
            loc_src = spctxt.attrLoc[attrId_src]
            cc.add('{} = {};'.format(loc_dst, loc_src))

    def genHashTable(self, spctxt, isBuild):
        algExpr = self.algExpr
        opId = algExpr.opId
        size = self.size

        buildCondAttributes = {}
        for attrId, attr in algExpr.conditionAttributes.items():
            if attrId in algExpr.conditionProbeAttributes: continue
            buildCondAttributes[attrId] = attr


        if algExpr.joinType == Join.INNER:
            if algExpr.multimatch:
                ds = MultiHT(opId, size, self.algExpr.leftChild.tupleNum*2, self.algExpr.leftChild.outRelation )
            else:
                ds = UniqueHT(opId, size, self.algExpr.leftChild.outRelation)
        elif algExpr.joinType == Join.SEMI or algExpr.joinType == Join.ANTI:
            ds = AggHT(opId, size, algExpr.buildKeyAttributes, buildCondAttributes, aggAttrs={})
        elif algExpr.joinType == Join.OUTER:
            assert(False)
        spctxt.addVar(ds)



    def genOperation(self, spctxt):

        algExpr = self.algExpr
        opId = algExpr.opId
        size = self.size

        self.genHashTable(spctxt, False)

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)

        self.buildHashkey(algExpr.probeKeyAttributes, spctxt)

        pc = spctxt.precode
        ac = spctxt.activecode

        if algExpr.joinType == Join.INNER:
            if algExpr.multimatch:
                for attrId, attr in self.algExpr.leftChild.outRelation.items():
                    if attrId not in spctxt.attrLoc:
                        spctxt.setLoc(attrId, Location.jpayl(algExpr.opId, attr, algExpr.tid))
                        attr.tid = algExpr.tid

                pc.add("Themis::Range local{}_range;".format(opId))
                ac.add('active &= hashProbeMulti(jht{}, {}, hash_key, local{}_range.start, local{}_range.end);'
                    .format(opId, size, opId, opId))
            else:
                for attrId, attr in self.algExpr.leftChild.outRelation.items():
                    if attrId not in spctxt.attrLoc:
                        spctxt.setLoc(attrId, Location.ujht(algExpr.opId, attr, algExpr.tid))
                        attr.tid = algExpr.tid
                spctxt.declare(algExpr.tid)
                spctxt.inReg.add(algExpr.tid.id)
                ac.add('int numLookups = 0;')
                ac.add('int bucketFound = 0;')
                ac.add('active = hashProbeUnique(ujht{}, {}, hash_key, numLookups, {});'
                    .format(algExpr.opId, size, algExpr.tid.id_name))
                ac.add('while (active) {')
                ac.add('\tbucketFound = 1;')

                for (attrId_b, attr_b), (attrId_p, attr_p) in zip(list(algExpr.buildKeyAttributes.items()), list(algExpr.probeKeyAttributes.items())):
                    spctxt.setLoc(attrId_b, Location.ujht(algExpr.opId, attr_b, algExpr.tid))
                    loc_left = spctxt.attrLoc[attrId_b]
                    loc_right = spctxt.attrLoc[attrId_p]
                    ac.add('\tbucketFound &= ({}) == ({});'.format(loc_left, loc_right))
                    
                ac.add('\tif (bucketFound) break;')
                ac.add('\tactive = hashProbeUnique(ujht{}, {}, hash_key, numLookups, {});'
                    .format(algExpr.opId, size, algExpr.tid.id_name))
                ac.add('}')
                ac.add('active = bucketFound;')

        elif algExpr.joinType == Join.SEMI or algExpr.joinType == Join.ANTI:
            spctxt.declare(algExpr.tid)
            spctxt.inReg.add(algExpr.tid.id)
            for attrId, attr in algExpr.conditionAttributes.items():
                if attrId in algExpr.conditionProbeAttributes:
                    spctxt.toReg(attr)
                else:
                    loc_left = Location.aht(algExpr.opId, attr, algExpr.tid)
                    spctxt.setLoc(attrId, loc_left)
                    attr.tid = algExpr.tid
            ac.add('int numLookups = 0;')
            ac.add('active = hashAggregateFindBucket(aht{}, {}, hash_key, numLookups, {});'
                .format(opId, size, algExpr.tid.id_name))
            ac.add('int bucketFound = 0;')
            ac.add('while (active) {')
            ac.add('\tbucketFound = 1;')
            for (attrId_l, attr_l), (attrId_r, attr_r) in zip(list(algExpr.buildKeyAttributes.items()), list(algExpr.probeKeyAttributes.items())):
                loc_left = Location.aht(algExpr.opId, attr_l, algExpr.tid)
                spctxt.setLoc(attrId_l, loc_left)
                spctxt.toReg(attr_l)
                loc_left = spctxt.attrLoc[attrId_l]
                loc_right = spctxt.attrLoc[attrId_r]
                ac.add('\tbucketFound &= {} == {};'.format(loc_left, loc_right))

            for attrId, attr in algExpr.conditionAttributes.items():
                if attrId in algExpr.conditionProbeAttributes:
                    spctxt.toReg(attr)
                else:
                    loc_left = Location.aht(algExpr.opId, attr, algExpr.tid)
                    spctxt.setLoc(attrId, loc_left)
                    attr.tid = algExpr.tid
                    #spctxt.toReg(attr)

            if algExpr.conditions is not None:
                ac.add('bucketFound &= {};'.format(algExpr.conditions.gen(spctxt)))

            ac.add('\tif (bucketFound) break;')
            ac.add('\tactive = hashAggregateFindBucket(aht{}, {}, hash_key, numLookups, {});'
                .format(opId, size, algExpr.tid.id_name))
            ac.add('}')
            if algExpr.joinType == Join.SEMI:
                ac.add('active = bucketFound;')
            else:
                ac.add('active = !bucketFound;')
        elif algExpr.joinType == Join.OUTER:
            pass
        else:
            assert(False)

    def genMaterialize(self, spctxt):

        algExpr = self.algExpr
        opId = algExpr.opId
        size = self.size

        self.genHashTable(spctxt, True)            

        cc = spctxt.activecode
        self.buildHashkey(algExpr.buildKeyAttributes, spctxt)


        if algExpr.joinType == Join.INNER:
            if algExpr.multimatch:
                if algExpr.materialized == 0:        
                    cc.add('hashCountMulti(jht{},{},hash_key);'
                        .format(opId, size, 0))
                    algExpr.materialized = 1
                    spctxt.pctxt.postexeccode.add("scanMultiHT<<<{},{}>>>(jht{},{},jht{}_offset);"
                        .format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize, opId, size, opId))
                elif algExpr.materialized == 1:
                    self.buildPayload(self.algExpr.leftChild.outRelation, self.algExpr.leftChild.outRelation, spctxt)
                    cc.add('hashInsertMulti(jht{}, jht{}_payload, jht{}_offset, {}, hash_key, &payl);'
                        .format(opId, opId, opId, size))
                    algExpr.materialized = 2
                else:
                    assert(False)
            else:
                self.buildPayload(self.algExpr.leftChild.outRelation, self.algExpr.leftChild.outRelation, spctxt)
                cc.add('hashBuildUnique(ujht{},{},hash_key,&payl);'.format(opId, size))
                
        elif algExpr.joinType == Join.SEMI or algExpr.joinType == Join.ANTI:
            spctxt.declare(algExpr.tid)
            self.buildPayload(self.algExpr.leftChild.outRelation, self.algExpr.leftChild.outRelation, spctxt)
            cc.add("int bucketFound = 0;")
            cc.add("int numLookups = 0;")
            cc.add("while (!bucketFound) {")
            cc.add("\t{} = hashAggregateGetBucket(aht{},{},hash_key,numLookups,&payl);".format(algExpr.tid.id_name, algExpr.opId, size))
            cc.add("\tbucketFound = 1;")
            for attrId, attr in algExpr.buildKeyAttributes.items():
                loc_left = Location.payl(attr)
                loc_right = Location.aht(algExpr.opId, attr, algExpr.tid)
                if attr.dataType == Type.STRING:
                    cc.add("\tbucketFound &= stringEquals({}, {});".format(loc_left, loc_right))
                else:
                    cc.add("\tbucketFound &= {} == {};".format(loc_left, loc_right))
            for attrId, attr in algExpr.conditionAttributes.items():
                if attrId in algExpr.conditionProbeAttributes: continue
                loc_left = Location.payl(attr)
                loc_right = Location.aht(algExpr.opId, attr, algExpr.tid)
                if attr.dataType == Type.STRING:
                    cc.add("\tbucketFound &= stringEquals({}, {});".format(loc_left, loc_right))
                else:
                    cc.add("\tbucketFound &= {} == {};".format(loc_left, loc_right))

            cc.add("}")
        elif algExpr.joinType == Join.OUTER:
            pass
        else:
            assert(False)

    def genPushTrapzoids(self, spctxt):

        if self.algExpr.multimatch:
            spctxt.ts_type = 1
            spctxt.ts_tid = self.algExpr.tid
            spctxt.ts_tid_build = "loopvar{}".format(spctxt.id+1)
        else:
            spctxt.ts_type = 2

        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        if self.algExpr.multimatch:
            return True
        return False

    def genTidBuild(self, spctxt):
        return "loopvar{}".format(spctxt.id+1)

class CrossJoin(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genOperation(self, spctxt):
        algExpr = self.algExpr
        opId = algExpr.opId
        tableName = "temp{}".format(opId)

        spctxt.addVar(Table(tableName, self.algExpr.leftChild.tupleNum, self.algExpr.outRelation))
        ac = spctxt.activecode
        pc = spctxt.precode
        pc.add("Themis::Range local{}_range;".format(algExpr.opId))
        ac.add("local{}_range.start = 0;".format(opId))
        ac.add("local{}_range.end = *nout_{};".format(opId, tableName))

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)

        for attrId, attr in self.algExpr.leftChild.outRelation.items():
            if attrId not in spctxt.attrLoc:
                attr.tid = algExpr.tid              
                spctxt.setLoc(attrId, Location.temptable(tableName, attr, algExpr.tid))
        
    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 1
        spctxt.ts_tid = self.algExpr.tid
        spctxt.ts_tid_build = "loopvar{}".format(spctxt.id + 1)
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return True

    def genTidBuild(self, spctxt):
        return "loopvar{}".format(spctxt.id + 1)

    def genMaterialize(self, spctxt):

        algExpr = self.algExpr
        opId = algExpr.opId
        tableName = "temp{}".format(opId)

        spctxt.addVar(Table(tableName, self.algExpr.leftChild.tupleNum, self.algExpr.outRelation))
        c = spctxt.precode
        c.add("int wp, writeMask, numProj;")
        c.add("writeMask = __ballot_sync(ALL_LANES, active);")
        c.add("numProj = __popc(writeMask);")
        c.add("if (thread_id == 0) { ")
        c.add("\twp = atomicAdd(nout_{}, numProj);".format(tableName))
        c.add("}")
        c.add("wp = __shfl_sync(ALL_LANES, wp, 0);")
        c.add("wp = wp + __popc(writeMask & prefixlanes);")

        cc = spctxt.activecode
        
        for attrId, attr in self.algExpr.leftChild.outRelation.items():
            #print(self.algExpr.opType, attr.tableName, attr.id, attr.name, attr.creation)
            #print(spctxt.attrLoc[attrId])
            loc_src = spctxt.toReg(attr)
            cc.add("{}_attr{}_{}[wp] = {};".format(tableName, attr.id, attr.name, loc_src))


class Aggregation(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)
        self.size = 1 if not self.algExpr.doGroup else self.algExpr.tupleNum * 2


    def genScanLoop(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)
        
        spctxt.addVar(AggHT(algExpr.opId, self.size, algExpr.groupAttributes, {}, aggAttrs=algExpr.aggregateTuplesCreated))

        if len(algExpr.avgAggregates) > 0:
            aggAttr, inputIdentifier, reductionType = list(algExpr.aggregateTuplesCreated.values())[-1]
            loc_cnt = Location.apayl(algExpr.opId, aggAttr, algExpr.tid)

        for attrId, attr in algExpr.outRelation.items():
            attr.tid = algExpr.tid
            if attrId in algExpr.groupAttributes:
                spctxt.setLoc(attrId, Location.aht(algExpr.opId, attr, algExpr.tid))
            else:
                inId, reductionType = algExpr.aggregateTuples[attrId]
                if reductionType == Reduction.AVG:
                    loc = Location.apayl(algExpr.opId, attr, algExpr.tid)    
                    spctxt.setLoc(attrId, Location.divide(loc, loc_cnt))
                else:
                    spctxt.setLoc(attrId, Location.apayl(algExpr.opId, attr, algExpr.tid))
                    
        spctxt.pctxt.scanSize = self.size
        
        c = spctxt.pctxt.precode

        spctxt.pctxt.num_inodes_at_lvl_zero = f'{self.size}'
        c.add('//Scan aggregation table')
        c.add(f'ts_0_range_cached.end = {self.size};')
        if KernelCall.args.lb and KernelCall.args.lb_mode == 'morsel':
            c.add('input_table_size = ts_0_range_cached.end;')
            c.add('if (thread_id == 0) {')
            c.add('ts_0_range_cached.start = 32 * 16 * warp_id;')
            c.add('ts_0_range_cached.start = ts_0_range_cached.start < input_table_size ? ts_0_range_cached.start : input_table_size;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.start + 32 * 16;')
            c.add('ts_0_range_cached.end = ts_0_range_cached.end < input_table_size ? ts_0_range_cached.end : input_table_size;')
            c.add('inodes_cnts = ts_0_range_cached.end - ts_0_range_cached.start;')
            c.add('}')
            c.add('ts_0_range_cached.start = __shfl_sync(ALL_LANES, ts_0_range_cached.start, 0) + thread_id;')
        else:
            if KernelCall.args.lb:
                c.add(f'int global_scan_end = ts_0_range_cached.end;')
                c.add(f'Themis::PullINodesAtZeroLvlDynamically<{spctxt.gctxt.num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
            else:
                c.add(f'Themis::PullINodesAtZeroLvlStatically(thread_id, ts_0_range_cached, inodes_cnts);')
            #if KernelCall.args.mode == 'stats':
            #    c.add('\tif (warp_id == 0 && thread_id == 0) atomicAdd(&global_stats_per_lvl[0].num_inodes, ts_0_range_cached.end - ts_0_range_cached.start);')
            
        
    def genScan(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)
        spctxt.declared.add(algExpr.tid.id)
        
        spctxt.addVar(AggHT(algExpr.opId, self.size, algExpr.groupAttributes, {}, aggAttrs=algExpr.aggregateTuplesCreated))

        if len(algExpr.avgAggregates) > 0:
            aggAttr, inputIdentifier, reductionType = list(algExpr.aggregateTuplesCreated.values())[-1]
            loc_cnt = Location.apayl(algExpr.opId, aggAttr, algExpr.tid)

        for attrId, attr in algExpr.outRelation.items():
            attr.tid = algExpr.tid
            if attrId in algExpr.groupAttributes:
                spctxt.setLoc(attrId, Location.aht(algExpr.opId, attr, algExpr.tid))
            else:
                inId, reductionType = algExpr.aggregateTuples[attrId]
                if reductionType == Reduction.AVG:
                    loc = Location.apayl(algExpr.opId, attr, algExpr.tid)    
                    spctxt.setLoc(attrId, Location.divide(loc, loc_cnt))
                else:
                    spctxt.setLoc(attrId, Location.apayl(algExpr.opId, attr, algExpr.tid))

            if KernelCall.args.use_pos_vec == False:
                spctxt.toReg(attr)
                spctxt.pctxt.materialized.add(attrId)
                spctxt.attrOriginLoc[attr.id] = Location.reg(attr)
                    
        spctxt.pctxt.scanSize = self.size

        c = spctxt.pctxt.precode
        c.add('int {} = scan_offset + threadIdx.x;'.format(algExpr.tid.id_name))        
        c.add('active = !isLast && ({} < {});'.format(algExpr.tid.id_name, self.size))
        
        c.add(f'if ((isLast && !keepGoing) || (!isLast && scan_offset >= {self.size})) break;')

    def genSingleScan(self, spctxt):

        algExpr = self.algExpr

        spctxt.setLoc(algExpr.tid.id, Location.reg(algExpr.tid))
        spctxt.pctxt.materialized.add(algExpr.tid.id)
        spctxt.declared.add(algExpr.tid.id)
        
        spctxt.addVar(AggHT(algExpr.opId, self.size, algExpr.groupAttributes, {}, aggAttrs=algExpr.aggregateTuplesCreated))

        if len(algExpr.avgAggregates) > 0:
            aggAttr, inputIdentifier, reductionType = list(algExpr.aggregateTuplesCreated.values())[-1]
            loc_cnt = Location.apayl(algExpr.opId, aggAttr, algExpr.tid)

        for attrId, attr in algExpr.outRelation.items():
            attr.tid = algExpr.tid
            if attrId in algExpr.groupAttributes:
                spctxt.setLoc(attrId, Location.aht(algExpr.opId, attr, algExpr.tid))
            else:
                inId, reductionType = algExpr.aggregateTuples[attrId]
                if reductionType == Reduction.AVG:
                    loc = Location.apayl(algExpr.opId, attr, algExpr.tid)    
                    spctxt.setLoc(attrId, Location.divide(loc, loc_cnt))
                else:
                    spctxt.setLoc(attrId, Location.apayl(algExpr.opId, attr, algExpr.tid))
                    
        spctxt.pctxt.scanSize = self.size

        c = spctxt.pctxt.precode
        c.add('if (thread_id != 0) return;')
        c.add('int {} = 0;'.format(algExpr.tid.id_name))
        c.add('for(; {} < {}; ++{})'.format(algExpr.tid.id_name, self.size, algExpr.tid.id_name) + '{')
        c.add('active = true;')

    def genMaterialize(self, spctxt):
        algExpr = self.algExpr

        spctxt.addVar(AggHT(algExpr.opId, self.size, algExpr.groupAttributes, {}, aggAttrs=algExpr.aggregateTuplesCreated))

        prec = spctxt.pctxt.precode
        postc = spctxt.pctxt.postcode
        cc = spctxt.activecode

        for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
            attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
            if reductionType != Reduction.COUNT: 
                spctxt.toReg(algExpr.inRelation[inId])

        groupKeySize = 0
        sizeDic = {
            Type.STRING : 16,
            Type.INT: 4,
            Type.FLOAT: 4,
            Type.DOUBLE: 8,
            Type.CHAR: 1,
            Type.DATE: 4,
            Type.BOOLEAN: 1,
            Type.ULL: 8,
            Type.PTR_INT: 8
        }
        for attrId, attr in algExpr.groupAttributes.items():
            groupKeySize += sizeDic[attr.dataType]


        if algExpr.doGroup:

            prec.add('int doMemoryFence = 0;')
            if KernelCall.args.system == 'Themis' and groupKeySize >= KernelCall.args.group_key_threshold and groupKeySize <= 64:
                prec.add('int coin = 0;')
                prec.add('int k = 16;')
                prec.add('unsigned long long old_num_memory_fences = 0;')
                prec.add('unsigned num_memory_fences = 0;')
                prec.add('unsigned long long old_clock_memory_fences = clock64();')

            cc.add('Payload{} buf_payl;'.format(algExpr.opId))    
            cc.add('uint64_t hash_key = 0;')
            for attrId, attr in algExpr.groupAttributes.items():
                loc_src = spctxt.toReg(attr)
                cc.add('buf_payl.{} = {};'.format(attr.id_name, loc_src))
                
            for attrId, attr in algExpr.groupAttributes.items():
                loc_src = spctxt.toReg(attr)
                if attr.dataType == Type.STRING:
                    cc.add('hash_key = hash(hash_key + stringHash({}));'.format(loc_src))
                else:
                    cc.add('hash_key = hash(hash_key + ((uint64_t) {}));'.format(loc_src))

            cc.add("\tint bucketFound = 0;")

            
            cc.add("int {} = 0;".format(algExpr.tid.id_name))
            cc.add("int numLookups = 0;")
            
            if KernelCall.args.system == 'Themis' and groupKeySize >= KernelCall.args.group_key_threshold and groupKeySize <= 64:
                cc.add('for (int i = 0; i < k; ++i) {')
                cc.add('if (thread_id % k != i) continue;')
            cc.add("\t\twhile (!bucketFound) {")

            cc.add('int location = -1;')
            cc.add('bool done = false;')
            cc.add('while (!done) {')
            cc.add('\tlocation = ( hash_key + numLookups) % {};'.format(int(self.size)))
            
            if KernelCall.args.system in ['pyper'] and self.size < KernelCall.args.agg_threshold:
                cc.add('\tagg_ht<Payload{}>& entry = shared_aht{}[location];'.format(algExpr.opId, algExpr.opId))
            else:
                cc.add('\tagg_ht<Payload{}>& entry = aht{}[location];'.format(algExpr.opId, algExpr.opId))
            cc.add('\tnumLookups++;')
            cc.add('\tif (entry.lock.enter()) {')
            cc.add('\t\tentry.payload = buf_payl;')
            cc.add('\t\tentry.hash = hash_key;')

            cc.add('\t\tentry.lock.done();')
            cc.add('\t\tbreak;')
            cc.add('\t} else {')
            cc.add('\t\tentry.lock.wait();')
            cc.add('\tdone = (entry.hash == hash_key);')
            cc.add('\t}')
            #cc.add('if ( numLookups == {})'.format(self.size) + '{')
            #cc.add('}')
            cc.add('}')
            cc.add('{} = location;'.format(algExpr.tid.id_name))
            #cc.add("\t\t\t{} = hashAggregateGetBucket(aht{},{},hash_key,numLookups,&buf_payl, doMemoryFence);".format(algExpr.tid.id_name, algExpr.opId, self.size))
            
            if KernelCall.args.system in ['pyper'] and self.size < KernelCall.args.agg_threshold:
                prefix = 'shared_'
            else:
                prefix = ''
            cc.add(f"Payload{algExpr.opId} entry = aht{algExpr.opId}[location].payload;")
            cc.add("\t\t\tbucketFound = 1;")
            for attrId, attr in algExpr.groupAttributes.items():
                loc_left = Location.payl(attr)
                loc_right = Location.aht(algExpr.opId, attr, algExpr.tid)
                if attr.dataType == Type.STRING:
                    cc.add("\t\t\tbucketFound &= stringEquals(buf_{}, {}entry.{});".format(loc_left, prefix, attr.id_name))
                else:
                    cc.add("\t\t\tbucketFound &= buf_{} == {}entry.{};".format(loc_left, prefix, attr.id_name))
            cc.add('\t\t\t}')
            
            if KernelCall.args.system == 'Themis' and groupKeySize >= KernelCall.args.group_key_threshold and groupKeySize <= 64:
                cc.add('}')

            if KernelCall.args.system == 'Themis' and groupKeySize >= KernelCall.args.group_key_threshold and groupKeySize <= 64:
                spctxt.postcode.add('num_memory_fences += __ballot_sync(ALL_LANES, doMemoryFence) != 0 ? 1 : 0;')
                spctxt.postcode.add('if (++coin > 4) {')
                spctxt.postcode.add('unsigned long long clock_memory_fences = clock64();')
                spctxt.postcode.add('unsigned long long new_num_memory_fences;')
                spctxt.postcode.add('if (thread_id == 0) new_num_memory_fences = atomicAdd(mf_cnt, num_memory_fences) + num_memory_fences;')
                spctxt.postcode.add('new_num_memory_fences = __shfl_sync(ALL_LANES, new_num_memory_fences, 0);')
                spctxt.postcode.add('double x = (double) (clock_memory_fences - old_clock_memory_fences);')
                spctxt.postcode.add('double y = (double) ((new_num_memory_fences - old_num_memory_fences) * 1024 * 1024);')
                spctxt.postcode.add('double dense = y / x;')
                spctxt.postcode.add('k = dense < 1000.0 ? 1 : 16;')
                spctxt.postcode.add('old_clock_memory_fences = clock_memory_fences;')
                spctxt.postcode.add('old_num_memory_fences = new_num_memory_fences;')
                spctxt.postcode.add('num_memory_fences = 0;')
                spctxt.postcode.add('coin = 0;')
                spctxt.postcode.add('}')            
            spctxt.postcode.add('doMemoryFence = 0;')
            

            
            #cc.add('if (active) {')
            for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                loc_dst = Location.apayl(algExpr.opId, attr, algExpr.tid)
                if reductionType == Reduction.COUNT:
                    cc.add("atomicAdd(&{}{},1);".format(prefix, loc_dst))
                else:
                    loc_src = spctxt.attrLoc[inId]
                    if reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                        cc.add("atomicAdd(&{}{},{});".format(prefix, loc_dst, loc_src))
                    elif reductionType == Reduction.MAX:
                        cc.add("atomicMax(&{}{},{});".format(prefix, loc_dst, loc_src))
                    elif reductionType == Reduction.MIN:
                        cc.add("atomicMin(&{}{},{});".format(prefix, loc_dst, loc_src))
            #cc.add('}')
            if KernelCall.args.system in ['pyper'] and self.size < KernelCall.args.agg_threshold:
                prec.add("__shared__ agg_ht<Payload{}> shared_aht{}[{}];".format(algExpr.opId, algExpr.opId, self.size))
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    dtype = langType(attr.dataType)


                    prec.add("__shared__ {} shared_aht{}_{}[{}];".format(dtype, algExpr.opId, attr.id_name, self.size))
                
                prec.add('for (int i = threadIdx.x; i < {}; i += {}) '.format(self.size, KernelCall.defaultBlockSize) + '{')
                prec.add('shared_aht{}[i].lock.init();'.format(algExpr.opId))
                prec.add('shared_aht{}[i].hash = HASH_EMPTY;'.format(algExpr.opId))
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    dtype = langType(attr.dataType)
                    if reductionType == Reduction.SUM or reductionType == Reduction.AVG or reductionType == Reduction.COUNT:
                        initValue = CType.zeroValue[dtype]
                    elif reductionType == Reduction.MAX:
                        initValue = CType.minValue[dtype]
                    elif reductionType == Reduction.MIN:
                        initValue = CType.maxValue[dtype]
                    else:
                        assert(False)            
                    prec.add('shared_aht{}_{}[i] = {};'.format(algExpr.opId, attr.id_name, initValue))
                prec.add('}')
                                
                prec.add('__syncthreads();')

                postc.add('{')
                postc.add('__syncthreads();')
                postc.add("Payload{} payl;".format(algExpr.opId))

                postc.add('for (int i = threadIdx.x; i < {}; i += {}) '.format(self.size, KernelCall.defaultBlockSize) + '{')
                postc.add('if (shared_aht{}[i].lock.lock != OnceLock::LOCK_DONE) continue;'.format(algExpr.opId))
                postc.add('uint64_t hash_key = 0;')
                for attrId, attr in algExpr.groupAttributes.items():
                    if attr.dataType == Type.STRING:
                        postc.add('hash_key = hash(hash_key + stringHash(shared_aht{}[i].payload.{}));'.format(algExpr.opId, attr.id_name))
                    else:
                        postc.add('hash_key = hash(hash_key + ((uint64_t) shared_aht{}[i].payload.{}));'.format(algExpr.opId, attr.id_name))
                    postc.add("payl.{} = shared_aht{}[i].payload.{};".format(attr.id_name, algExpr.opId, attr.id_name))
                
                postc.add("int bucketFound = 0;")
                postc.add("int numLookups = 0;")
                postc.add('int bucket;')
                postc.add("while (!bucketFound) {")
                postc.add("bucket = hashAggregateGetBucket(aht{},{},hash_key,numLookups,&payl);".format(algExpr.opId, self.size))
                postc.add("bucketFound = 1;")

                for attrId, attr in algExpr.groupAttributes.items():
                    if attr.dataType == Type.STRING:
                        postc.add("bucketFound &= stringEquals(shared_aht{}[i].payload.{}, payl.{});".format(algExpr.opId, attr.id_name, attr.id_name))
                    else:
                        postc.add("bucketFound &= shared_aht{}[i].payload.{} == payl.{};".format(algExpr.opId, attr.id_name, attr.id_name))

                postc.add("}")

                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    if reductionType == Reduction.SUM or reductionType == Reduction.AVG or reductionType == Reduction.COUNT:
                        postc.add("\t\tatomicAdd(&aht{}_{}[bucket],shared_aht{}_{}[i]);".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                    elif reductionType == Reduction.MAX:
                        postc.add("\t\tatomicMax(&aht{}_{}[bucket],shared_aht{}_{}[i];".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                    elif reductionType == Reduction.MIN:
                        postc.add("\t\tatomicMin(&aht{}_{}[bucket],shared_aht{}_{}[i];".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                
                postc.add('}')
                postc.add('}')

            
            if KernelCall.args.system == 'Themis' and self.size < KernelCall.args.agg_threshold:
                postc.add('{')
                prec.add("\taht{} += (warp_id+1) * {};".format(algExpr.opId, self.size))
                postc.add("\taht{} -= (warp_id+1) * {};".format(algExpr.opId, self.size))
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    prec.add('\taht{}_{} += (warp_id+1) * {};'.format(algExpr.opId, attr.id_name, self.size))
                    postc.add('\taht{}_{} -= (warp_id+1) * {};'.format(algExpr.opId, attr.id_name, self.size))

                
                postc.add("\tPayload{} payl;".format(algExpr.opId))

                postc.add('\tfor (int t = thread_id; t < {}; t += 32) '.format(self.size) + '{')
                postc.add('\t\tint i = t + (warp_id+1) * {};'.format(self.size))
                postc.add('\t\tif (aht{}[i].lock.lock != OnceLock::LOCK_DONE) continue;'.format(algExpr.opId))
                
                postc.add('\t\tuint64_t hash_key = 0;')
                for attrId, attr in algExpr.groupAttributes.items():
                    if attr.dataType == Type.STRING:
                        postc.add('\t\thash_key = hash(hash_key + stringHash(aht{}[i].payload.{}));'.format(algExpr.opId, attr.id_name))
                    else:
                        postc.add('\t\thash_key = hash(hash_key + ((uint64_t) aht{}[i].payload.{}));'.format(algExpr.opId, attr.id_name))
                    postc.add("\t\tpayl.{} = aht{}[i].payload.{};".format(attr.id_name, algExpr.opId, attr.id_name))
                    
                postc.add("\t\tint bucketFound = 0;")
                postc.add("\t\tint numLookups = 0;")
                postc.add('\t\tint bucket;')
                postc.add("\t\twhile (!bucketFound) {")
                postc.add("\t\t\tbucket = hashAggregateGetBucket(aht{},{},hash_key,numLookups,&payl);".format(algExpr.opId, self.size))
                postc.add("\t\t\tbucketFound = 1;")

                for attrId, attr in algExpr.groupAttributes.items():
                    if attr.dataType == Type.STRING:
                        postc.add("\t\t\tbucketFound &= stringEquals(aht{}[i].payload.{}, payl.{});".format(algExpr.opId, attr.id_name, attr.id_name))
                    else:
                        postc.add("\t\t\tbucketFound &= aht{}[i].payload.{} == payl.{};".format(algExpr.opId, attr.id_name, attr.id_name))

                postc.add("\t\t}")

                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    if reductionType == Reduction.SUM or reductionType == Reduction.AVG or reductionType == Reduction.COUNT:
                        postc.add("\t\tatomicAdd(&aht{}_{}[bucket],aht{}_{}[i]);".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                    elif reductionType == Reduction.MAX:
                        postc.add("\t\tatomicMax(&aht{}_{}[bucket],aht{}_{}[i];".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                    elif reductionType == Reduction.MIN:
                        postc.add("\t\tatomicMin(&aht{}_{}[bucket],aht{}_{}[i];".format(algExpr.opId, attr.id_name, algExpr.opId, attr.id_name))
                postc.add('\t}')
                postc.add('}')
            
        else:
            if KernelCall.args.local_agg and KernelCall.args.system in ['Themis']:
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT:
                        prec.add('{} {} = {};'.format(dType, local_loc_dst, CType.zeroValue[dType]))
                        cc.add("{} += 1;".format(local_loc_dst))
                    else:
                        loc_src = spctxt.attrLoc[inId]
                        if reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                            prec.add('{} {} = {};'.format(dType, local_loc_dst, CType.zeroValue[dType]))
                            cc.add("{} += {};".format(local_loc_dst, loc_src))
                        elif reductionType == Reduction.MAX:
                            prec.add('{} {} = {};'.format(dType, local_loc_dst, CType.minValue[dType]))
                            cc.add("{} = {} > {} ? {} : {};".format(local_loc_dst, loc_src, local_loc_dst, loc_src, local_loc_dst))
                        elif reductionType == Reduction.MIN:
                            prec.add('{} {} = {};'.format(dType, local_loc_dst, CType.maxValue[dType]))
                            cc.add("local_{} = {} < {} ? {} : {};".format(local_loc_dst, loc_src, local_loc_dst, loc_src, local_loc_dst))
                
                num_warps_per_block = int(KernelCall.defaultBlockSize / 32)
                
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType) # 
                    postc.add(f"__shared__ {dType} shared_{local_loc_dst}[{num_warps_per_block}];")
                
                postc.add("for (int offset = 16; offset > 0; offset /= 2) {")
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                        postc.add(f"{local_loc_dst} += __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")                        
                    elif reductionType == Reduction.MAX:
                        postc.add("{")
                        postc.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                        postc.add(f"{local_loc_dst} = {local_loc_dst} < v ? v : {local_loc_dst};")
                        postc.add("}")
                    elif reductionType == Reduction.MIN:
                        postc.add("{")
                        postc.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                        postc.add(f"{local_loc_dst} = {local_loc_dst} > v ? v : {local_loc_dst};")
                        postc.add("}")
                postc.add("}")                
                postc.add("if (thread_id == 0) {")
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    postc.add(f"shared_{local_loc_dst}[warp_id % {num_warps_per_block}] = {local_loc_dst};")
                postc.add("}")
                postc.add("__syncthreads();")
                postc.add(f"if (warp_id % {num_warps_per_block} == 0) ")
                postc.add("{")
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                        postc.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.zeroValue[dType]};")                     
                    elif reductionType == Reduction.MAX:
                        postc.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.minValue[dType]};")
                    elif reductionType == Reduction.MIN:
                        postc.add(f"{local_loc_dst} = thread_id < {num_warps_per_block} ? shared_{local_loc_dst}[thread_id] : {CType.maxValue[dType]};")
                postc.add(f"for (int offset = {int(num_warps_per_block/2)}; offset > 0; offset /= 2)")
                postc.add("{")
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                        postc.add(f"{local_loc_dst} += __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")                        
                    elif reductionType == Reduction.MAX:
                        postc.add("{")
                        postc.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                        postc.add(f"{local_loc_dst} = {local_loc_dst} < v ? v : {local_loc_dst}")
                        postc.add("}")
                    elif reductionType == Reduction.MIN:
                        postc.add("{")
                        postc.add(f"{dType} v = __shfl_down_sync(0xffffffff, {local_loc_dst}, offset);")
                        postc.add(f"{local_loc_dst} = {local_loc_dst} > v ? v : {local_loc_dst}")
                        postc.add("}")
                postc.add("}")
                postc.add("}")
                
                postc.add("if (threadIdx.x == 0) {")
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    local_loc_dst = 'local_{}'.format(attr.id_name)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT or reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                        postc.add(f"atomicAdd(&{loc_dst},{local_loc_dst});")
                    elif reductionType == Reduction.MAX:
                        postc.add(f"atomicMax(&{loc_dst},{local_loc_dst});")
                    elif reductionType == Reduction.MIN:
                        postc.add(f"atomicMin(&{loc_dst},{local_loc_dst});")
                postc.add("}")
            else:
                for attrId, (inId, reductionType) in algExpr.aggregateTuples.items():
                    attr, inputIdentifier, reductionType = algExpr.aggregateTuplesCreated[attrId]
                    loc_dst = Location.apayl(algExpr.opId, attr, None)
                    dType = langType(attr.dataType)
                    if reductionType == Reduction.COUNT:
                        cc.add("atomicAdd(&{},1);".format(loc_dst))
                    else:
                        loc_src = spctxt.attrLoc[inId]
                        if reductionType == Reduction.SUM or reductionType == Reduction.AVG:
                            cc.add("atomicAdd(&{},{});".format(loc_dst, loc_src))
                        elif reductionType == Reduction.MAX:
                            cc.add("atomicMax(&{},{});".format(loc_dst, loc_src))
                        elif reductionType == Reduction.MIN:
                            cc.add("atomicMin(&{},{});".format(loc_dst, loc_src))

    def genOperation(self, spctxt):
        algExpr = self.algExpr
        cc = spctxt.activecode
        if algExpr.doGroup:
            cc.add("active = aht{}[{}].lock.lock == OnceLock::LOCK_DONE;"
                .format(algExpr.opId, algExpr.tid.id_name))
        #cc.add("active &= aht{}[tid{}].hash != HASH_EMPTY;"
        #    .format(algExpr.opId, algExpr.opId))

    def genPushTrapzoids(self, spctxt):
        spctxt.ts_type = 2
        super().genPushTrapzoids(spctxt)

    def doExpansion(self):
        return False


class Materialize(PhysicalOperator):

    def __init__(self, op):
        super().__init__(op)

    def genMaterialize(self, spctxt):

        tableName = "result" if self.algExpr.isResult else self.algExpr.table['name'] 

        spctxt.addVar(Table(tableName, self.algExpr.tupleNum, self.algExpr.outRelation))

        c = spctxt.precode
        
        c.add("int wp, writeMask, numProj;")
        c.add("writeMask = __ballot_sync(ALL_LANES, active);")
        c.add("numProj = __popc(writeMask);")
            
        if KernelCall.args.system == 'Pyper' and not self.algExpr.isResult:
            c.add("if (thread_id == 0 && numProj > 0) { ")
            c.add('\twp = atomicAdd(nout_{}, 32);'.format(tableName))
            c.add("}")
            c.add("wp = __shfl_sync(ALL_LANES, wp, 0) + thread_id;")
        else:
            
            c.add("if (thread_id == 0) { ")
            c.add("\twp = atomicAdd(nout_{}, numProj);".format(tableName))
            c.add("}")
            c.add("wp = __shfl_sync(ALL_LANES, wp, 0);")
            c.add("wp = wp + __popc(writeMask & prefixlanes);")

        if tableName == "result":
            pc = spctxt.gctxt.printcode
            pc.add('std::clock_t start_copyTime = std::clock();')
            pc.add('cudaMemcpy(&cpu_nout_result, nout_result, sizeof(int), cudaMemcpyDeviceToHost);')

            for attrId, attr in self.algExpr.outRelation.items():
                #if attr.dataType == Type.STRING: 
                #    continue
                pc.add('std::vector<{}>  cpu_result_{}(cpu_nout_result);'
                    .format(langType(attr.dataType), attr.id_name))
                pc.add('cudaMemcpy(cpu_result_{}.data(), result_{}, cpu_nout_result * sizeof({}), cudaMemcpyDeviceToHost);'
                    .format(attr.id_name, attr.id_name, langType(attr.dataType)))
            pc.add('std::clock_t stop_copyTime = std::clock();')
            pc.add('printf ( "%32s: %6.1f ms\\n", "copyTime", (stop_copyTime - start_copyTime) / (double) (CLOCKS_PER_SEC / 1000) );')
            pc.add("for (int pv = 0; pv < 5 && pv < cpu_nout_result; ++pv) {")

            for attrId, attr in self.algExpr.outRelation.items():
                pc.add('\tprintf("{}: ");'.format(attr.id_name))
                if attr.dataType == Type.STRING:
                    pass
                    #pc.add('\tstringPrint(cpu_{}_attr{}_{}[pv]);'.format(tableName, attr.id, attr.name))
                else:
                    pc.add('\tprintf("{}", cpu_{}_{}[pv]);'.format(CType.printFormat[langType(attr.dataType)],tableName, attr.id_name))
                pc.add('\tprintf("\t");')

            pc.add('\tprintf("\\n");')
            pc.add("}")
        
        cc = spctxt.activecode
        if tableName != 'result' and KernelCall.args.system == 'Pyper':
            cc.add('select_{}[wp] = 1;'.format(tableName))
        for attrId, attr in self.algExpr.outRelation.items():
            #print(self.algExpr.opType, attr.tableName, attr.id, attr.name, attr.creation)
            #print(spctxt.attrLoc[attrId])
            loc_src = spctxt.toReg(attr)
            #cc.add("{}_attr{}_{}[wp] = {};".format(tableName, attr.id, attr.name, loc_src))
            if tableName == 'result':
                cc.add("{}_{}[wp] = {};".format(tableName, attr.id_name, loc_src))
            else:
                
                cc.add("{}_{}[wp] = {};".format(tableName, attr.name, loc_src))


        if KernelCall.args.system != 'pyper' and tableName != 'result':
            c = spctxt.pctxt.precode_outer_cpu
            c.add(f'int cpu_nout_{tableName} = 0;')
            c = spctxt.pctxt.postcode_cpu
            c.add(f'cudaMemcpy(&cpu_nout_{tableName}, nout_{tableName}, sizeof(int), cudaMemcpyDeviceToHost);')
            c.add('cudaDeviceSynchronize();')
            spctxt.gctxt.addVar(SingleVar('int', f'cpu_nout_{tableName}'))


class GlobalContext:

    def __init__(self, num_warps):
        self.num_warps = num_warps
        self.ts_width = KernelCall.args.ts_width
        self.vars = {}
        self.payloads = {}

        self.maincode = Code()
        self.krnlexeccode = Code()
        self.precode = Code()
        self.printcode = Code()

        self.pctxts = []


    def resolveTrapezoidStacks(self, c):
        max_ts_speculated_size_attributes = 0
        max_ts_speculated_num_ranges = 0
        
        for pctxt in self.pctxts:
            spctxt = pctxt.spctxts[-1]
            ts_speculated_size_attributes = spctxt.ts_speculated_size_attributes
            ts_speculated_num_ranges = spctxt.ts_speculated_num_ranges

            max_ts_speculated_size_attributes = max(max_ts_speculated_size_attributes, ts_speculated_size_attributes)
            max_ts_speculated_num_ranges = max(max_ts_speculated_num_ranges, ts_speculated_num_ranges)

        self.max_ts_speculated_num_ranges = max_ts_speculated_num_ranges
        if max_ts_speculated_size_attributes > 0:
            self.max_ts_speculated_size_attributes = (int((max_ts_speculated_size_attributes - 1) / 128) + 1) * 128
        else:
            self.max_ts_speculated_size_attributes = 0

        self.ts = TrapezoidStacks(self.ts_width, max_ts_speculated_num_ranges, max_ts_speculated_size_attributes)
        self.ts.declare(self)

        c.add('#define SIZE_TS_PER_WARP {}'.format(self.ts.size_per_warp))
        c.add('#define SIZE_TS_RANGES_PER_WARP {}'.format(self.max_ts_speculated_num_ranges * 8 * self.ts_width))

    def resolvePushedParts(self, c):
        max_ts_size_attributes = 0
        max_accumulated_ts_size_attributes = 0
        max_height = 0
        for pctxt in self.pctxts:
            accumulated_ts_size_attributes = 0
            for spctxt in pctxt.spctxts:
                ts_size_attributes = spctxt.ts_size_attributes
                max_ts_size_attributes = max(max_ts_size_attributes, ts_size_attributes)
                accumulated_ts_size_attributes += ts_size_attributes
            max_height = max(max_height, len(pctxt.spctxts))
            max_accumulated_ts_size_attributes = max(max_accumulated_ts_size_attributes, accumulated_ts_size_attributes)

        self.max_ts_size_attributes = 0 if max_ts_size_attributes == 0 else (int((max_ts_size_attributes-1)/128) + 1)  * 128
        self.max_accumulated_ts_size_attributes = 0 if max_accumulated_ts_size_attributes == 0 else (int((max_accumulated_ts_size_attributes-1)/128) + 1)  * 128
        self.max_height = max_height
        self.gts = PushedParts(self.ts_width, self.max_height, self.max_ts_size_attributes, self.max_accumulated_ts_size_attributes)
        self.gts.declare(self)
        c.add(f'#define SIZE_GTS_PER_WARP {self.max_accumulated_ts_size_attributes}')
        #c.add('#define SIZE_GTS_RANGES_PER_WARP {}'.format(self.max_ts_num_ranges * 8 * self.ts_width))



    def genIncludes(self, c):
        c.add('#include <chrono>')
        c.add('#include <list>')
        c.add('#include <unordered_map>')
        c.add('#include <vector>')
        c.add('#include <iostream>')
        c.add('#include <ctime>')
        c.add('#include <limits.h>')
        c.add('#include <float.h>')
        c.add('#include "dogqc/include/csv.h"')
        c.add('#include "dogqc/include/mappedmalloc.h"')
        c.add('#include "themis/include/themis.cuh"')
        c.add('#include "themis/include/worksharing.cuh"')

    def genInitProfileCounters(self, c, s):
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                size = int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32) if s not in KernelCall.args.num_counters else KernelCall.args.num_counters[counter]    
                c.add('\tunsigned long long* {}s = NULL;'.format(counter))
                c.add('\tcudaMalloc((void**)&{}s, sizeof(unsigned long long) * {});'.format(counter, size))
                if counter == 'min_clock':
                    c.add('\tinitArray<<<{},{}>>>({}s, 0xFFFFFFFFFFFFFFFF, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter, size))
                else:
                    c.add('\tinitArray<<<{},{}>>>({}s, 0, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter, size))
            c.add('\tstd::vector<unsigned long long> cpu_counters({});'.format(size))
            
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                size = int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)
                c.add('\tunsigned long long* global_{} = NULL;'.format(counter))
                c.add('\tcudaMalloc((void**) &global_{}, sizeof(unsigned long long) * {});'.format(counter, size))
                c.add('\tinitArray<<<{},{}>>>(global_{}, 0, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter,size))
            c.add('\tstd::vector<unsigned long long> cpu_counters({});'.format(size))            
            

    def genInitSampling(self, c):
        if KernelCall.args.mode == 'sample':
            c.add('\tunsigned long long* samples = NULL;')
            c.add('\tcudaMalloc((void**)&samples, sizeof(unsigned long long) * WIDTH_SAMPLES * {});'.format(self.num_warps))
            c.add('\tstd::vector<unsigned long long> cpu_samples(WIDTH_SAMPLES * {});'.format(self.num_warps))
            c.add('\tunsigned long long* sample_start;')
            c.add('\tcudaMalloc((void**)&sample_start, sizeof(unsigned long long));')

    def genError(self, c, msg_loc):
        c.add('{')
        c.add('\tcudaError err = cudaGetLastError();')
        c.add('\tif(err != cudaSuccess) {')
        c.add('\t\tstd::cerr << "Cuda Error in {} " << cudaGetErrorString( err ) << std::endl;'.format(msg_loc))
        c.add('\t\tERROR("{}")'.format(msg_loc))
        c.add('\t}')
        c.add('}')


    def genTimer(self, c):
        c.add(f'std::chrono::steady_clock::time_point start_timepoint, end_timepoint;')
        for pid, pctxt in enumerate(self.pctxts):
            c.add(f'std::chrono::steady_clock::time_point start_timepoint_{pid}, end_timepoint_{pid};')


    def printElapsedTimes(self, c):
        for pid, pctxt in enumerate(self.pctxts):
            c.add(f'\tprintf("%32s: %6.1f ms\\n", "KernelTime{pid}",((double) std::chrono::duration_cast<std::chrono::microseconds>(end_timepoint_{pid} - start_timepoint_{pid}).count()) / 1000);')
            
        c.add(f'\tprintf("%32s: %6.1f ms\\n", "totalKernelTime0",((double) std::chrono::duration_cast<std::chrono::microseconds>(end_timepoint - start_timepoint).count()) / 1000);')


    def toSingleCode(self):
        c = Code()
        self.genIncludes(c)
        for v in self.vars.values():
            v.declare(self)
        c.addAll(self.precode.lines)
        for pctxt in self.pctxts:
            pc = pctxt.toSingleString()
            c.addAll(pc.lines)

        c.add('int main() {')
        #c.add('cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);')
        
        self.genTimer(c)
        
        c.add('\tint cpu_cnts[32];')
        c.add('\tint* cnt = NULL;')
        c.add('\tcudaMalloc((void**)&cnt, 32 * sizeof(int));')
        #c.add('\tunsigned long long* mf_cnt = NULL;')
        #c.add('\tcudaMalloc((void**)&mf_cnt, sizeof(unsigned long long));')

        c.addAll(self.maincode.lines)

        self.genError(c, 'allocation')
        
        c.add('\tstd::clock_t start_totalKernelTime0 = std::clock();')
        c.addAll(self.krnlexeccode.lines)
        c.add('\tstd::clock_t stop_totalKernelTime0 = std::clock();')

        for pid, pctxt in enumerate(self.pctxts):
            c.add(f'\tprintf ( "%32s: %6.1f ms\\n", "KernelTime{pid}", (stop_kernelTime{pid} - start_kernelTime{pid}) / (double) (CLOCKS_PER_SEC / 1000) );')
            #c.add('\tprintf ( "%32s: %6.1f ms\\n", "KernelTime{}", elapsedTime{} );'.format(pid, pid))
        #c.ad
        c.add('\tprintf ( "%32s: %6.1f ms\\n", "totalKernelTime", (stop_totalKernelTime0 - start_totalKernelTime0) / (double) (CLOCKS_PER_SEC / 1000) );')

        c.add('\tint cpu_nout_result;')
        c.add('\tcudaMemcpy(&cpu_nout_result, nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);')
        c.add('\tprintf("Result: %d tuples\\n", cpu_nout_result);')

        c.addAll(list(map(lambda x: '\t' + x, self.printcode.lines)))
        c.add("}")

        return c

    def toPyperCode(self):
        c = Code()
        self.genIncludes(c)


        c.add('#ifdef MODE_SAMPLE')
        c.add('#define TYPE_SAMPLE_START 1')
        c.add('#define TYPE_SAMPLE_ACTIVE 2')
        c.add('#define TYPE_SAMPLE_PUSHING 3')
        c.add('#define TYPE_SAMPLE_PULLING 4')
        c.add('#define TYPE_SAMPLE_END 5')
        c.add('#endif')


        for v in self.vars.values():
            v.declare(self)
        c.addAll(self.precode.lines)
        for pctxt in self.pctxts:
            pc = pctxt.toPyperString()
            c.addAll(pc.lines)

        c.add('int main() {')
        c.add('\tint cpu_cnts[32];')
        c.add('\tint* cnt = NULL;')
        c.add('\tcudaMalloc((void**)&cnt, 32 * sizeof(int));')
        #c.add('\tunsigned long long* mf_cnt = NULL;')
        #c.add('\tcudaMalloc((void**)&mf_cnt, sizeof(unsigned long long));')

        c.add('#ifdef MODE_SAMPLE')
        c.add('\tunsigned long long* sample_start;')
        c.add('\tcudaMalloc((void**)&sample_start, sizeof(unsigned long long));')
        
        c.add('#endif')
        c.addAll(self.maincode.lines)

        self.genError(c, 'allocation')
        
        c.add("cudaDeviceSynchronize();")
        c.add(f"cudaEvent_t start_totalKernelTime0, stop_totalKernelTime0;")
        c.add(f"float totalElapsedTime0;")
        c.add(f"cudaEventCreate(&start_totalKernelTime0);")
        c.add(f"cudaEventCreate(&stop_totalKernelTime0);")
        c.add(f"cudaEventRecord(start_totalKernelTime0);")
        
        c.addAll(self.krnlexeccode.lines)
        c.add(f"cudaEventRecord(stop_totalKernelTime0);")
        c.add(f"cudaEventSynchronize(stop_totalKernelTime0);")
        c.add(f"cudaEventElapsedTime(&totalElapsedTime0, start_totalKernelTime0, stop_totalKernelTime0);")
        for pid, pctxt in enumerate(self.pctxts):
            c.add('\tprintf ( "%32s: %6.1f ms\\n", "KernelTime{}", elapsedTime{} );'.format(pid, pid))
        c.add('\tprintf ( "%32s: %6.1f ms\\n", "totalKernelTime0", totalElapsedTime0 );')

        c.add('\tint cpu_nout_result;')
        c.add('\tcudaMemcpy(&cpu_nout_result, nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);')
        c.add('\tprintf("Result: %d tuples\\n", cpu_nout_result);')

        c.addAll(list(map(lambda x: '\t' + x, self.printcode.lines)))
        c.add("}")

        return c


    def toCode(self):
        c = Code()
        self.genIncludes(c)
        if KernelCall.args.mode == 'sample':
            c.add('#define WIDTH_SAMPLES (1 << 17)')
            c.add('#define TYPE_SAMPLE_START 0')
            c.add('#define TYPE_SAMPLE_ACTIVE 1')
            c.add('#define TYPE_SAMPLE_DETECTING 2')
            c.add('#define TYPE_SAMPLE_TRY_PUSHING 3')
            c.add('#define TYPE_SAMPLE_PUSHING 4')
            c.add('#define TYPE_SAMPLE_PULLING 5')
            c.add('#define TYPE_SAMPLE_END 6')

        if KernelCall.args.mode == 'stats':
            c.add('#define TYPE_STATS_PROCESSING 0')
            c.add('#define TYPE_STATS_PUSHING 1')
            c.add('#define TYPE_STATS_WAITING 2')
            c.add('#define TYPE_STATS_NUM_IDLE 3')
            c.add('#define TYPE_STATS_NUM_PUSHED 4')

        

        for v in self.vars.values():
            v.declare(self)
            
        
        
        self.resolveTrapezoidStacks(c)
        if KernelCall.args.lb:
            self.resolvePushedParts(c)

        c.addAll(self.precode.lines)

        for pctxt in self.pctxts:
            pc = pctxt.toString()
            c.addAll(pc.lines)

        c.add("int main() {")
        c.add('cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);')
        #c.add('\tunsigned long long* mf_cnt = NULL;')
        #c.add('\tcudaMalloc((void**)&mf_cnt, sizeof(unsigned long long));')
        self.genInitProfileCounters(c, self.num_warps)
        self.genInitSampling(c)

        
            
        c.addAll(self.maincode.lines)

        c.add('\tint cpu_cnts[32];')
        c.add('\tint* cnt = NULL;')
        c.add('\tcudaMalloc((void**)&cnt, 32 * sizeof(int));')
        if KernelCall.args.lb:
            c.add('Themis::StatisticsPerLvl* global_stats_per_lvl = NULL;')
            c.add(f'Themis::InitStatisticsPerLvl(global_stats_per_lvl, {self.num_warps});')
        self.genError(c, 'allocation')
        
        self.genTimer(c)
        
        c.add("cudaDeviceSynchronize();")
        c.add(f"cudaEvent_t start_totalKernelTime0, stop_totalKernelTime0;")
        c.add(f"float totalElapsedTime0;")
        c.add(f"cudaEventCreate(&start_totalKernelTime0);")
        c.add(f"cudaEventCreate(&stop_totalKernelTime0);")
        c.add(f"cudaEventRecord(start_totalKernelTime0);")
        
        c.addAll(self.krnlexeccode.lines)
        
        c.add(f"cudaEventRecord(stop_totalKernelTime0);")
        c.add(f"cudaEventSynchronize(stop_totalKernelTime0);")
        c.add(f"cudaEventElapsedTime(&totalElapsedTime0, start_totalKernelTime0, stop_totalKernelTime0);")
        for pid, pctxt in enumerate(self.pctxts):
            c.add('\tprintf ( "%32s: %6.1f ms\\n", "KernelTime{}", elapsedTime{} );'.format(pid, pid))
        c.add('\tprintf ( "%32s: %6.1f ms\\n", "totalKernelTime0", totalElapsedTime0 );')
        
        
        #self.printElapsedTimes(c)
        
        
        
        c.add('\tint cpu_nout_result;')
        c.add('\tcudaMemcpy(&cpu_nout_result, nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);')
        c.add('\tprintf("Result: %d tuples\\n", cpu_nout_result);')
        
        

        c.addAll(list(map(lambda x: '\t' + x, self.printcode.lines)))
        c.add("}")

        #print(c.toString())
        return c

    def update(self, pctxt):
        self.pctxts.append(pctxt)

    def addVar(self, ds):
        self.vars[ds.name] = ds


class PipeContext:

    def __init__(self, pid, gctxt):
        self.gctxt = gctxt
        self.id = pid
        self.attrLoc = {}
        self.attrOriginLoc = {}
        self.stringConstantsDic = {}
        self.inReg = set([])
        self.declared = set([])
        self.materialized = set([])

        self.ts_type = 0
        self.ts_num_ranges = 0
        self.ts_speculated_num_ranges = 0
        self.ts_list_attributes = []
        self.ts_size_attributes = 0        
        self.ts_speculated_size_attributes = 0
        self.ts_tid = None
        self.ts_tid_build = None
        
        self.spctxts = []
        self.vars = []

        self.defcode = Code()
        self.switchcode = Code()
        self.selectparameter = Code()
        self.precode = Code()
        self.postcode = Code()

        self.precode_outer_cpu = Code()
        self.precode_cpu = Code()
        self.postcode_cpu = Code()


        self.postexeccode = Code()
        self.pushcode = Code()
        self.pullcode = Code()
        
        self.num_loops = 0


    def genCode(self):
        pass

    def update(self, spctxt):
        self.spctxts.append(spctxt)
        self.attrLoc.update(spctxt.attrLoc)
        pass

    def stringConstants(self, token):
        if token not in self.stringConstantsDic:
            self.stringConstantsDic[token] = len(self.stringConstantsDic)
        return "string_constant{}".format(self.stringConstantsDic[token])


    def genInitWarp(self, c):
        
        if KernelCall.args.system == 'Themis':
            c.add(f'__shared__ int active_thread_ids[{KernelCall.defaultBlockSize}];')
        
        if KernelCall.args.lb:
            if KernelCall.args.lb_mode == 'morsel':
                pass
            else:
                c.add(f'if (blockIdx.x > {int(1024 / KernelCall.defaultBlockSize) * 82}) return;')
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
                
        c.add("int thread_id = threadIdx.x % 32;")        
        c.add("int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;")
        c.add("unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));")


    def genInitTrapezoidStack(self, c):
        c.add('int inodes_cnts = 0;')
        if KernelCall.args.mode == 'stats':
            c.add('unsigned long long accumulated_inodes_cnts = 0;')
            c.add('int max_inodes_cnts = 0;')

        if KernelCall.args.lb:
            if KernelCall.args.lb_type == 'ws': # work sharing
                c.add('int locally_lowest_lvl = 0;')
                c.add('int local_scan_offset = 0;')
            else:
                if KernelCall.args.lb_mode == 'morsel':
                    c.add('int input_table_size;')
                else:
                    c.add('int local_scan_offset = 0;')
                    pass
        c.add('Themis::Range ts_0_range_cached;')


    def genInitStringConstants(self, c):
        for token, token_id in self.stringConstantsDic.items():
            c.add('str_t string_constant{} = stringConstant("{}",{});'
                .format(token_id, token, len(token)))


    def genInitProfileCountersInKernel(self, c):
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                c.add('unsigned long long {} = 0;'.format(counter))
                
        if KernelCall.args.mode == 'stats':
            c.add(f'unsigned long long stat_counters[{len(stat_counters)}];')
            for i, counter in enumerate(stat_counters):
                c.add(f'stat_counters[{i}] = 0;')
            c.add(f"int current_status = -1;")
            c.add(f"unsigned long long tp = 0;")



    def genInitProfileCounters(self, c, s):
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                size = s
                c.add('\tunsigned long long* {}s = NULL;'.format(counter))
                c.add('\tcudaMalloc((void**)&{}s, sizeof(unsigned long long) * {});'.format(counter, size))
                if counter == 'min_clock':
                    c.add('\tinitArray<<<{},{}>>>({}s, 0xFFFFFFFFFFFFFFFF, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter, size))
                else:
                    c.add('\tinitArray<<<{},{}>>>({}s, 0, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter, size))
            c.add('\tstd::vector<unsigned long long> cpu_counters({});'.format(size))
            
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                size = s
                c.add('\tunsigned long long* global_{} = NULL;'.format(counter))
                c.add('\tcudaMalloc((void**)&global_{}, sizeof(unsigned long long) * {});'.format(counter, size))
                c.add('\tinitArray<<<{},{}>>>(global_{}, 0, {});'.format(KernelCall.defaultGridSize,KernelCall.defaultBlockSize,counter, size))


    def genInitSampling(self, c):
        if KernelCall.args.mode == 'sample':
            pass
            #c.add('unsigned long long sampling_start = clock64();')
            #c.add('samples += warp_id * WIDTH_SAMPLES;')
            #c.add('if (thread_id == 0) { samples[0] = 0; }')


    def genWorksharingPullingCode(self, c):
        if KernelCall.args.mode == 'sample':
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_PULLING);')
        c.add('unsigned int num_idle_warps = 0;')
        c.add('int src_warp_id = -1;')
        c.add(f'bool is_successful = Themis::PullINodesAtZeroLvlDynamically<{self.gctxt.num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
        c.add('if (is_successful) {')
        c.add('lvl = 0;')
        c.add('loop = 0;')
        c.add('} else {')
        c.add('int start = 0;')
        c.add('int end = 0;')
        c.add(f'char* attr = WorkSharing::Wait(thread_id, lvl, start, end, {KernelCall.args.ws_rs}, taskbook, taskstack, global_num_idle_warps, gpart_ids);')
        c.add('if (lvl == -2) {')
        if KernelCall.args.mode == 'stats':
            c.add(f"stat_counters[TYPE_STATS_WAITING] += (clock64() - tp);")
        c.add('break;')
        c.add('}')
        
        if len(self.spctxts) > 0:
            c.add(f'if ((end - start) >= 32) mask_32 = 0x1 << lvl;')
            c.add(f'mask_1 = 0x1 << lvl;')

            c.add('switch (lvl) {')
            for i, spctxt in enumerate(self.spctxts):
                if i == 0: continue
                if spctxt.prev.ts_num_ranges == 1:
                    c.add('case {}:'.format(i) + '{')
                    c.add(f'if (thread_id == {i}) inodes_cnts = (end - start);')
                    c.add(f'ts_{i}_range_cached.start = start + thread_id;')
                    c.add(f'ts_{i}_range_cached.end = end;')
                    speculated_size_attributes = 0
                    for attr in spctxt.prev.ts_list_attributes:
                        if attr.dataType == Type.STRING:
                            c.add(f'ts_{spctxt.id}_attr_{attr.id_name}_cached.start =  *((char**) attr);')
                            c.add(f'ts_{spctxt.id}_attr_{attr.id_name}_cached.end =  *((char**) (attr + 8));')
                        else:
                            c.add(f'ts_{spctxt.id}_attr_{attr.id_name}_cached = *(({langType(attr.dataType)}*) attr);')
                            
                        c.add(f'attr += {CType.size[langType(attr.dataType)]};')                    
                    c.add('} break;')
            c.add('}')

        c.add('}')

    def genPullingCode(self, c):
        if KernelCall.args.mode == 'sample':
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_PULLING);')

        c.add('unsigned int num_idle_warps = 0;')
        c.add('int src_warp_id = -1;')
        c.add('int lowest_lvl = -1;')
        c.add(f'bool is_successful = Themis::PullINodesAtZeroLvlDynamically<{self.gctxt.num_warps}>(thread_id, global_scan_offset, global_scan_end, local_scan_offset, ts_0_range_cached, inodes_cnts);')
        c.add('if (is_successful) {')
        c.add('loop = 0;')
        c.add('lowest_lvl = 0;')
        c.add('} else {')
        #c.add('return;')
        min_num_warps = min(self.gctxt.num_warps, KernelCall.args.min_num_warps)
        c.add(f'Themis::Wait<{self.gctxt.num_warps}, {min_num_warps}>(')
        c.add('gpart_id, src_warp_id,')
        c.add('warp_id, thread_id,')
        c.add('lowest_lvl,')
        c.add('warp_status,')
        c.add('num_idle_warps,')
        c.add('global_stats_per_lvl,')
        c.add('gts, size_of_stack_per_warp')
        if KernelCall.args.lb_detection == 'twolvlbitmaps':
            c.add(',global_bit1, global_bit2')
        elif KernelCall.args.lb_detection == 'randomized':
            c.add(',global_bit2')
        elif KernelCall.args.lb_detection == 'simple':
            c.add(',global_id_stack')
        c.add(');')
        c.add('if (src_warp_id == -2) {')
        if KernelCall.args.mode == 'sample':
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_END);')
        if KernelCall.args.mode == 'stats':
                c.add(f"stat_counters[TYPE_STATS_WAITING] += (clock64() - tp);")
        c.add('if (blockIdx.x == 0 && threadIdx.x == 0) warp_status->terminate();')
        c.add('break;')
        c.add('}')
        
        c.add('Themis::PushedParts::PushedPartsStack* stack = Themis::PushedParts::GetIthStack(gts, size_of_stack_per_warp, (size_t) src_warp_id);')
        
        c.add('switch (lowest_lvl) ')
        c.add('{')
        #speculated_size_attributes = 0
        for i, spctxt in enumerate(self.spctxts):
            c.add('case {}:'.format(i) + '{')
            if i == 0:
                c.add('Themis::PushedParts::PushedPartsAtZeroLvl* src_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
                c.add('Themis::PullINodesFromPPartAtZeroLvl(thread_id, src_pparts, ts_0_range_cached, inodes_cnts);')
                #c.add('int num_nodes = 0;')
                #c.add(f'Themis::CountINodesAtZeroLvl(thread_id, ts_0_range_cached, num_nodes);')
                #c.add(f'if (thread_id == 0) assert(num_nodes == inodes_cnts);')
                c.add(f'if (thread_id == 0) stack->PopPartsAtZeroLvl();')
            elif spctxt.prev.ts_num_ranges == 1: # Loop lvl
                c.add('Themis::PushedParts::PushedPartsAtLoopLvl* src_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
                c.add(f'Themis::PullINodesFromPPartAtLoopLvl(thread_id, {i}, src_pparts, ts_{i}_range_cached, ts_{i}_range, inodes_cnts);')
                #c.add('int num_nodes = 0;')
                #c.add(f'Themis::CountINodesAtLoopLvl(thread_id, {i}, ts_{i}_range_cached, ts_{i}_range_cached, num_nodes);')
                #c.add(f'unsigned dpart_mask = __ballot_sync(ALL_LANES, ts_{i}_range_cached.start < ts_{i}_range_cached.end);')
                #c.add(f'if (thread_id == 0 && num_nodes >= 32) assert(dpart_mask == 0xFFFFFFFFu);')
                #c.add(f'if (thread_id == 0) assert(num_nodes == inodes_cnts);')
                if len(spctxt.prev.ts_list_attributes) > 0:
                    c.add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')
                speculated_size_attributes = 0
                for attr in spctxt.prev.ts_list_attributes:
                    if attr.dataType == Type.STRING:
                        c.add(f'Themis::PullStrAttributesAtLoopLvl(thread_id, ts_{spctxt.id}_attr_{attr.id_name}_cached, ts_{spctxt.id}_attr_{attr.id_name}, (volatile str_t*) (src_pparts_attrs + {speculated_size_attributes}));')
                    else:
                        c.add(f'Themis::PullAttributesAtLoopLvl<{langType(attr.dataType)}>(thread_id, ts_{spctxt.id}_attr_{attr.id_name}_cached, ts_{spctxt.id}_attr_{attr.id_name}, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size_attributes}));')
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
                c.add(f'if (thread_id == 0) stack->PopPartsAtLoopLvl({speculated_size_attributes});')
            else: # If lvl
                c.add('Themis::PushedParts::PushedPartsAtIfLvl* src_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
                c.add('Themis::PullINodesFromPPartAtIfLvl(thread_id, {}, src_pparts, inodes_cnts);'.format(i))
                if len(spctxt.prev.ts_list_attributes) > 0:
                    c.add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')
                speculated_size_attributes = 0
                for attr in spctxt.prev.ts_list_attributes:
                    if attr.dataType == Type.STRING:
                        c.add('Themis::PullStrAttributesAtIfLvl(thread_id, ts_{}_attr_{}_flushed, (volatile str_t*) (src_pparts_attrs + {}));'
                            .format(spctxt.id, attr.id_name, speculated_size_attributes))
                    else:
                        c.add('Themis::PullAttributesAtIfLvl<{}>(thread_id, ts_{}_attr_{}_flushed, ({}*) (src_pparts_attrs + {}));'
                            .format(langType(attr.dataType), spctxt.id, attr.id_name, langType(attr.dataType), speculated_size_attributes))
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
                c.add(f'if (thread_id == 0) stack->PopPartsAtIfLvl({speculated_size_attributes});')
            #c.add(f'src_pparts +=  {self.gctxt.num_warps};')
            c.add('} break;')
        c.add('}')
        c.add('if (thread_id == 0) {')
        c.add('__threadfence();')
        c.add('stack->FreeLock();')
        c.add('}')

        if KernelCall.args.lb_aggressive:
            c.add('loop = {} - 1;'.format(KernelCall.args.lb_push_period))
        else:
            c.add('\tloop = 0;')
        
        c.add('}')

        if KernelCall.args.lb and KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
            c.add('Themis::WorkloadTracking::InitLocalWorkloadSize(lowest_lvl, inodes_cnts, local_info, global_stats_per_lvl);')
        
        c.add('lvl = lowest_lvl;')
        c.add('if (thread_id == lvl) mask_32 = inodes_cnts >= 32 ? 0x1 << lvl : 0;')
        c.add('mask_1 = 0x1 << lvl;')
        c.add('mask_32 = __shfl_sync(ALL_LANES, mask_32, lvl);')

    def genPushingCode(self, c):
        if KernelCall.args.mode == 'sample':
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_DETECTING);')
            
        c.add('int target_warp_id = -1;')
        c.add('unsigned int num_idle_warps = 0;')
        c.add('unsigned int num_warps = 0;')
        if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
            c.add('bool is_allowed = Themis::isPushingAllowed(thread_id, warp_status, num_idle_warps, num_warps, local_info, global_stats_per_lvl);')
        else:
            c.add('bool is_allowed = Themis::isPushingAllowed(thread_id, warp_status, num_idle_warps, num_warps);')
        c.add('if (is_allowed) {')
        
        min_num_warps = min(self.gctxt.num_warps, KernelCall.args.min_num_warps)
        
        c.add(f'Themis::FindIdleWarp<{len(self.spctxts)},{self.gctxt.num_warps}, {min_num_warps}>(')
        #c.add('target_warp_id, warp_id, thread_id, global_num_idle_warps, num_idle_warps, num_warps,gts, size_of_stack_per_warp')
        c.add('target_warp_id, warp_id, thread_id, warp_status, num_idle_warps, num_warps,gts, size_of_stack_per_warp')
        if KernelCall.args.lb_detection == 'twolvlbitmaps':
            c.add(', global_bit1, global_bit2')
        elif KernelCall.args.lb_detection == 'randomized':
            c.add(', global_bit2')
        elif KernelCall.args.lb_detection == 'simple':
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
        
        if KernelCall.args.lb_mode == 'clvl':
            c.add('int lvl_to_push = lvl;')
        elif KernelCall.args.lb_mode == 'highest':
            c.add('int lvl_to_push = Themis::CalculateHighestLvl(inodes_cnts);')
        elif KernelCall.args.lb_mode == 'llvl': # glvl or llvl
            c.add('int lvl_to_push = __ffs(mask_1) - 1;')
        elif KernelCall.args.lb_mode == 'glvl':
            c.add('int lvl_to_push = local_info.locally_lowest_lvl;')

        c.add('int num_to_push = 0;')
        c.add('int num_remaining = 0;') 
        if KernelCall.args.lb_mode == 'glvl':
            c.add('int num_nodes = local_info.num_nodes_at_locally_lowest_lvl > 0 ? local_info.num_nodes_at_locally_lowest_lvl : 0;')
        else:
            c.add('int num_nodes = 0;')
        c.add(f'unsigned m = 0x1u << lvl_to_push;')
        c.add('switch(lvl_to_push)')
        c.add('{')
        
        #speculated_size_attributes = 0    
            
        for i, spctxt in enumerate(self.spctxts):
            c.add('case {}:'.format(i) + '{')
            if i == 0:
                c.add(f'if (thread_id == 0) stack->PushPartsAtZeroLvl();')
                c.add('Themis::PushedParts::PushedPartsAtZeroLvl* target_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
                step = 'blockDim.x * gridDim.x'
                if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode != 'glvl':
                    c.add('CountINodesAtZeroLvl(thread_id, ts_0_range_cached, inodes_cnts);')
                    c.add('num_nodes = __shfl_sync(ALL_LANES, inodes_cnts, 0);')

                c.add(f'num_to_push = Themis::PushINodesToPPartAtZeroLvl(thread_id, target_pparts, ts_0_range_cached, {step});')
                c.add(f'num_remaining = num_nodes - num_to_push;')
                c.add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
                c.add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
            elif spctxt.prev.ts_num_ranges == 1:

                speculated_size_attributes = 0    
                for attr in spctxt.prev.ts_list_attributes:
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
                c.add(f'if (thread_id == 0) stack->PushPartsAtLoopLvl({i}, {speculated_size_attributes});')
                c.add('Themis::PushedParts::PushedPartsAtLoopLvl* target_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
                
                if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode != 'glvl':
                    c.add(f'Themis::CountINodesAtLoopLvl(thread_id, {i}, ts_{i}_range_cached, ts_{i}_range, inodes_cnts);')
                    c.add(f'num_nodes = __shfl_sync(ALL_LANES, inodes_cnts, {i});')
                c.add(f'num_to_push = Themis::PushINodesToPPartAtLoopLvl(thread_id, {i}, target_pparts, ts_{i}_range_cached, ts_{i}_range);')
                c.add(f'num_remaining = num_nodes - num_to_push;')
                c.add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
                c.add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
                c.add(f'int ts_src = 32;')
                c.add(f'Themis::DistributeFromPartToDPart(thread_id, {i}, ts_src, ts_{i}_range, ts_{i}_range_cached, mask_32, mask_1);')
                speculated_size_attributes = 0    
                if len(spctxt.prev.ts_list_attributes) > 0: 
                    c.add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
                for attr in spctxt.prev.ts_list_attributes:
                    c.add('{')
                    if attr.dataType == Type.STRING:
                        c.add(f'Themis::PushStrAttributesAtLoopLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size_attributes}), ts_{spctxt.id}_attr_{attr.id_name}_cached, ts_{spctxt.id}_attr_{attr.id_name});')
                        c.add(f'char* ts_{spctxt.id}_attr_{attr.id_name}_start_cached = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}.start, ts_src);')
                        c.add(f'char* ts_{spctxt.id}_attr_{attr.id_name}_end_cached = (char*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}.end, ts_src);')
                        c.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached.start = ts_{spctxt.id}_attr_{attr.id_name}_start_cached;')
                        c.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached.end = ts_{spctxt.id}_attr_{attr.id_name}_end_cached;')
                    elif attr.dataType == Type.PTR_INT:
                        c.add(f'Themis::PushPtrIntAttributesAtLoopLvl(thread_id, (volatile int**) (target_pparts_attrs + {speculated_size_attributes}), ts_{spctxt.id}_attr_{attr.id_name}_cached, ts_{spctxt.id}_attr_{attr.id_name});')
                        c.add(f'int* ts_{spctxt.id}_attr_{attr.id_name}_cached0 = (int*) __shfl_sync(ALL_LANES, (uint64_t) ts_{spctxt.id}_attr_{attr.id_name}[0], ts_src);')
                        c.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached = ts_{spctxt.id}_attr_{attr.id_name}_cached0;')
                    else:
                        c.add(f'Themis::PushAttributesAtLoopLvl<{langType(attr.dataType)}>(thread_id, (volatile {langType(attr.dataType)}*) (target_pparts_attrs + {speculated_size_attributes}), ts_{spctxt.id}_attr_{attr.id_name}_cached, ts_{spctxt.id}_attr_{attr.id_name});')
                        c.add(f'{langType(attr.dataType)} ts_{spctxt.id}_attr_{attr.id_name}_cached0 = __shfl_sync(ALL_LANES, ts_{spctxt.id}_attr_{attr.id_name}, ts_src);')
                        c.add(f'if (ts_src < 32) ts_{spctxt.id}_attr_{attr.id_name}_cached = ts_{spctxt.id}_attr_{attr.id_name}_cached0;')
                    c.add('}')
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
            else:
                speculated_size_attributes = 0    
                for attr in spctxt.prev.ts_list_attributes:
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
                c.add(f'if (thread_id == 0) stack->PushPartsAtIfLvl({i}, {speculated_size_attributes});')
                c.add('Themis::PushedParts::PushedPartsAtIfLvl* target_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
                if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode != 'glvl':
                    c.add(f'num_nodes = __shfl_sync(ALL_LANES, inodes_cnts, {i});')
                c.add(f'num_to_push = Themis::PushINodesToPPartAtIfLvl(thread_id, {i}, target_pparts, inodes_cnts);')
                #c.add(f'assert(num_nodes == num_to_push);')
                c.add(f'mask_32 = ((~m) & mask_32);')
                c.add(f'mask_1 = ((~m) & mask_1);')
                if len(spctxt.prev.ts_list_attributes) > 0:
                    c.add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
                speculated_size_attributes = 0    
                for attr in spctxt.prev.ts_list_attributes:
                    if attr.dataType == Type.STRING:
                        c.add('Themis::PushStrAttributesAtIfLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {}), ts_{}_attr_{}_flushed);'
                            .format(speculated_size_attributes, spctxt.id, attr.id_name))
                    elif attr.dataType == Type.PTR_INT:
                        c.add('Themis::PushPtrIntAttributesAtIfLvl(thread_id, (volatile int**) (target_pparts_attrs + {}), ts_{}_attr_{}_flushed);'
                            .format(speculated_size_attributes, spctxt.id, attr.id_name))
                    else:
                        c.add('Themis::PushAttributesAtIfLvl<{}>(thread_id, (volatile {}*) (target_pparts_attrs + {}), ts_{}_attr_{}_flushed);'
                            .format(langType(attr.dataType), langType(attr.dataType), speculated_size_attributes, spctxt.id, attr.id_name))
                    
                    speculated_size_attributes += CType.size[langType(attr.dataType)] * (2 * spctxt.gctxt.ts_width)
            #c.add(f'target_pparts +=  {self.gctxt.num_warps};')
            c.add('} break;')
        c.add('}')
        num_warps_per_block = int(KernelCall.defaultBlockSize / 32)
        c.add(f'if ((target_warp_id / {num_warps_per_block}) == (gpart_id / {num_warps_per_block})) __threadfence_block();')
        c.add('else __threadfence();')
        
        if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
            c.add("Themis::WorkloadTracking::UpdateWorkloadSizeOfIdleWarpAfterPush(thread_id, lvl_to_push, num_to_push, global_stats_per_lvl);")

        c.add('if (thread_id == 0) stack->FreeLock();')
        
        if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
            c.add('// Calculate the workload size of this busy warp')
            c.add('int new_num_nodes_at_locally_lowest_lvl = num_remaining;')
            c.add('int8_t new_local_max_order = Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);')
            c.add('int new_local_lowest_lvl = new_num_nodes_at_locally_lowest_lvl > 0 ? lvl_to_push : -1;')
            c.add('if (new_num_nodes_at_locally_lowest_lvl == 0 && mask_1 != 0) {')
            c.add('new_local_lowest_lvl = __ffs(mask_1) - 1;')
            if len(self.spctxts) > 1:
                c.add('switch (new_local_lowest_lvl) {')
                for i, spctxt in enumerate(self.spctxts):
                    if i == 0: continue
                    c.add('case {}:'.format(i) + '{')
                    if spctxt.prev.ts_num_ranges == 1:
                        c.add(f'CountINodesAtLoopLvl(thread_id, {i}, ts_{i}_range_cached, ts_{i}_range, inodes_cnts);')
                        c.add(f'new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {i});')
                        c.add(f'new_local_max_order = Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);')
                    else:
                        c.add(f'new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {i});')
                        c.add(f'new_local_max_order = 0;')
                        pass
                    c.add('} break;')
                c.add('}')      
            c.add('}')
            c.add('Themis::WorkloadTracking::UpdateWorkloadSizeOfBusyWarpAfterPush(thread_id, mask_1, new_num_nodes_at_locally_lowest_lvl, new_local_lowest_lvl, new_local_max_order, local_info, global_stats_per_lvl);')
        
        if KernelCall.args.lb_aggressive:
            c.add('interval = 0;')
            #c.add('interval = new_num_nodes_at_locally_lowest_lvl == 0 ? 1 : 0;')
            

        c.add('Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        c.add('}')
        if KernelCall.args.lb_aggressive:
            c.add('else { ')
            if KernelCall.args.lb_mode == 'glvl':
                c.add(f'Themis::chooseNextIntervalAfterPush(interval, local_info, num_warps, num_idle_warps, is_allowed, {KernelCall.args.lb_push_period});')
            else:
                c.add(f'Themis::chooseNextIntervalAfterPush(interval, num_warps, num_idle_warps, is_allowed, {KernelCall.args.lb_push_period});')
            c.add('}')

            c.add('loop = {} - interval;'.format(KernelCall.args.lb_push_period))
        else:
            c.add('loop = 0;')

        if KernelCall.args.mode == 'sample':
            pass
            #c.add("#ifdef MODE_SAMPLE")
            #c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_PUSHING);')
            #c.add("#endif")
        

    def genLoadBalancing(self, c):
        
        if KernelCall.args.lb_type == 'ws':
            self.genWorksharingPullingCode(self.pullcode)
        else:        
            self.genPullingCode(self.pullcode)
            self.genPushingCode(self.pushcode)
        

    def toSingleString(self):

        c = Code()
        self.genInitWarp(c)
        self.genInitStringConstants(c)
        c.add('int active = 0;')
        c.addAll(self.precode.lines)
        c.addAll(self.switchcode.lines)
        c.add('}')        
        c.addAll(self.postcode.lines)
        cc = c

        c = Code()
        c.addAll(self.defcode.lines)

        mc = self.gctxt.maincode
        
        c.add("__global__ void")
        c.add("krnl_{} (".format(self.id))
        for var in self.vars:
            c.add('\t' + var.toArg())
        c.add('\tint* cnt')
        c.add("\t) {")
        c.addAll(list(map(lambda x: '\t' + x, cc.lines)))
        c.add("}")

        cc = self.gctxt.krnlexeccode
        cc.add('std::clock_t start_kernelTime{}, stop_kernelTime{};'.format(self.id, self.id))
        cc.add('{')

        cc.add('\tcudaMemset(cnt, 0, 32 * sizeof(int));')
        cc.add('\tprintf("krnl_{} start\\n");'.format(self.id))
        cc.add('\tstart_kernelTime{} = std::clock();'.format(self.id))

        gridSize = 1
        blockSize = 1
        #gridSize = 1024 * 1024
        cc.add('\tkrnl_{}<<<{},{}>>>('.format(self.id, gridSize, blockSize))
        for var in self.vars:
            cc.add('\t\t' + var.toCall())
        cc.add('\t\tcnt')
        cc.add('\t\t);')
        cc.addAll(list(map(lambda x: '\t' + x, self.postexeccode.lines)))
        cc.add('\tcudaDeviceSynchronize();')

        cc.add('\tstop_kernelTime{} = std::clock();'.format(self.id))
        cc.add('\tcudaError err = cudaGetLastError();')
        cc.add('\tif(err != cudaSuccess) {')
        cc.add('\t\tstd::cerr << "Cuda Error in krnl_{}! " << cudaGetErrorString( err ) << std::endl;'.format(self.id))
        cc.add('\t\tERROR("krnl_{}")'.format(self.id))
        cc.add('\t}')
        
        
        #cc.add('\tcudaMemcpy(&cpu_cnts, cnt, 32 * sizeof(int), cudaMemcpyDeviceToHost);')
        #cc.add('\tfor (int i = 0; i < {}; ++i)'.format(len(self.spctxts)) + '{')
        #cc.add('\t\tprintf("krnl_{} cnt[%d]: %d\\n", i, cpu_cnts[i]);'.format(self.id))
        #cc.add('\t}')
        cc.add('\tprintf("krnl_{} result: %d, time: %6.1f\\n", cpu_cnts[0], (stop_kernelTime{} - start_kernelTime{}) / (double) (CLOCKS_PER_SEC / 1000));'.format(self.id, self.id, self.id))
        
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                size = self.gctxt.num_warps if counter not in KernelCall.args.num_counters else KernelCall.args.num_counters[counter]
                cc.add('\tcudaMemcpy(cpu_counters.data(), {}s, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter,size))
                cc.add('\tprintf("krnl_{} {}s:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(size) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");')

        cc.add('\tprintf("krnl_{} stop\\n");'.format(self.id))
        cc.add('}')
        return c


    def toPyperString(self):

        c = Code()
        self.genInitProfileCountersInKernel(c)
        self.genInitWarp(c)
        self.genInitStringConstants(c)

        #c.add('bool isLast = (gridDim.x - blockIdx.x) <= 82 * 16;')

        c.add('int active = isLast ? 0 : 1;')
        c.add('bool keepGoing = isLast;')
        c.add('__shared__ int shmActThreadsB;')
        c.add('__shared__ int shmNumWarps;')
        c.add('__shared__ int shmNumInBuffer;')
        c.add('__shared__ unsigned shmWarpActive[{}];'.format(int(KernelCall.defaultBlockSize / 32)))

        if KernelCall.args.mode == 'profile':
            c.add('active_clock = clock64();')

        if KernelCall.args.mode == 'sample':
            c.add('if (threadIdx.x == 0) {')
            c.add('uint32_t smid32;')
            c.add('asm volatile("mov.u32 %0, %%smid;" : "=r"(smid32));')
            c.add('uint64_t smid = (uint64_t) smid32;')
            c.add('unsigned long long t = TYPE_SAMPLE_ACTIVE;')
            c.add('unsigned long long cck = clock64();')
            c.add('samples[blockIdx.x*2] = (cck > (*kernel_start) ? cck - (*kernel_start) : 0) | (t << 60) | (smid << 52);')
            c.add('}')
            
        c.add('int scan_offset = blockIdx.x * blockDim.x;')
        c.add('while (true) {')
        c.addAll(self.precode.lines)
        c.addAll(self.switchcode.lines)        
        c.addAll(self.postcode.lines)
        c.add('keepGoing = false;')
        c.add('scan_offset += blockDim.x * gridDim.x;')
        c.add('}')

        if KernelCall.args.mode == 'sample':
            c.add('if (threadIdx.x == 0) {')
            c.add('uint32_t smid32;')
            c.add('asm volatile("mov.u32 %0, %%smid;" : "=r"(smid32));')
            c.add('uint64_t smid = (uint64_t) smid32;')
            c.add('unsigned long long t = TYPE_SAMPLE_ACTIVE;')
            c.add('unsigned long long cck = clock64();')
            c.add('samples[blockIdx.x*2+1] = (cck > (*kernel_start) ? cck - (*kernel_start) : 0) | (t << 60) | (smid << 52);')
            c.add('}')

        if KernelCall.args.mode == 'profile':
            c.add('active_clock = clock64() - active_clock;')
            c.add("if (thread_id == 0) {")            
            c.add('atomicAdd(&active_clocks[warp_id], active_clock);')
            c.add('atomicAdd(&active_lanes_nums[warp_id], active_lanes_num);')
            c.add('atomicAdd(&oracle_active_lanes_nums[warp_id], oracle_active_lanes_num);')
            c.add("}")

        cc = c

        c = Code()
        c.addAll(self.defcode.lines)

        mc = self.gctxt.maincode
        for i, spctxt in enumerate(self.spctxts[:len(self.spctxts)-1]):
            mc.add('int* global{}_{}_lock;'.format(self.id, spctxt.id))
            mc.add('cudaMalloc((void**)&global{}_{}_lock, sizeof(int));'.format(self.id, spctxt.id))
            mc.add('cudaMemset(global{}_{}_lock, 0, sizeof(int));'.format(self.id, spctxt.id))
            mc.add('int* global{}_{}_num;'.format(self.id, spctxt.id))
            mc.add('cudaMalloc((void**)&global{}_{}_num, sizeof(int));'.format(self.id, spctxt.id))
            mc.add('cudaMemset(global{}_{}_num, 0, sizeof(int));'.format(self.id, spctxt.id))
            for attr in spctxt.ts_list_attributes:
                if attr.dataType == Type.STRING:
                    mc.add('str_t* global{}_{}_{};'.format(self.id, spctxt.id, attr.id_name))
                    mc.add('cudaMalloc((void**)&global{}_{}_{}, {} * sizeof(str_t));'
                        .format(self.id, spctxt.id, attr.id_name, KernelCall.defaultBlockSize))
                else:
                    mc.add('{}* global{}_{}_{};'.format(langType(attr.dataType), self.id, spctxt.id, attr.id_name))
                    mc.add('cudaMalloc((void**)&global{}_{}_{}, {} * sizeof({}));'
                        .format(self.id, spctxt.id, attr.id_name, KernelCall.defaultBlockSize, langType(attr.dataType)))



        c.add("__global__ void")
        c.add("krnl_{} (".format(self.id))
        for var in self.vars:
            c.add('\t' + var.toArg())
        for i, spctxt in enumerate(self.spctxts[:len(self.spctxts)-1]):
            c.add('int* global{}_{}_lock,'.format(self.id, spctxt.id))
            c.add('int* global{}_{}_num,'.format(self.id, spctxt.id))
            for attr in spctxt.ts_list_attributes:
                if attr.dataType == Type.STRING:
                    c.add('volatile str_t* global{}_{}_{},'.format(self.id, spctxt.id, attr.id_name))
                else:
                    c.add('volatile {}* global{}_{}_{},'.format(langType(attr.dataType), self.id, spctxt.id, attr.id_name))
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                c.add('\tunsigned long long* {}s,'.format(counter))    
        if KernelCall.args.mode == 'stats':
            for counter in ['pushing_clock', 'processing_clock', 'waiting_clock', 'num_idle']:
                c.add(f'unsigned long long* global_{counter}')    
                
        if KernelCall.args.mode == 'sample':
            c.add('unsigned long long* samples, unsigned long long *kernel_start,')

        c.add('bool isLast,')
        c.add('int* cnt')
        c.add("\t) {")
        c.addAll(list(map(lambda x: '\t' + x, cc.lines)))
        c.add("}")

        cc = self.gctxt.krnlexeccode
        cc.add(f"cudaEvent_t start_kernelTime{self.id}, stop_kernelTime{self.id};")
        cc.add(f"float elapsedTime{self.id};")
        cc.add(f"cudaEventCreate(&start_kernelTime{self.id});")
        cc.add(f"cudaEventCreate(&stop_kernelTime{self.id});")
        #cc.add('std::clock_t start_kernelTime{}, stop_kernelTime{};'.format(self.id, self.id))
        cc.add('{')
        if KernelCall.args.lb:
            cc.add('\tcudaMemset(global_num_idle_warps, 0, sizeof(unsigned int));')
        cc.add('\tcudaMemset(cnt, 0, 32 * sizeof(int));')
        cc.add('\tprintf("krnl_{} start\\n");'.format(self.id))
        
        
        
        
        #gridSize = int(self.scanSize / KernelCall.defaultBlockSize)+ 1 #+ 82 * 16

        gridSize = KernelCall.defaultGridSize
        numWarps = KernelCall.defaultGridSize * int(KernelCall.defaultBlockSize / 4)

        if KernelCall.args.mode == 'sample':
            cc.add('unsigned long long* samples;')
            cc.add('cudaMalloc((void**)&samples, sizeof(unsigned long long) * {} * 2);'.format(gridSize))
            cc.add('cudaDeviceSynchronize();')
            cc.add('\tkrnl_sample_start<<<1,32>>>(sample_start);')
            cc.add('cudaMemset(samples, 0, {});')
            cc.add('cudaDeviceSynchronize();')
            cc.add('sleep(1);')


        cc.add('{')
        cc.add('//gridSize:{}'.format(gridSize))
        self.genInitProfileCounters(cc, numWarps)


        cc.add(f"cudaEventRecord(start_kernelTime{self.id});")

        cc.add(f'krnl_{self.id}<<<{KernelCall.defaultGridSize},{KernelCall.defaultBlockSize}>>>(')
        #cc.add('\tkrnl_{}<<<{},{}>>>('.format(self.id, gridSize, KernelCall.defaultBlockSize))
        for var in self.vars:
            cc.add('\t\t' + var.toCall())
        for i, spctxt in enumerate(self.spctxts[:len(self.spctxts)-1]):
            cc.add('global{}_{}_lock,'.format(self.id, spctxt.id))
            cc.add('global{}_{}_num,'.format(self.id, spctxt.id))
            for attr in spctxt.ts_list_attributes:
                if attr.dataType == Type.STRING:
                    cc.add('global{}_{}_{},'.format(self.id, spctxt.id, attr.id_name))
                else:
                    cc.add('global{}_{}_{},'.format(self.id, spctxt.id, attr.id_name))
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                cc.add('\t\t{}s,'.format(counter))  
        if KernelCall.args.mode == 'stats':
            for counter in ['pushing_clock', 'processing_clock', 'waiting_clock', 'num_idle']:
                cc.add(f'global_{counter},')


        if KernelCall.args.mode == 'sample':
            cc.add('\t\tsamples,sample_start,')
        cc.add('false,')
        cc.add('\t\tcnt')
        cc.add('\t\t);')
        cc.add('\tcudaDeviceSynchronize();')

        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                size = self.gctxt.num_warps if counter not in KernelCall.args.num_counters else KernelCall.args.num_counters[counter]
                cc.add('\tcudaMemcpy(cpu_counters.data(), {}s, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter, numWarps))
                cc.add('\tprintf("krnl_{} {}s:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(numWarps) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");')
                
        if KernelCall.args.mode == 'stats':
            for counter in ['pushing_clock', 'processing_clock', 'waiting_clock', 'num_idle']:
                size = self.gctxt.num_warps
                cc.add('\tcudaMemcpy(cpu_counters.data(), global_{}, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter, numWarps))
                cc.add('\tprintf("krnl_{} global_{}:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(size) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");')
                
        cc.add('}')

        cc.add('{')
        self.genInitProfileCounters(cc, (32 * KernelCall.defaultBlockSize))
        cc.add('\tkrnl_{}<<<{},{}>>>('.format(self.id, 32, KernelCall.defaultBlockSize))
        for var in self.vars:
            cc.add('\t\t' + var.toCall())
        for i, spctxt in enumerate(self.spctxts[:len(self.spctxts)-1]):
            cc.add('global{}_{}_lock,'.format(self.id, spctxt.id))
            cc.add('global{}_{}_num,'.format(self.id, spctxt.id))
            for attr in spctxt.ts_list_attributes:
                if attr.dataType == Type.STRING:
                    cc.add('global{}_{}_{},'.format(self.id, spctxt.id, attr.id_name))
                else:
                    cc.add('global{}_{}_{},'.format(self.id, spctxt.id, attr.id_name))
        if KernelCall.args.mode == 'profile':
            cc.add('#ifdef MODE_PROFILE')
            for counter in KernelCall.counters:
                cc.add('\t\t{}s,'.format(counter))   
            cc.add('#endif')
        if KernelCall.args.mode == 'sample':
            cc.add('\t\tsamples,sample_start,')
        cc.add('true,')
        cc.add('\t\tcnt')
        cc.add('\t\t);')
        cc.add('\tcudaDeviceSynchronize();')

        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                size = self.gctxt.num_warps if counter not in KernelCall.args.num_counters else KernelCall.args.num_counters[counter]
                cc.add('\tcudaMemcpy(cpu_counters.data(), {}s, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter,32))
                cc.add('\tprintf("krnl_{}_tail {}s:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(16) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");')

        cc.add('}')
        
        if len(self.postexeccode.lines) > 0:
            cc.addAll(list(map(lambda x: '\t' + x, self.postexeccode.lines)))
            cc.add('\tcudaDeviceSynchronize();')
        
        cc.add(f"cudaEventRecord(stop_kernelTime{self.id});")
        cc.add(f"cudaEventSynchronize(stop_kernelTime{self.id});")
        cc.add(f"cudaEventElapsedTime(&elapsedTime{self.id}, start_kernelTime{self.id}, stop_kernelTime{self.id});")
        cc.add('\tcudaError err = cudaGetLastError();')
        cc.add('\tif(err != cudaSuccess) {')
        cc.add('\t\tstd::cerr << "Cuda Error in krnl_{}! " << cudaGetErrorString( err ) << std::endl;'.format(self.id))
        cc.add('\t\tERROR("krnl_{}")'.format(self.id))
        cc.add('\t}')

        cc.add('int global_num;')
        for i, spctxt in enumerate(self.spctxts[:len(self.spctxts)-1]):  
            cc.add('cudaMemcpy(&global_num, global{}_{}_num, sizeof(int), cudaMemcpyDeviceToHost);'.format(self.id, spctxt.id))
            cc.add('printf("global{}_{}_num: %d\\n", global_num);'.format(self.id, spctxt.id))

        cc.add('\tcudaMemcpy(&cpu_cnts, cnt, 32 * sizeof(int), cudaMemcpyDeviceToHost);')
        cc.add('\tfor (int i = 0; i < {}; ++i)'.format(len(self.spctxts)) + '{')
        cc.add('\t\tprintf("krnl_{} cnt[%d]: %d\\n", i, cpu_cnts[i]);'.format(self.id))
        cc.add('\t}')
        cc.add('\tprintf("krnl_{} result: %d, time: %6.1f\\n", cpu_cnts[0], elapsedTime{});'.format(self.id, self.id))
        


        if KernelCall.args.mode == 'sample':
        #gridSize = int(self.scanSize / KernelCall.defaultBlockSize)+ 1 + 82 * 4
            cc.add('std::vector<unsigned long long> cpu_samples({});'.format(gridSize*2))
            cc.add('cudaMemcpy(cpu_samples.data(), samples, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(gridSize*2))
            cc.add('cudaFree(samples);')
            cc.add('for (int w = 0; w < {}; ++w) '.format(gridSize) + ' {')
            cc.add('\tprintf("sample krnl_{} %d: ", w);'.format(self.id))
            cc.add('\tunsigned long long* d = (&cpu_samples[w * 2]);')
            cc.add('\tfor (int e = 0; e < 2; ++e) {')
            cc.add('\t\tprintf(" %lld/%lld/%d/%lld", d[e] & 0x000FFFFFFFFFFFFF, d[e] >> 52 & 0xFF, 0, d[e] >> 60);')
            cc.add('\t}')
            cc.add('\tprintf("\\n");')
            cc.add('}')

        cc.add('int numBlocks = 0;')
        cc.add(f'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, krnl_{self.id}, {KernelCall.defaultBlockSize}, 0);')
        cc.add(f'printf("krnl_{self.id} # blocks: %d\\n", numBlocks);')
        
        cc.add('cudaFuncAttributes attr;')
        cc.add(f'cudaFuncGetAttributes(&attr, krnl_{self.id});')
        cc.add('printf("Number of registers per thread: %d\\n", attr.numRegs);')
        
        
        cc.add('\tprintf("krnl_{} stop\\n");'.format(self.id))
        cc.add('}')
        return c


    def toString(self):
        
        doCnt = True

        c = Code()
        self.genInitProfileCountersInKernel(c)
        self.genInitWarp(c)
        self.genInitStringConstants(c)
        self.genInitSampling(c)
        self.genInitTrapezoidStack(c)
        
        c.addAll(self.precode.lines)

        if KernelCall.args.mode == 'sample':
            c.add('unsigned long long sampling_start = 0;')
            c.add('samples += gpart_id * WIDTH_SAMPLES;')
            c.add('if (thread_id == 0) samples[0] = 0;')
            c.add('sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_START);')

        if KernelCall.args.lb:
            if KernelCall.args.lb_type == 'ws': # work sharing
                pass
            else:
                if KernelCall.args.lb_mode == 'morsel' :
                    pass
                else:
                    #c.add('int8_t locally_current_order, globally_current_max_order, globally_max_order;')
                    #c.add('if (thread_id == 0) {')
                    c.add('Themis::LocalLevelAndOrderInfo local_info;')
                    c.add('Themis::WorkloadTracking::InitLocalWorkloadSizeAtZeroLvl(inodes_cnts, local_info, global_stats_per_lvl);')
            
                if KernelCall.args.mode == 'stats':
                    c.add('if (warp_id == 0 && thread_id == 0) {')
                    c.add('global_stats_per_lvl[0].num_inodes_at_that_time = ts_0_range_cached.end - ts_0_range_cached.start;')
                    c.add('global_stats_per_lvl[0].num_nodes = ts_0_range_cached.end - ts_0_range_cached.start;')
                    c.add('}')
            #c.add('Themis::Heap::insertKey(minheaps, 0, warp_id, thread_id, num_inodes_at_lowest_lvl);')
            self.genLoadBalancing(c)

        c.add('int lvl = -1;')
        c.add('int active;')
        #c.add('int sid = -1;')
        if KernelCall.args.lb:
            c.add('unsigned interval = 1;')
            c.add('unsigned loop = {} - 1;'.format(KernelCall.args.lb_push_period))
        #c.add('int num_nodes_at_loop_lvl = __shfl_sync(ALL_LANES, inodes_cnts, 0);')
        #c.add('int loop_lvl = num_nodes_at_loop_lvl > 0 ? 0 : -1;')
        c.add('unsigned mask_32 = 0;')
        c.add('unsigned mask_1 = 0;')
        c.add('Themis::UpdateMaskAtZeroLvl(0, thread_id, ts_0_range_cached, mask_32, mask_1);')
        
        
        c.add('Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
        
        if KernelCall.args.lb:
            c.add('do {')
            
        if KernelCall.args.mode == 'sample':
            #c.add('unsigned long long sampling_start = *sample_start;')
            c.add(f'if (lvl != -1 && loop < {KernelCall.args.lb_push_period}) sample(locally_lowest_lvl, thread_id, samples, sampling_start, TYPE_SAMPLE_ACTIVE);')
        
        if KernelCall.args.mode == 'profile':
            c.add('active_clock -= clock64();')

        if KernelCall.args.mode == 'stats':
            c.add('unsigned long long current_tp = clock64();')
            c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
            c.add('tp = current_tp;')
            c.add('current_status = TYPE_STATS_PROCESSING;')

        if KernelCall.args.switch:
            if KernelCall.args.lb and KernelCall.args.lb_type != 'ws':
                c.add(f'while (lvl > -1 && loop < {KernelCall.args.lb_push_period})' + ' {')
            else:
                c.add('\twhile (lvl > -1) {')
                
            c.add('\t\tswitch (lvl) {')
            c.addAll(self.switchcode.lines)
            c.add('\t\t}')
            c.add('if (num_nodes_at_loop_lvl >= 32) lvl = loop_lvl;')
            c.add('else Themis::ChooseLvl(thread_id, inodes_cnts, lvl);')   
            c.add('\t}')  
        else:
            c.addAll(self.switchcode.lines)
            c.add('}' * self.num_loops + f'// num_loops: {self.num_loops}')
        
        
        
        c.addAll(self.selectparameter.lines)

        if KernelCall.args.mode == 'stats' and KernelCall.args.lb == False:
            c.add(f'stat_counters[TYPE_STATS_PROCESSING] += (clock64() - tp);')
        c.add('') 
        
        
        if KernelCall.args.mode == 'profile':
            c.add('active_clock += clock64();')

        
        if KernelCall.args.lb == True:
            c.add('\tloop = 0;')
            c.add('\tif (lvl == -1) {')
            #c.add('return;')
            c.add('inodes_cnts = 0;')
            c.add('mask_32 = mask_1 = 0;')
            if KernelCall.args.lb_type == 'wp' and KernelCall.args.lb_mode == 'glvl':
                c.add('if (thread_id == 0 && local_info.locally_lowest_lvl != -1) atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_warps, 1);')
                
            
            if KernelCall.args.mode == 'stats':
                c.add('unsigned long long current_tp = clock64();')
                c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
                c.add('tp = current_tp;')
                c.add('current_status = TYPE_STATS_WAITING;')
                c.add('stat_counters[TYPE_STATS_NUM_IDLE] += 1;')
            
            
            if KernelCall.args.lb_mode == 'morsel':
                c.add('//Morsel')
                c.add('if (thread_id == 0) {')
                c.add('int new_start = atomicAdd(global_num_idle_warps, 1);')
                c.add(f'ts_0_range_cached.start = 32 * 16 * (new_start + {self.gctxt.num_warps});')
                c.add('ts_0_range_cached.start = ts_0_range_cached.start < input_table_size ? ts_0_range_cached.start : input_table_size;')
                c.add('ts_0_range_cached.end = ts_0_range_cached.start + 32 * 16;')
                c.add('ts_0_range_cached.end = ts_0_range_cached.end < input_table_size ? ts_0_range_cached.end : input_table_size;')
                c.add('inodes_cnts = ts_0_range_cached.end - ts_0_range_cached.start;')
                c.add('lvl = inodes_cnts > 0 ? 0 : -2;')
                c.add('}')
                c.add('ts_0_range_cached.start = __shfl_sync(ALL_LANES, ts_0_range_cached.start, 0) + thread_id;')
                c.add('lvl = __shfl_sync(ALL_LANES, lvl, 0);')
            else:
                c.addAll(self.pullcode.lines)
                
                if KernelCall.args.mode == 'stats':
                    c.add('stat_counters[TYPE_STATS_NUM_PUSHED] += 1;')
            
            c.add('\t} else {')
            if KernelCall.args.lb_type != 'ws' and KernelCall.args.mode == 'stats':
                c.add('unsigned long long current_tp = clock64();')
                c.add('if (current_status != -1) stat_counters[current_status] += current_tp - tp;')
                c.add('tp = current_tp;')
                c.add('current_status = TYPE_STATS_PUSHING;')

            if KernelCall.args.lb_mode == 'morsel':
                c.add('//Morsel')
            else:
                c.addAll(self.pushcode.lines)


            c.add('\t}')
            c.add('} while (lvl != -2);')
        else:
            pass
            #c.add('\tloop = 0;')
            #c.add('} while (lvl != -1);')

        c.addAll(self.postcode.lines)
        c.add("if (thread_id == 0) {")
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                c.add('{}s[warp_id] = {};'.format(counter, counter))
                
        if KernelCall.args.mode == 'stats':
            for i, counter in enumerate(stat_counters):
                c.add(f'global_{counter}[warp_id] = stat_counters[{i}];')

        c.add("}")
 
        cc = c
        c = Code()
        c.addAll(self.defcode.lines)

        c.add("__global__ void")
        if KernelCall.args.lb:
            c.add("__launch_bounds__({}, 8)".format(KernelCall.defaultBlockSize))
        elif KernelCall.defaultBlockSize > 4:
            c.add("__launch_bounds__({}, 8)".format(KernelCall.defaultBlockSize))

        c.add("krnl_{} (".format(self.id))
        for var in self.vars:
            c.add('\t' + var.toArg())
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                c.add('\tunsigned long long* {}s,'.format(counter))        
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                c.add(f'unsigned long long* global_{counter},')

        if KernelCall.args.mode == 'sample':
            c.add('unsigned long long* samples, unsigned long long* sample_start,')
            
            
        if KernelCall.args.lb:
            c.add('unsigned int* global_num_idle_warps, int* global_scan_offset,')
            
            if KernelCall.args.lb_type == 'ws':
                c.add('WorkSharing::TaskBook* taskbook, WorkSharing::TaskStack* taskstack,')
            else:
                if KernelCall.args.lb_mode == 'morsel':
                    pass
                else:
                    c.add('Themis::PushedParts::PushedPartsStack* gts, size_t size_of_stack_per_warp,')
                    #c.add('char* gts_attrs,')
                    c.add('Themis::StatisticsPerLvl* global_stats_per_lvl,')
                    if KernelCall.args.lb_detection == 'twolvlbitmaps':
                        c.add('unsigned long long* global_bit1, unsigned long long* global_bit2,')
                    elif KernelCall.args.lb_detection == 'randomized':
                        c.add('unsigned long long* global_bit2,')
                    elif  KernelCall.args.lb_detection == 'simple':
                        c.add('Themis::Detection::Stack::IdStack* global_id_stack,')
            
        #c.add('\tunsigned long long* mf_cnt,')
        c.add('\tint* cnt')
        #c.add('\tThemis::Heap::MinHeap* minheaps')
        c.add("\t) {")
        c.addAll(list(map(lambda x: '\t' + x, cc.lines)))
        c.add("}")

        cc = self.gctxt.krnlexeccode
        
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                cc.add('\tcudaMemset(global_{}, 0, sizeof(unsigned long long) * {});'.format(counter, self.gctxt.num_warps))
            cc.add('cudaDeviceSynchronize();')


        cc.add(f"start_timepoint_{self.id} = std::chrono::steady_clock::now();")
        
        cc.add(f"cudaEvent_t start_kernelTime{self.id}, stop_kernelTime{self.id};")
        cc.add(f"float elapsedTime{self.id};")
        cc.add(f"cudaEventCreate(&start_kernelTime{self.id});")
        cc.add(f"cudaEventCreate(&stop_kernelTime{self.id});")
        cc.addAll(self.precode_outer_cpu.lines)
        cc.add('{')
        

        

        if KernelCall.args.lb:
            cc.add('cudaMemset(global_info, 0, 64 * sizeof(unsigned int));')

            if KernelCall.args.lb_type == 'ws': # work sharing
                cc.add('cudaMemset(taskbook, 0, 128);')
                cc.add('cudaMemset(taskstack, 0, 128);')
            else: # work pushing
                if KernelCall.args.lb_mode == 'morsel':
                    pass
                else:
                    if KernelCall.args.lb_detection == 'twolvlbitmaps':
                        #cc.add('cudaMemset(global_bit1, 0, 2 * sizeof(unsigned long long));')
                        #cc.add(f'cudaMemset(global_bit2, 0, {128 * (int((self.gctxt.num_warps-1) / 64) + 1)});')
                        
                        bitmapsize = (int((self.gctxt.num_warps-1) / 64) + 1)
                        #c.add(f'//bitmapsize: {bitmapsize} for {gctxt.num_warps}')
                        #self.cuMalloc('global_bit1', 'unsigned long long',  16 + bitmapsize * 16, gctxt)
                        #c.add('unsigned long long* global_bit2 = global_bit1 + 16;')
                        #c.add('cudaMemset(global_bit1, 0, sizeof(unsigned long long) * 2);')
                        cc.add(f'cudaMemset(global_bit1, 0, sizeof(unsigned long long) * {16 + bitmapsize * 16});')
                        
                        
                    if KernelCall.args.lb_detection == 'randomized':
                        cc.add(f'cudaMemset(global_bit2, 0, {128 * (int((self.gctxt.num_warps-1) / 64) + 1)});')
        
        if doCnt:
            cc.add("cudaMemset(cnt, 0, sizeof(int)*32);")
        
        #cc.add('\tcudaMemset(cnt, 0, sizeof(int));')
        #cc.add('\tcudaMemset(mf_cnt, 0, sizeof(unsigned long long));')

        
        #cc.add('Themis::Heap::ClearMinHeaps(minheaps);')
        if KernelCall.args.mode == 'sample':
            cc.add('\tkrnl_sample_start<<<1,32>>>(sample_start);')
            cc.add(f'cudaMemset(samples, 0, sizeof(unsigned long long) * WIDTH_SAMPLES * {self.gctxt.num_warps});')
            cc.add('cudaDeviceSynchronize();')
            
            #cc.add('sleep(1);')

        cc.add('\tprintf("krnl_{} start\\n");'.format(self.id))
        cc.add(f"cudaEventRecord(start_kernelTime{self.id});")
        

        cc.addAll(self.precode_cpu.lines)
        
        if KernelCall.args.lb:

            if KernelCall.args.lb_type == 'ws': # work sharing
                pass
            else:
                if KernelCall.args.lb_mode == 'morsel':
                    pass
                else:
                    
                    cc.add(f'\tcudaMemset(global_stats_per_lvl, 0, sizeof(Themis::StatisticsPerLvl) * {len(self.spctxts)});')
                    cc.add(f'Themis::InitStatisticsPerLvl(global_stats_per_lvl, {self.gctxt.num_warps}, {self.num_inodes_at_lvl_zero},  {len(self.spctxts)});')
                    #cc.add(f'Themis::krnl_InitStatisticsPerLvl<<<1,128>>>(global_stats_per_lvl, {self.gctxt.num_warps}, {self.num_inodes_at_lvl_zero},  {len(self.spctxts)});')
                    cc.add('cudaDeviceSynchronize();')

        cc.add('\tkrnl_{}<<<{},{}>>>('.format(self.id, KernelCall.defaultGridSize, KernelCall.defaultBlockSize))
        for var in self.vars:
            cc.add('\t\t' + var.toCall())

        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                cc.add('\t\t{}s,'.format(counter))   
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                cc.add(f'global_{counter},')


        if KernelCall.args.mode == 'sample':
            cc.add('\t\tsamples, sample_start,')
        if KernelCall.args.lb:
            cc.add('global_num_idle_warps, global_scan_offset,')
            
            if KernelCall.args.lb_type == 'ws': # work sharing
                cc.add('taskbook, taskstack,')
            else: # work pushing                
                if KernelCall.args.lb_mode == 'morsel':
                    pass
                else:
                    cc.add('gts, size_of_stack_per_warp, global_stats_per_lvl,')
                    if KernelCall.args.lb_detection == 'twolvlbitmaps':
                        cc.add('global_bit1, global_bit2,')
                    elif KernelCall.args.lb_detection == 'randomized':
                        cc.add('global_bit2,')
                    elif KernelCall.args.lb_detection == 'simple':
                        cc.add('global_id_stack,')
        cc.add('\t\tcnt')
        cc.add('\t\t);')

        cc.addAll(list(map(lambda x: '\t' + x, self.postexeccode.lines)))
        cc.add('\tcudaDeviceSynchronize();')
        cc.add(f"end_timepoint_{self.id} = std::chrono::steady_clock::now();")

        cc.addAll(self.postcode_cpu.lines)
        cc.add(f"cudaEventRecord(stop_kernelTime{self.id});")
        cc.add(f"cudaEventSynchronize(stop_kernelTime{self.id});")
        cc.add(f"cudaEventElapsedTime(&elapsedTime{self.id}, start_kernelTime{self.id}, stop_kernelTime{self.id});")
        #cc.add('\tcudaError err = cudaGetLastError();')
        #cc.add('\tif(err != cudaSuccess) {')
        #cc.add('\t\tstd::cerr << "Cuda Error in krnl_{}! " << cudaGetErrorString( err ) << std::endl;'.format(self.id))
        #cc.add('\t\tERROR("krnl_{}")'.format(self.id))
        #cc.add('\t}')
        
        if doCnt:
            cc.add('\tcudaMemcpy(&cpu_cnts, cnt, 32 * sizeof(int), cudaMemcpyDeviceToHost);')
            cc.add('\tfor (int i = 0; i < {}; ++i)'.format(len(self.spctxts)) + '{')
            cc.add('\t\tprintf("krnl_{} cnt[%d]: %d\\n", i, cpu_cnts[i]);'.format(self.id))
            cc.add('\t}')
            cc.add('\tprintf("krnl_{} result: %d, time: %6.1f\\n", cpu_cnts[0], elapsedTime{});'.format(self.id, self.id))
        
        if KernelCall.args.mode == 'profile':
            for counter in KernelCall.counters:
                cc.add('\tcudaMemcpy(cpu_counters.data(), {}s, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter, self.gctxt.num_warps))
                cc.add('\tprintf("krnl_{} {}s:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(self.gctxt.num_warps) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");')
        if KernelCall.args.mode == 'stats':
            for counter in stat_counters:
                cc.add('\tcudaMemcpy(cpu_counters.data(), global_{}, sizeof(unsigned long long) * {}, cudaMemcpyDeviceToHost);'.format(counter, self.gctxt.num_warps))
                cc.add('\tprintf("krnl_{} global_{}:");'.format(self.id, counter))
                cc.add('\tfor (int i = 0; i < {}; ++i)'.format(self.gctxt.num_warps) + '{')
                cc.add('\t\tprintf(" %lld", cpu_counters[i]);')
                cc.add('\t}')
                cc.add('\tprintf("\\n");') 
        
        if KernelCall.args.lb:
            if KernelCall.args.lb_mode == 'morsel':
                pass
            else:
                if False:
                    cc.add(f'Themis::PrintStatisticsPerLvl(global_stats_per_lvl, "krnl_{self.id}", {len(self.spctxts)}, {self.gctxt.num_warps});')
        #cc.add(f'Themis::Heap::PrintMinHeaps(minheaps, {len(self.spctxts)});')

        if KernelCall.args.mode == 'sample':
            cc.add('cudaMemcpy(cpu_samples.data(), samples, sizeof(unsigned long long) * WIDTH_SAMPLES * {}, cudaMemcpyDeviceToHost);'
                .format(self.gctxt.num_warps))
            cc.add('for (int w = 0; w < {}; ++w) '.format(self.gctxt.num_warps) + ' {')
            
            cc.add('\tunsigned long long* d = (&cpu_samples[w * WIDTH_SAMPLES]) + 1;')
            cc.add('\tint num = d[-1];')
            cc.add('\tif (num == 0) continue;')
            cc.add('\tprintf("sample krnl_{} %d: ", w);'.format(self.id))
            cc.add('\tfor (int e = 0; e < num; ++e) {')
            cc.add('\t\tprintf(" %lld/%lld/%d/%lld", d[e] & 0x000FFFFFFFFFFFFF, d[e] >> 52 & 0xFF, w % 4, d[e] >> 60);')
            cc.add('\t}')
            cc.add('\tprintf("\\n");')
            cc.add('}')
            
        if True:
            cc.add('int numBlocks = 0;')
            cc.add(f'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, krnl_{self.id}, {KernelCall.defaultBlockSize}, 0);')
            cc.add(f'printf("krnl_{self.id} # blocks: %d\\n", numBlocks);')
            cc.add('cudaFuncAttributes attr;')
            cc.add(f'cudaFuncGetAttributes(&attr, krnl_{self.id});')
            cc.add('printf("Number of registers per thread: %d\\n", attr.numRegs);')
        cc.add(f'printf("krnl_{self.id} stop\\n");')
        cc.add('}')
        return c

    def addVar(self, ds):
        for var in self.vars:
            if var.name == ds.name:
                return
        self.vars.append(ds)
        self.gctxt.addVar(ds)



class SubPipeContext:

    def __init__(self, spid, pctxt, prev=None):

        self.id = spid
        self.pctxt = pctxt
        self.gctxt = pctxt.gctxt

        self.precode = Code()
        self.activecode = Code()
        self.postcode = Code()
        self.updatecode = Code()

        self.inReg = set([])
        self.declared = set([])

        self.prev = prev

        self.attrLoc = {}
        self.attrLoc.update(prev.attrOriginLoc)
        self.attrOriginLoc = {}
        self.attrOriginLoc.update(prev.attrOriginLoc)

        self.inReg.update(pctxt.inReg)
        self.declared.update(pctxt.declared)

        self.ts_type = None
        self.ts_num_ranges = 0
        self.ts_speculated_num_ranges = prev.ts_speculated_num_ranges
        self.ts_list_attributes = []
        self.ts_size_attributes = 0
        self.ts_speculated_size_attributes = prev.ts_speculated_size_attributes
        self.ts_tid = None
        self.ts_tid_build = None
        
        self.active_codes = []
        self.pre_codes = []


    def addVar(self, ds):
        self.pctxt.addVar(ds)
    
    def toString(self):
        c = Code()
        #c.addAll(self.precode.lines)
        
        for i, active_code in enumerate(self.active_codes):
            c.addAll(self.pre_codes[i].lines)
            if not active_code.isEmpty():
                
                if KernelCall.args.mode == 'profile':
                    c.add('#ifdef MODE_PROFILE')
                    c.add('{')
                    c.add('int num_active = __popc(__ballot_sync(ALL_LANES, active));')
                    c.add('if (num_active > 0) {')
                    c.add(f'active_lanes_num += num_active * {self.length};')
                    c.add(f'oracle_active_lanes_num += num_active > 0 ? 32 * {self.length} : 0;')
                    c.add('}')
                    c.add('}')
                    c.add('#endif')
                
                #endif
                
                c.add('if (active) { ' + '// spctxt[{}].toString()'.format(self.id))
                c.addAll(active_code.lines)
                c.add('}')

        c.addAll(self.precode.lines)
        if not self.activecode.isEmpty():
            if KernelCall.args.mode == 'profile':
                c.add('#ifdef MODE_PROFILE')
                c.add('{')
                c.add('int num_active = __popc(__ballot_sync(ALL_LANES, active));')
                c.add('if (num_active > 0) {')
                c.add(f'active_lanes_num += num_active * {self.length};')
                c.add(f'oracle_active_lanes_num += num_active > 0 ? 32 * {self.length} : 0;')
                c.add('}')
                c.add('}')
                c.add('#endif')
            c.add('if (active) { ' + '// last spctxt[{}].toString()'.format(self.id))
            #self.activecode.addTabs(1)
            c.addAll(list(map(lambda x: '\t' + x, self.activecode.lines)))
            c.add("}")
            
        c.addAll(self.postcode.lines)
        #c.addAll(self.updatecode.lines)
        return c

    def toSingleString(self):
        c = Code()
        for i, active_code in enumerate(self.active_codes):
            c.addAll(self.pre_codes[i].lines)
            if not active_code.isEmpty():
                c.add('if (active) { ' + '// spctxt[{}].toString()'.format(self.id))
                c.addAll(active_code.lines)
                c.add('}')

        c.addAll(self.precode.lines)
        if not self.activecode.isEmpty():
            c.add('if (active) { ' + '// last spctxt[{}].toString()'.format(self.id))
            #self.activecode.addTabs(1)
            c.addAll(list(map(lambda x: '\t' + x, self.activecode.lines)))
            #c.add("}")
            
        #c.addAll(self.postcode.lines)
        #c.addAll(self.updatecode.lines)
        return c


    def langType(self, a):
        return langType(a)


    def declare(self, attr):
        if attr.id in self.declared:
            return
        loc_dst = Location.reg(attr)
        self.precode.add("{} {};".format(langType(attr.dataType), loc_dst))
        self.declared.add(attr.id)
        return loc_dst


    def setLoc(self, attrId, loc):
        self.attrOriginLoc[attrId] = loc
        self.attrLoc[attrId] = loc
        return self.attrLoc[attrId]

    def toReg(self, attr):

        if attr.id in self.inReg:
            return self.attrLoc[attr.id]

        #print(attr.id, attr.name)
        loc = self.attrLoc[attr.id]
        for tid in loc.tids.values():
            #print('\t', tid.id, tid.name)
            self.toReg(tid)

        loc_dst = Location.reg(attr)
        if attr.id not in self.declared:
            self.precode.add("{} {};".format(langType(attr.dataType), loc_dst))
            self.declared.add(attr.id)
        
        self.attrLoc[attr.id] = loc_dst
        if str(loc) != str(loc_dst):
            self.activecode.add("{} = {};".format(loc_dst, loc))
        self.inReg.add(attr.id)
        return self.attrLoc[attr.id]


class themisCompiler:

    def __init__(self, dbpath, schema):
        self.dbpath = dbpath
        self.schema = schema
        self.num_warps = int(KernelCall.defaultGridSize * KernelCall.defaultBlockSize / 32)
        print('shit!!!!', self.num_warps)


    def genpipes(self, plan):

        info = {}
        info['pipelines'] = []

        for node in plan:
            info['pipelines'].append([[]])
            node.genPipes(info)

        
        p_pipes = []

        for pid, pipe in enumerate(info['pipelines']):
            p_pipes.append([])
            #print("pipe {}".format(pid))
            prev_pop = None
            for spid, subpipe in enumerate(pipe):
                #print("subpipe {}-{}".format(pid, spid))
                #pp.pprint(subpipe)
                p_pipes[-1].append([])
                for opid, op in enumerate(subpipe):
                    pop = None
                    if op.opType == "indexjoin":
                        pop = IndexJoin(op)
                    elif op.opType == "exist":
                        pop = Exist(op)
                    elif op.opType == "scan":
                        pop = Scan(op)
                    elif op.opType == "selection":
                        pop = Selection(op)
                    elif op.opType == "map":
                        pop = Map(op)
                    elif op.opType == 'multimap':
                        pop = MultiMap(op)
                    elif op.opType == "equijoin":
                        pop = EquiJoin(op)
                    elif op.opType == "crossjoin":
                        pop = CrossJoin(op)
                    elif op.opType == "aggregation":
                        pop = Aggregation(op)
                    elif op.opType == "materialize":
                        pop = Materialize(op)
                    elif op.opType == 'triescan':
                        pop = TrieScan(op)
                    elif op.opType == 'intersection':
                        pop = Intersection(op)
                    elif op.opType == 'intersectionselection':
                        pop = IntersectionSelection(op)
                    p_pipes[-1][-1].append(pop)
                    pop.prev = prev_pop
                    prev_pop = pop
        
        
        for pid, pipe in enumerate(p_pipes):
            print('pid', pid)
            for spid, subpipe in enumerate(pipe):
                print('\tspid')
                for op in subpipe:
                    print('\t\t{}'.format(op))
        return p_pipes

    def gencode(self, pipes):

        gctxt = GlobalContext(self.num_warps)
        self.gctxt = gctxt
        gctxt.dbpath = self.dbpath
        gctxt.schema = self.schema

        for pid, pipe in enumerate(pipes):
            print("pipe {}".format(pid))
            pctxt = PipeContext(pid, gctxt)
            # do scan..
            spctxt = SubPipeContext(0, pctxt, pctxt)
            spctxt.loop_lvl = 0

            pipe[0][0].genScanLoop(spctxt)
            pipe[0][0].genPopKnownNodes(spctxt)
            prev_push = False
            pctxt.num_loops = 0
            loop_lvl = 0

            for spid, subpipe in enumerate(pipe):
                if prev_push:
                    spctxt = SubPipeContext(spctxt.id+1, pctxt, spctxt)
                    spctxt.loop_lvl = loop_lvl
                    subpipe[0].genPopKnownNodes(spctxt)
                    prev_push = False

                print("Generate loading... ")
                for opid, op in enumerate(subpipe[:len(subpipe)]):
                    print(op.algExpr)
                    for attrId, attr in op.algExpr.outRelation.items():
                        pass
                        #if attrId in spctxt.attrLoc:
                        #    spctxt.toReg(attr)
                        #    print('Y ', attrId, attr)        
                        #else:
                        #    print('N ', attrId, attr)        
                for op in subpipe[:len(subpipe)-1]:            
                    op.genOperation(spctxt)
                spctxt.length = len(subpipe) if spid != 0 else len(subpipe) - 1

                if spid + 1 < len(pipe):
                    subpipe[-1].genOperation(spctxt)
                    if 'doLB' not in subpipe[-1].algExpr.__dict__ or subpipe[-1].algExpr.doLB == True:
                        subpipe[-1].genPushTrapzoids(spctxt)
                        if spctxt.ts_type == 1:
                            loop_lvl = spid + 1
                        prev_push = True
                else:
                    subpipe[-1].genMaterialize(spctxt)
                    
                    if KernelCall.args.switch:
                        pass
                    else:
                        loop_lvl = spctxt.loop_lvl
                        spctxt.postcode.add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
                        spctxt.postcode.add(f'else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
                    
                    prev_push = True
                    
                if prev_push: 
                    spcode = spctxt.toString()
                    if KernelCall.args.switch:
                        pctxt.switchcode.add(f'case {spctxt.id}:')
                    else:
                        if spctxt.id == 0 or pctxt.spctxts[-1].ts_type == 1:
                            pctxt.num_loops += 1
                            if KernelCall.args.lb:
                                pctxt.switchcode.add(f'while (lvl >= {spctxt.id} && loop < {KernelCall.args.lb_push_period})' + '{ // nested loop based code generation')
                            else:
                                pctxt.switchcode.add(f'while (lvl >= {spctxt.id})' + "{")
                            pctxt.switchcode.add('__syncwarp();')
                        pctxt.switchcode.add(f'if (lvl == {spctxt.id})')
                    pctxt.switchcode.add('{')
                    if KernelCall.args.lb:
                        pctxt.switchcode.add('++loop;')
                    pctxt.switchcode.addAll(spcode.lines)
                    pctxt.switchcode.add('};')
                    pctxt.update(spctxt)
                
                spctxt.active_codes.append(spctxt.activecode)
                spctxt.activecode = Code()
                spctxt.pre_codes.append(spctxt.precode)
                spctxt.precode = Code()
            gctxt.update(pctxt)
        self.codestring = self.gctxt.toCode().toString()

    def genPyperCode(self, pipes):
        gctxt = GlobalContext(self.num_warps)
        self.gctxt = gctxt
        gctxt.dbpath = self.dbpath
        gctxt.schema = self.schema
        for pid, pipe in enumerate(pipes):
            #print("pipe {}".format(pid))
            pctxt = PipeContext(pid, gctxt)
            # do scan..
            spctxt = SubPipeContext(0, pctxt, pctxt)
            pipe[0][0].genScan(spctxt)
            
            c = spctxt.pctxt.precode      
            
            prev_push = False
            prev_expand = False
            num_loops = 0
            for spid, subpipe in enumerate(pipe):

                if prev_expand:

                    prev_expand = True
                    pass

                if prev_push:
                    spctxt = SubPipeContext(spctxt.id+1, pctxt, spctxt)
                    spctxt.inReg.update(spctxt.prev.inReg)
                    spctxt.declared.update(spctxt.prev.declared)
                    prev_push = False

                for op in subpipe[:len(subpipe)-1]:
                    op.genOperation(spctxt)
                    
                spctxt.length = len(subpipe) if spid != 0 else len(subpipe) - 1

                if spid + 1 < len(pipe):
                    
                    subpipe[-1].genOperation(spctxt)
                    if subpipe[-1].doExpansion():
                        spctxt.ts_tid = subpipe[-1].algExpr.tid
                        spctxt.ts_tid_build = "loopvar{}".format(spctxt.id + 1)
                        spctxt.activecode.add('}')
                        spctxt.activecode.add('int loopvar{} = active ? local{}_range.start-1 : local{}_range.end;'.format(spctxt.id+1, subpipe[-1].algExpr.opId,subpipe[-1].algExpr.opId))
                        
                        attrToMaterialize = subpipe[-1].genAttrToMaterialize(spctxt)
                        for attrId, attr in attrToMaterialize.items():
                            #print(attr.id_name)
                            #print(attr)
                            #print(spctxt.ts_tid.id)
                            #print(attr.tid)
                            if spctxt.ts_tid != None and attr.tid != None and attr.tid != -1 and (spctxt.ts_tid.id == attrId or spctxt.ts_tid.id == attr.tid.id):
                                continue
                            spctxt.declare(attr)
                            #spctxt.toReg(attr)
                            if attr.dataType == Type.STRING:
                                spctxt.activecode.add(
                                    'str_t buf{}_{};'.format(spctxt.id+1, attr.id_name))
                            else:
                                spctxt.activecode.add(
                                    '{} buf{}_{};'.format(langType(attr.dataType), spctxt.id+1, attr.id_name))
                        spctxt.activecode.add('if (active) {')
                        
                        for attrId, attr in attrToMaterialize.items():
                            if spctxt.ts_tid != None and attr.tid != None and attr.tid != -1 and (spctxt.ts_tid.id == attrId or spctxt.ts_tid.id == attr.tid.id):
                                continue
                            spctxt.toReg(attr)
                            if KernelCall.args.use_pos_vec == False:
                                spctxt.pctxt.materialized.add(attr.id)
                                spctxt.attrOriginLoc[attr.id] = Location.reg(attr)
                            #spctxt.activecode.add('//{} {} {} {}'.format(attr.id_name, spctxt.ts_tid.id, attrId, attr.tid.id))
                            if attr.dataType == Type.STRING:
                                spctxt.activecode.add(
                                    'buf{}_{} = {};'.format(spctxt.id+1, attr.id_name, attr.id_name))
                            else:
                                spctxt.activecode.add(
                                    'buf{}_{} = {};'.format(spctxt.id+1, attr.id_name, attr.id_name))
                        spctxt.activecode.add('}')
                        spctxt.activecode.add(
                            'while (__syncthreads_count(++loopvar{} < local{}_range.end || keepGoing))'.format(spctxt.id+1, subpipe[-1].algExpr.opId)
                            + '{')
                        spctxt.activecode.add('active = (!keepGoing) && (loopvar{} < local{}_range.end);'.format(spctxt.id+1, subpipe[-1].algExpr.opId))
                        spctxt.activecode.add('if (active) {')
                        
                        attrToMaterialize = subpipe[-1].genAttrToMaterialize(spctxt)
                        for attrId, attr in attrToMaterialize.items():
                            if spctxt.ts_tid != None and attr.tid != None and attr.tid != -1 and (spctxt.ts_tid.id == attrId or spctxt.ts_tid.id == attr.tid.id):
                                continue
                            spctxt.activecode.add(
                                '{} = buf{}_{};'.format(attr.id_name, spctxt.id+1, attr.id_name))
                        
                                        
                        #spctxt.activecode.add('atomicAdd(cnt+{}, 1);'.format(spctxt.id+1))
                        spctxt.declare(subpipe[-1].algExpr.tid)
                        tidLoc = spctxt.attrLoc[subpipe[-1].algExpr.tid.id]
                        spctxt.activecode.add("{} = {};".format(tidLoc, subpipe[-1].genTidBuild(spctxt)))
                        #spctxt.activecode.add('atomicAdd(cnt+{}, {});'.format(spctxt.id+1, tidLoc))
                        if isinstance(subpipe[-1], CrossJoin) == False and subpipe[-1].algExpr.doShuffle:
                            spctxt.activecode.add('if (active) {')
                            for attrId, attr in subpipe[-1].algExpr.outRelation.items():
                                spctxt.activecode.add('//' + str(attrId) + ' ' + str(attr))
                                spctxt.toReg(attr)
                                if True or KernelCall.args.use_pos_vec == False:
                                    spctxt.pctxt.materialized.add(attrId)
                                    spctxt.attrOriginLoc[attr.id] = Location.reg(attr)
                            spctxt.activecode.add('}')
                            subpipe[-1].genShuffle(spctxt, True)
                        
                        prev_push = True
                        num_loops += 1
                    elif 'doLB' not in subpipe[-1].algExpr.__dict__ or subpipe[-1].algExpr.doLB == True:
                        spctxt.activecode.add('if (active) {')
                        for attrId, attr in subpipe[-1].algExpr.outRelation.items():
                            spctxt.toReg(attr)
                            if KernelCall.args.use_pos_vec == False:
                                spctxt.pctxt.materialized.add(attrId)
                                spctxt.attrOriginLoc[attr.id] = Location.reg(attr)
                                
                        
                        spctxt.activecode.add('}')
                        subpipe[-1].genShuffle(spctxt)
                        prev_push = True
                else:
                    subpipe[-1].genMaterialize(spctxt)
                    prev_push = True
                    
                if prev_push:
                    spcode = spctxt.toString()
                    pctxt.switchcode.addAll(spcode.lines)
                    pctxt.update(spctxt)
                
                spctxt.active_codes.append(spctxt.activecode)
                spctxt.activecode = Code()
                spctxt.pre_codes.append(spctxt.precode)
                spctxt.precode = Code()

            for i in range(num_loops):
                pctxt.switchcode.add('keepGoing = false;')
                pctxt.switchcode.add('}' + '// loop {}'.format(i))
            #pctxt.switchcode.add('}')

            gctxt.update(pctxt)
        self.codestring = self.gctxt.toPyperCode().toString()

    def genSingleCode(self, pipes):
        gctxt = GlobalContext(self.num_warps)
        self.gctxt = gctxt
        gctxt.dbpath = self.dbpath
        gctxt.schema = self.schema
        for pid, pipe in enumerate(pipes):
            pctxt = PipeContext(pid, gctxt)
            # do scan..
            spctxt = SubPipeContext(0, pctxt, pctxt)
            pipe[0][0].genSingleScan(spctxt)
            prev_push = False
            for spid, subpipe in enumerate(pipe):
                if prev_push:
                    spctxt = SubPipeContext(spctxt.id+1, pctxt, spctxt)
                    spctxt.inReg.update(spctxt.prev.inReg)
                    spctxt.declared.update(spctxt.prev.declared)
                    prev_push = False

                for op in subpipe[:len(subpipe)-1]:
                    op.genOperation(spctxt)

                if spid + 1 < len(pipe):
                    subpipe[-1].genOperation(spctxt)
                    if subpipe[-1].doExpansion():
                        spctxt.activecode.add('for (int loopvar{} = local{}_range.start; loopvar{} < local{}_range.end; ++loopvar{})'
                            .format(spctxt.id+1, subpipe[-1].algExpr.opId, spctxt.id+1, subpipe[-1].algExpr.opId, spctxt.id+1) + '{')
                        spctxt.activecode.add('active = true;')
                        spctxt.declare(subpipe[-1].algExpr.tid)
                        tidLoc = spctxt.attrLoc[subpipe[-1].algExpr.tid.id]
                        spctxt.activecode.add("{} = {};".format(tidLoc, subpipe[-1].genTidBuild(spctxt)))
                        prev_push = True
                else:
                    subpipe[-1].genMaterialize(spctxt)
                    prev_push = True

                if prev_push:
                    spcode = spctxt.toSingleString()
                    pctxt.switchcode.addAll(spcode.lines)
                    pctxt.update(spctxt)

                spctxt.active_codes.append(spctxt.activecode)
                spctxt.activecode = Code()
                spctxt.pre_codes.append(spctxt.precode)
                spctxt.precode = Code()
            
            for i in range(spctxt.id*2):
                pctxt.switchcode.add('}' + '// spctxt {}'.format(i))

            pctxt.switchcode.add('}')
            gctxt.update(pctxt)
        self.codestring = self.gctxt.toSingleCode().toString()


    def writeCodeFile ( self, code, filename ):
        with open(filename, 'w') as f:
            f.write( code )
        # format sourcecode
        cmd = "astyle --indent-col1-comments " + filename
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)

    def compile( self, filename, compileOption, arch="sm_75", debug=False ):
        print("compilation...")
        sys.stdout.flush()

        self.filename = filename
        cuFilename = filename + ".cu"

        self.writeCodeFile (self.codestring, cuFilename )        

        # compile
        
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


    def execute(self, filename, deviceid=None, timeout=None):
        print("\nexecution...")
        sys.stdout.flush()
        cmd = filename
        #output = subprocess.check_output(cmd, shell=True, timeout=timeout).decode('utf-8')
        try:
            if deviceid != None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceid)
            proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate(timeout=timeout)
            out, err = out.decode('utf-8'), err.decode('utf-8')
            print(out)
            print(err)
            with open(filename + ".log", "w") as log_file:
                print(out, file=log_file)
            return (out)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            print('cmd: {}'.format(e.cmd))
            print('output:\n{}'.format(e.output))
            print('timeout: {}'.format(e.timeout))
            with open(self.filename + ".log", "w") as log_file:
                print(e.output, file=log_file)
                print('totalKernelTime: {}'.format(timeout*1000), file=log_file)

            time.sleep(3)
            
