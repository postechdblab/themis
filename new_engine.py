import sys
sys.path.insert(0,"../../")
sys.path.insert(0,"../")
sys.path.insert(0,'.')
import os
import importlib
import argparse
import sqlite3
import subprocess
import time
import json
from datetime import datetime
import dogqc.dbio as io
from schema import schema
from dogqc.util import loadScript

# algebra types
from dogqc.relationalAlgebra import Context
from dogqc.relationalAlgebra import RelationalAlgebra
from dogqc.relationalAlgebra import Reduction
from dogqc.cudaTranslator import CudaCompiler
from dogqc.types import Type
from dogqc.cudalang import CType
import dogqc.scalarAlgebra as scal
from dogqc.kernel import KernelCall
from dogqc.hashJoins import EquiJoinTranslator 

from themis.pipes2cuda import themisCompiler
from themis.plan2pipes import genPipesForPlan
from themis.pipes2naive import genCode as genNaiveCode
from themis.pipes2naive import compileCode as compileNaiveCode
from themis.pipes2themis import genCode as genThemisCode
from themis.pipes2themis import compileCode as compileThemisCode
from themis.pipes2dogqc import genCode as genDogqcCode
from themis.pipes2dogqc import compileCode as compileDogqcCode
from themis.pipes2pyper import genCode as genPyperCode
from themis.pipes2pyper import compileCode as compilePyperCode

def str2bool(v): 
    if isinstance(v, bool): return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False 
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def build():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dummy", type=str)
    parser.add_argument("--dbpath", type=str, help='location for the database to build')
    parser.add_argument("--csvpath", type=str, help='location of the data')
    args = parser.parse_args()

    io.dbBuild (schema, args.dbpath, args.csvpath)


def query():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("dummy", type=str)

    parser.add_argument("--sqlite3_path", type=str, default='', help='database to store results' )
    parser.add_argument("--exp_id", type=str, default='test', help='experiment id' )
    parser.add_argument("--mode", type=str, default='timecheck', help='mode timecheck / profile / debug / sample / stats / fake')
    parser.add_argument("--timeout", type=int, default=10800)

    parser.add_argument("--dbpath", type=str, help='location of the database')
    parser.add_argument("--qpath", type=str, help='location of the query to run')
    parser.add_argument("--system", type=str, help='system to run: dogqc / themis')
    parser.add_argument("--resultpath", type=str, help='path for result' )

    parser.add_argument("--gridsize", type=int, default=3280, help='# of thread blocks')
    parser.add_argument("--blocksize", type=int, default=128, help='size of a thread block')
    parser.add_argument("--device", type=int, default=0, )
    parser.add_argument("--min_num_warps", default=3936)
    parser.add_argument("--sm_num", type=int, default=82)
    parser.add_argument("--local_aggregation", type=str2bool, default=False)
    parser.add_argument("--lazy_materialization", type=str2bool, default=False)
    
    parser.add_argument("--inter_warp_lb", type=str2bool, default=False)
    parser.add_argument("--inter_warp_lb_method", default="aws", help='aws / ws')
    parser.add_argument("--inter_warp_lb_interval", default=32)
    parser.add_argument("--inter_warp_lb_detection_method", default="twolvlbitmaps", help='twolvlbitmaps / idqueue')
    parser.add_argument("--inter_warp_lb_ws_threshold", default=1024, help='')

    parser.add_argument("--pyper_grid_threshold", type=int, default=24)
    
    args = parser.parse_args()
    KernelCall.args = args

    KernelCall.counters = ['active_clock', 'active_lanes_num', 'oracle_active_lanes_num']
    KernelCall.args.num_counters = {}
    KernelCall.args.isOldCompiler = False
    
    f = open(args.resultpath + '.log', 'w')
    for k, v in vars(args).items():
        f.write('\n{}: {}'.format(k, v))
    f.write('Start query execution...')
    f.close()

    if args.mode != 'fake':
        acc = io.dbAccess (args.dbpath)
        alg = RelationalAlgebra ( acc )
        plan = eval (loadScript (args.qpath) )

        cfgpath = args.qpath.replace(".py", "cfg.py")
        cfg = {}
        plan = alg.resolveAlgebraPlan ( plan, cfg )

        KernelCall.args.schema = acc.schema
        KernelCall.defaultBlockSize = args.blocksize
        KernelCall.defaultGridSize = args.gridsize        
        compileOption = ' '
        result = ''
        try:
            os.remove(args.resultpath)
        except:
            pass
        if args.system == 'DogQC++':
            dss, pipes = genPipesForPlan(plan)
            code = genDogqcCode(dss, pipes)
            compileDogqcCode(args.resultpath, code, compileOption, arch="sm_75")
        elif args.system == 'Themis':
            dss, pipes = genPipesForPlan(plan)            
            code = genThemisCode(dss, pipes)
            compileThemisCode(args.resultpath, code, compileOption, arch="sm_75")
        elif args.system == 'Pyper':
            dss, pipes = genPipesForPlan(plan)            
            code = genPyperCode(dss, pipes)
            compilePyperCode(args.resultpath, code, compileOption, arch="sm_75")
        elif args.system == 'Single':
            KernelCall.defaultBlockSize = 1
            KernelCall.defaultGridSize = 1
            KernelCall.args.intra_warp_lb = False
            dss, pipes = genPipesForPlan(plan)
            code = genNaiveCode(dss, pipes)
            compileNaiveCode(args.resultpath, code, compileOption, arch="sm_75")


def build_sqlite3():
    sqlite3_path = sys.argv[2]
    conn = sqlite3.connect(sqlite3_path, isolation_level=None)
    c = conn.cursor()
    c.execute("CREATE TABLE logs (id integer PRIMARY KEY, t integer, exp integer, mode varchar, config text, code text, result text);")
    conn.commit()    
    pass

if sys.argv[1] == 'build':
    build()
elif sys.argv[1] == 'query':
    query()
elif sys.argv[1] == 'build_sqlite3':
    build_sqlite3()
