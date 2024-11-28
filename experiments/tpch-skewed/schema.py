import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')

from collections import OrderedDict
from dogqc.types import Type 

schema = {}

schema["dateformat"] = ("int toDate ( const char* c ) {"
                            "    int d=0;"
                            "    d += (int)( c[0] - 48 ) * 10000000;"
                            "    d += (int)( c[1] - 48 ) *  1000000;"
                            "    d += (int)( c[2] - 48 ) *   100000;"
                            "    d += (int)( c[3] - 48 ) *    10000;"
                            "    d += (int)( c[5] - 48 ) *     1000;"
                            "    d += (int)( c[6] - 48 ) *      100;"
                            "    d += (int)( c[8] - 48 ) *       10;"
                            "    d += (int)( c[9] - 48 ) *        1;"
                            "    return d;"
                            "}")


schema["lineitem"] = { "name":"lineitem", "size":0, "numColumns":16, "attributes": 
                             OrderedDict([
                                 ("l_orderkey",Type.INT),
                                 ("l_partkey",Type.INT),
                                 ("l_suppkey",Type.INT),
                                 ("l_linenumber",Type.INT),
                                 ("l_quantity",Type.INT),
                                 ("l_extendedprice",Type.FLOAT),
                                 ("l_discount",Type.FLOAT),
                                 ("l_tax",Type.FLOAT),
                                 ("l_returnflag",Type.CHAR),
                                 ("l_linestatus",Type.CHAR),
                                 ("l_shipdate",Type.DATE),
                                 ("l_commitdate",Type.DATE),
                                 ("l_receiptdate",Type.DATE),
                                 ("l_shipinstruct",Type.STRING),
                                 ("l_shipmode",Type.STRING),
                                 ("l_comment",Type.STRING)
                             ]),
                             "relationships": [
                                 ("orders", [["l_orderkey"], ["o_orderkey"]] ),
                                 ("part", [["l_partkey"], ["p_partkey"]]),
                                 ("supplier", [["l_suppkey"], ["s_suppkey"]]),
                                 ("partsupp", [["l_partkey", "l_suppkey"], ["ps_partkey", "ps_suppkey"]])
                             ], "indexes": [
                                 ["l_orderkey"],
                                 ["l_partkey", "l_suppkey"],
                                 ["l_suppkey", "l_partkey"]
                             ]
                         }

schema["customer"] = { "name":"customer", "size":0, "numColumns":8, "attributes": 
                             OrderedDict([
                                 ("c_custkey",Type.INT),
                                 ("c_name",Type.STRING),
                                 ("c_address",Type.STRING),
                                 ("c_nationkey",Type.INT),
                                 ("c_phone",Type.STRING),
                                 ("c_acctbal",Type.FLOAT),
                                 ("c_mktsegment",Type.STRING),
                                 ("c_comment",Type.STRING)
                             ]),
                             "relationships": [
                                 ("orders", [["c_custkey"], ["o_custkey"]]),
                                 ("nation", [["c_nationkey"], ["n_nationkey"]])
                             ], "indexes": [["c_custkey"],["c_nationkey"]]
                         }

schema["orders"] = { "name":"orders", "size":0, "numColumns":9, "attributes": 
                           OrderedDict([
                               ("o_orderkey",Type.INT),
                               ("o_custkey",Type.INT),
                               ("o_orderstatus",Type.CHAR),
                               ("o_totalprice",Type.FLOAT),
                               ("o_orderdate",Type.DATE),
                               ("o_orderpriority",Type.STRING),
                               ("o_clerk",Type.STRING),
                               ("o_shippriority",Type.INT),
                               ("o_comment",Type.STRING)
                           ]),
                           "relationships": [
                               ("lineitem", [["o_orderkey"], ["l_orderkey"]]),
                               ("customer", [["o_custkey"], ["c_custkey"]])
                           ], "indexes": [["o_orderkey"],["o_custkey"]]
                       }

schema["partsupp"] = { "name":"partsupp", "size":0, "numColumns":5, "attributes": 
                             OrderedDict([
                                 ("ps_partkey",Type.INT),
                                 ("ps_suppkey",Type.INT),
                                 ("ps_availqty",Type.INT),
                                 ("ps_supplycost",Type.FLOAT),
                                 ("ps_comment",Type.STRING)
                             ]),
                             "relationships": [
                                 ("lineitem", [["ps_partkey", "ps_suppkey"], ["l_partkey", "l_suppkey"]]),
                                 ("part", [["ps_partkey"], ["p_partkey"]]),
                                 ("supplier", [["ps_suppkey"], ["s_suppkey"]])
                             ], 
                             "indexes": [["ps_partkey", "ps_suppkey"],["ps_suppkey", "ps_partkey"]]
                         }

schema["part"] = { "name":"part", "size":0, "numColumns":9, "attributes": 
                         OrderedDict([
                             ("p_partkey",Type.INT),
                             ("p_name",Type.STRING),
                             ("p_mfgr",Type.STRING),
                             ("p_brand",Type.STRING),
                             ("p_type",Type.STRING),
                             ("p_size",Type.INT),
                             ("p_container",Type.STRING),
                             ("p_retailprice",Type.FLOAT),
                             ("p_comment",Type.STRING)
                         ]),
                         "relationships": [
                             ("lineitem", [["p_partkey"], ["l_partkey"]]),
                             ("partsupp", [["p_partkey"], ["ps_partkey"]])
                         ], 
                         "indexes": [["p_partkey"]]
                     }

schema["supplier"] = { "name":"supplier", "size":0, "numColumns":7, "attributes": 
                             OrderedDict([
                                 ("s_suppkey",Type.INT),
                                 ("s_name",Type.STRING),
                                 ("s_address",Type.STRING),
                                 ("s_nationkey",Type.INT),
                                 ("s_phone",Type.STRING),
                                 ("s_acctbal",Type.FLOAT),
                                 ("s_comment",Type.STRING)
                             ]),
                             "relationships": [
                                ("lineitem", [["s_suppkey"], ["l_suppkey"]]),
                                ("partsupp", [["s_suppkey"], ["ps_suppkey"]]),
                                ("nation", [["s_nationkey"], ["n_nationkey"]])
                             ], 
                             "indexes": [["s_nationkey"],["s_suppkey"]]
                             
                         }

schema["nation"] = { "name":"nation", "size":0, "numColumns":4, "attributes": 
                           OrderedDict([
                               ("n_nationkey",Type.INT),
                               ("n_name",Type.STRING),
                               ("n_regionkey",Type.INT),
                               ("n_comment",Type.STRING)
                           ]),
                           "relationships": [
                               ("region", [["n_regionkey"], ["r_regionkey"]]),
                               ("customer", [["n_nationkey"], ["c_nationkey"]]),
                               ("supplier", [["n_nationkey"], ["s_nationkey"]])
                           ], "indexes": [["n_regionkey"],["n_nationkey"]]
                       }

schema["region"] = { "name":"region", "size":0, "numColumns":3, "attributes": 
                           OrderedDict([
                               ("r_regionkey",Type.INT),
                               ("r_name",Type.STRING),
                               ("r_comment",Type.STRING)
                           ]),
                           "relationships": [
                               ("nation", [["r_regionkey"], ["n_regionkey"]])
                           ], "indexes": [["r_regionkey"]]
                       }


schema
