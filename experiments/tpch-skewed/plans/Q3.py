alg.aggregation (
    [ "l_orderkey", "oorderdate", "oshippriority" ],
    [ ( Reduction.SUM, "revenue", "sum_rev" ) ],
    2 * 1024 * 1024, True, # 549,618 / 16,020,869
    alg.map (
        "revenue",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.SubExpr ( 
                scal.ConstExpr ( "1.0f", Type.FLOAT ),
                scal.AttrExpr ( "l_discount" )
            )
        ),
        alg.selection(
            scal.LargerExpr (
                scal.AttrExpr ( "l_shipdate" ),
                scal.ConstExpr ( "19930531", Type.DATE )
            ), None, True,
            alg.indexjoin(
                scal.AttrExpr("orders.tid"), 'orders', ["o_orderkey"],
                "lineitem", None, ["l_orderkey"], None, False,
                alg.multimap([["oorderdate", scal.AttrExpr("o_orderdate")], ["oshippriority", scal.AttrExpr("o_shippriority")]],
                    alg.selection (
                        scal.SmallerExpr (
                            scal.AttrExpr ( "o_orderdate" ),
                            scal.ConstExpr ( "19930531", Type.DATE )
                        ), None, True,
                        alg.indexjoin(
                            scal.AttrExpr("customer.tid"), 'customer', ["c_custkey"],
                            "orders", None, ["o_custkey"], None, False,
                            alg.selection (
                                scal.EqualsExpr (
                                    scal.AttrExpr ( "c_mktsegment" ),
                                    scal.ConstExpr ( "HOUSEHOLD", Type.STRING )
                                ), None, True,
                                alg.scan ( "customer" )
                            ), True
                        )
                    )
                ), True
            )
        )
    )
)
