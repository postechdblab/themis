



alg.aggregation (
    [ "c_name", "c_custkey", "o1.o_orderkey", "o1.o_orderdate", "o1.o_totalprice" ],
    [ ( Reduction.SUM, "l2.l_quantity", "sum_qty" ) ], 543182, True,
    alg.indexjoin (
        scal.AttrExpr("o1.tid"), 'orders', ["o_orderkey"], "lineitem", "l2", ["l_orderkey"], None, False,
        alg.indexjoin(
            scal.AttrExpr("o1.tid"), 'orders', ["o_custkey"], "customer", None, ["c_custkey"], None, True,
            alg.join (
                [("o1.o_orderkey", "l1.l_orderkey")], None, False, True,
                alg.selection (
                    scal.LargerExpr (
                        scal.AttrExpr ( "sum_qty" ),
                        scal.ConstExpr ( "812.0f", Type.FLOAT )
                    ), 75000000 / 10, True,
                    alg.aggregation ( 
                        [ "l1.l_orderkey" ],
                        [ ( Reduction.SUM, "l1.l_quantity", "sum_qty" ) ], 75000000, True,
                        alg.selection (
                            scal.SmallerEqualExpr (
                                scal.AttrExpr ("l1.l_quantity"),
                                scal.ConstExpr("100.0f", Type.DOUBLE)
                            ), None, True,
                            alg.scan( "lineitem", "l1")
                        )
                    )
                ),
                alg.scan("orders", "o1"),
            )
        ), True,
    )
)
