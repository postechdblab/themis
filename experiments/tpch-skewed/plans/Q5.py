

alg.aggregation ( 
    [ "n_name" ],
    [ ( Reduction.SUM, "revenue", "sum_rev" ) ],
    128, False,
    alg.map (
        "revenue",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.SubExpr ( 
                scal.ConstExpr ( "1.0f", Type.FLOAT ),
                scal.AttrExpr ( "l_discount" )
            )
        ),
        alg.join (
            [ ( "l_suppkey", "s_suppkey" ), ( "s_nationkey", "c_nationkey" ) ],
            None, False, True,
            alg.scan ( "supplier" ),
            alg.indexjoin (
                scal.AttrExpr("orders.tid"), 'orders', ["o_orderkey"],
                "lineitem", None, ["l_orderkey"], None, False,
                alg.selection (
                    scal.AndExpr (
                        scal.LargerEqualExpr (
                            scal.AttrExpr ( "o_orderdate" ),
                            scal.ConstExpr ( "19930101", Type.DATE )
                        ),
                        scal.SmallerExpr (
                            scal.AttrExpr ( "o_orderdate" ),
                            scal.ConstExpr ( "19940101", Type.DATE )
                        )
                    ), None, True,
                    alg.indexjoin (
                        scal.AttrExpr("customer.tid"), 'customer', ["c_custkey"],
                        "orders", None, ["o_custkey"], None, False,
                        alg.indexjoin (
                            scal.AttrExpr("nation.tid"), 'nation', ["n_nationkey"],
                            "customer", None, ["c_nationkey"], None, False,
                            alg.indexjoin (
                                scal.AttrExpr("region.tid"), 'region', ["r_regionkey"],
                                "nation", None, ["n_regionkey"], None, False,
                                alg.selection ( 
                                    scal.EqualsExpr (
                                        scal.AttrExpr ( "r_name" ),
                                        scal.ConstExpr ( "AMERICA", Type.STRING )
                                    ), None, False,
                                    alg.scan ( "region" )
                                )
                            ), True
                        ), True
                    )
                ), True
            )
        )
    )
)
