alg.projection (
    [ "promo_revenue" ],
    alg.map (
        "promo_revenue",
        scal.DivExpr (
            scal.MulExpr (
                scal.ConstExpr ( "100.0f", Type.DOUBLE ),
                scal.AttrExpr ( "sum_promo" )
            ),
            scal.AttrExpr ( "sum_rev" )
        ),
        alg.aggregation (
            [],
            [ ( Reduction.SUM, "rev", "sum_rev" ), ( Reduction.SUM, "promo", "sum_promo" ) ], 1, False,
            alg.map (
                "promo",
                scal.CaseExpr (
                    [
                        (
                            scal.LikeExpr ( 
                                scal.AttrExpr ( "p_type" ),
                                scal.ConstExpr ( "PROMO%", Type.STRING )
                            ),
                            scal.AttrExpr("rev"),
                        )
                    ],
                    scal.ConstExpr ( "0.0f", Type.INT )
                ),
                alg.map (
                    "rev",
                    scal.MulExpr ( 
                        scal.AttrExpr ( "l_extendedprice" ),
                        scal.SubExpr ( 
                            scal.ConstExpr ( "1.0f", Type.FLOAT ),
                            scal.AttrExpr ( "l_discount" )
                        )
                    ),
                    alg.indexjoin (
                        scal.AttrExpr("lineitem.tid"), 'lineitem', ["l_partkey"],
                        "part", None, ["p_partkey"], None, True,
                        alg.selection ( 
                            scal.LargerEqualExpr (
                                scal.AttrExpr ( "l_shipdate" ),
                                scal.ConstExpr ( "19940801", Type.DATE )
                            ), None, True,
                            alg.selection(
                                scal.SmallerExpr (
                                    scal.AttrExpr ( "l_shipdate" ),
                                    scal.ConstExpr ( "19940901", Type.DATE )
                                ), None, True,
                                #alg.multimap(
                                #    [("lshipdate", scal.AttrExpr("l_shipdate")),("leprice", scal.AttrExpr("l_extendedprice")),("ldiscount", scal.AttrExpr("l_discount"))], 
                                alg.scan("lineitem")
                                #)
                            )
                        )
                    )
                )
            )
        )
    )
)
