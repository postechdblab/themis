alg.projection (
    [ "avg_yearly" ],
    alg.map (
        "avg_yearly",
        scal.DivExpr ( 
            scal.AttrExpr ( "sum_price" ),
            scal.ConstExpr ( "7.0f", Type.DOUBLE )
        ),
        alg.aggregation (
            [],
            [ ( Reduction.SUM, "l1.l_extendedprice", "sum_price" ) ], 1, False,
            alg.selection (
                scal.SmallerExpr (
                    scal.AttrExpr ( "l1.l_quantity" ),
                    scal.AttrExpr ( "lim_quan" )
                ), None, True,
                alg.indexjoin (
                    scal.AttrExpr("part.tid"), 'part', ["p_partkey"],
                    "lineitem", "l1", ["l_partkey"], None, False,
                    alg.map (
                        "lim_quan",
                        scal.MulExpr (
                            scal.AttrExpr ( "avg_quan" ),
                            scal.ConstExpr ( "0.2f", Type.DOUBLE )
                        ),
                        alg.aggregation (
                            [ "part.tid" ],
                            [ ( Reduction.AVG, "l2.l_quantity", "avg_quan" ) ], 4 * 291016, True,
                            alg.indexjoin (
                                scal.AttrExpr("part.tid"), 'part', ["p_partkey"],
                                "lineitem", "l2", ["l_partkey"], None, False,
                                alg.selection (
                                    scal.AndExpr(
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "pbrand" ),
                                            scal.ConstExpr ( "Brand#55", Type.STRING )
                                        ),
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "pcontainer" ),
                                            scal.ConstExpr ( "LG BOX", Type.STRING )
                                        ), 
                                    ), None, True,
                                    alg.multimap([["pbrand", scal.AttrExpr("p_brand")], ["pcontainer", scal.AttrExpr("p_container")]],
                                        alg.scan("part")
                                    )
                                ), True
                            )
                        )
                    ), True
                )
            )
        )
    )
)
