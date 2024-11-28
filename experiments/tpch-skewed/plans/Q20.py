alg.projection (
    [ "s_name", "s_address" ],
    alg.semijoin (
        ( "s_suppkey", "ps_suppkey" ), None, None, False,
        alg.selection (
            scal.LargerExpr (
                scal.AttrExpr ( "ps_availqty" ),
                scal.MulExpr (
                    scal.ConstExpr ( "0.5f", Type.DOUBLE ),
                    scal.AttrExpr ( "sum_qty" )
                )
            ), None, True,
            alg.join (
                [ ( "p1.tid", "p2.tid" ), ( "lsuppkey", "ps_suppkey" ) ], None, False, False,
                alg.indexjoin(
                    scal.AttrExpr("part.tid"), 'part', ["p_partkey"], "partsupp", None, ["ps_partkey"], None, False,
                    alg.selection (
                        scal.LikeExpr (
                            scal.AttrExpr ( "p_name" ),
                            scal.ConstExpr( "shiny mined%", Type.STRING )
                        ), None, True,
                        alg.scan ( "part", "p2" )
                    ), True
                ),
                alg.aggregation (
                    [ "p1.tid", "lsuppkey" ],
                    [ ( Reduction.SUM, "lquantity", "sum_qty" ) ], 32 * 1024 * 1024, True,
                    alg.selection (
                        scal.AndExpr (
                            scal.LargerEqualExpr (
                                scal.AttrExpr ( "lshipdate" ),
                                scal.ConstExpr ( "19930101", Type.DATE )
                            ),
                            scal.SmallerExpr (
                                scal.AttrExpr ( "lshipdate" ),
                                scal.ConstExpr ( "19940101", Type.DATE )
                            )
                        ), 24000000, True,
                        alg.multimap(
                            [("lsuppkey", scal.AttrExpr("l_suppkey")),("lshipdate", scal.AttrExpr("l_shipdate")), ("lquantity", scal.AttrExpr("l_quantity"))],
                            alg.indexjoin(
                                scal.AttrExpr("part.tid"), 'part', ["p_partkey"], "lineitem", None, ["l_partkey"], None, False,
                                alg.selection (
                                    scal.LikeExpr (
                                        scal.AttrExpr ( "pname" ),
                                        scal.ConstExpr( "shiny mined%", Type.STRING )
                                    ), None, True,
                                    alg.multimap([("pname", scal.AttrExpr("p_name"))], alg.scan("part", "p1"))
                                ), True
                            ),                            
                        )
                    )
                )
            )
        ),
        alg.indexjoin (
            scal.AttrExpr("nation.tid"), 'nation', ["n_nationkey"], "supplier", None, ["s_nationkey"], None, False,
            alg.selection (
                scal.EqualsExpr (
                    scal.AttrExpr ( "n_name" ),
                    scal.ConstExpr ( "GERMANY", Type.STRING )
                ), None, False,
                alg.scan ( "nation" )
            ), True
        )
    )
)

