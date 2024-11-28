alg.projection ( 
    [ "n_name", "o_year", "sum_profit"],
    alg.aggregation (
        [ "n_name", "o_year" ],
        [ ( Reduction.SUM, "amount", "sum_profit" ) ], 60150, True,
        alg.indexjoin(
            scal.AttrExpr( "s1.tid" ), 'supplier', ["s_nationkey"],
            "nation", None, ["n_nationkey"], None, True,
            alg.indexjoin(
                scal.AttrExpr("l1.tid"), 'lineitem', ["l_suppkey"],
                "supplier", "s1", ["s_suppkey"], None, True,
                alg.map (
                    "o_year",
                    scal.ExtractExpr ( scal.AttrExpr ( "o_orderdate" ), scal.ExtractType.YEAR ),   
                    alg.indexjoin(
                        scal.AttrExpr( "l1.tid" ), 'lineitem', ["l_orderkey"],
                        "orders", None, ["o_orderkey"], None, True,
                        alg.map(
                            "amount",
                            scal.SubExpr (
                                scal.MulExpr ( 
                                    scal.AttrExpr ( "l_extendedprice" ),
                                    scal.SubExpr ( 
                                        scal.ConstExpr ( "1.0f", Type.FLOAT ),
                                        scal.AttrExpr ( "l_discount" )
                                    )
                                ),
                                scal.MulExpr ( 
                                    scal.AttrExpr ( "ps_supplycost" ),
                                    scal.AttrExpr ( "l_quantity" ),
                                )
                            ),
                            alg.indexjoin(
                                scal.AttrExpr("ps1.tid"), 'partsupp', ["ps_partkey","ps_suppkey"],
                                "lineitem", "l1", ["l_partkey", "l_suppkey"], None, False,
                                alg.indexjoin(
                                    scal.AttrExpr("part.tid"), 'part', ["p_partkey"],
                                    "partsupp", "ps1", ["ps_partkey"], None, False,
                                    alg.selection(
                                        scal.LikeExpr ( 
                                            scal.AttrExpr ( "p_name" ),
                                            scal.ConstExpr ( "%shiny mined gold%", Type.STRING )
                                        ), None, True,
                                        alg.scan ( "part", "p1" )
                                    ), True
                                ), True
                            )
                        )
                    )
                )
            )
        )
    )
)
