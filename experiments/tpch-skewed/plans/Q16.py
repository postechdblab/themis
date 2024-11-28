

alg.aggregation (
    ["pbrand", "ptype", "psize"], [( Reduction.COUNT, "", "supp_count" )], 27840*64, True,
    alg.aggregation (
        [ "pbrand", "ptype", "psize", "suppkey" ],[ ( Reduction.COUNT, "", "supp_count_resolvedistinct" ) ], 128 * 1024 * 1024, True,
        alg.antijoin (
            ( "suppkey", "supkey" ), None, None, True,
            alg.selection ( 
                scal.LikeExpr ( 
                    scal.AttrExpr ( "s_comment" ),
                    scal.ConstExpr ( "%Customer%Complaints%", Type.STRING )
                ), 16 * 1024, True,
                alg.map("supkey", scal.AttrExpr("s_suppkey"), alg.scan ( "supplier" ))
            ),
            alg.map(
                "suppkey", scal.AttrExpr("ps_suppkey"),
                alg.indexjoin(
                    scal.AttrExpr("part.tid"), 'part', ["p_partkey"], "partsupp", None, ["ps_partkey"], None, False,
                    alg.selection(
                        scal.NotExpr ( 
                            scal.EqualsExpr ( 
                                scal.AttrExpr ( "p_brand" ),
                                scal.ConstExpr ( "Brand#11", Type.STRING )
                            )
                        ), None, True,
                        
                        alg.selection(
                            scal.NotExpr ( 
                                scal.LikeExpr ( 
                                    scal.AttrExpr ( "p_type" ),
                                    scal.ConstExpr ( "SMALL ANODIZED%", Type.STRING )
                                )
                            ), None, True,
                            alg.selection(
                                scal.InExpr (
                                    scal.AttrExpr ( "p_size" ),
                                    [
                                        scal.ConstExpr ( "6", Type.INT ),
                                        scal.ConstExpr ( "7", Type.INT ),
                                        scal.ConstExpr ( "22", Type.INT ),
                                        scal.ConstExpr ( "2", Type.INT ),
                                        scal.ConstExpr ( "37", Type.INT ),
                                        scal.ConstExpr ( "31", Type.INT ),
                                        scal.ConstExpr ( "15", Type.INT ),
                                        scal.ConstExpr ( "1", Type.INT ),
                                        
                                    ]
                                ), None, True,
                                alg.multimap([["pbrand", scal.AttrExpr("p_brand")], ["ptype", scal.AttrExpr("p_type")], ["psize", scal.AttrExpr("p_size")]],
                                    alg.scan("part")
                                )
                            )
                        )
                    ), True
                )
            )
        )
    )
    #)
    #)))
)
