alg.aggregation ( 
    [ "cntrycode" ],
    [ ( Reduction.COUNT, "", "numcust" ),
    ( Reduction.SUM, "c_acctbal", "totacctbal" ) ], 32, False,
    alg.map ( 
        "cntrycode",
        scal.SubstringExpr ( 
            scal.AttrExpr ( "c_phone" ),
            scal.ConstExpr ( "1", Type.INT ),
            scal.ConstExpr ( "2", Type.INT )
        ),
        alg.antijoin (
            [ ( "c2.c_custkey", "orders.o_custkey" ) ],
            None, None, True,
            #alg.scan ( "orders" ), 
            alg.indexjoin(
                scal.AttrExpr("c3.tid"), 'customer', ["c_custkey"], "orders", None, ["o_custkey"], None, False,
                alg.selection ( 
                    scal.InExpr ( 
                        scal.SubstringExpr ( 
                            scal.AttrExpr ( "c3.c_phone" ),
                            scal.ConstExpr ( "1", Type.INT ),
                            scal.ConstExpr ( "2", Type.INT )
                        ),
                        [
                            scal.ConstExpr ( "40", Type.STRING ),
                            scal.ConstExpr ( "50", Type.STRING ),
                            scal.ConstExpr ( "60", Type.STRING ),
                            scal.ConstExpr ( "70", Type.STRING ),
                            scal.ConstExpr ( "80", Type.STRING ),
                            scal.ConstExpr ( "33", Type.STRING ),
                            scal.ConstExpr ( "15", Type.STRING )
                        ]
                    ), None, True,
                    alg.scan ( "customer", "c3" )
                ), True
            ),
            alg.crossjoin (
                scal.LargerExpr ( 
                    scal.AttrExpr ( "c2.c_acctbal" ), 
                    scal.ConstExpr("0", Type.DOUBLE),
                    #scal.AttrExpr ( "avg" )
                ), None,
                alg.aggregation ( 
                    [],
                    [ ( Reduction.AVG, "c1.c_acctbal", "avg" ) ], 1, True,
                    alg.selection (
                        scal.InExpr ( 
                            scal.SubstringExpr ( 
                                scal.AttrExpr ( "c1.c_phone" ),
                                scal.ConstExpr ( "1", Type.INT ),
                                scal.ConstExpr ( "2", Type.INT )
                            ),
                            [
                                scal.ConstExpr ( "40", Type.STRING ),
                                scal.ConstExpr ( "50", Type.STRING ),
                                scal.ConstExpr ( "60", Type.STRING ),
                                scal.ConstExpr ( "70", Type.STRING ),
                                scal.ConstExpr ( "80", Type.STRING ),
                                scal.ConstExpr ( "33", Type.STRING ),
                                scal.ConstExpr ( "15", Type.STRING )
                            ]
                        ), None, True,
                        alg.selection (
                            scal.LargerExpr ( 
                                scal.AttrExpr ( "c1.c_acctbal" ), 
                                scal.ConstExpr ( "0.00", Type.DOUBLE )
                            ), None, True,
                            alg.scan ( "customer", "c1" )
                        )
                    )
                ),
                alg.selection ( 
                    scal.InExpr ( 
                        scal.SubstringExpr ( 
                            scal.AttrExpr ( "c2.c_phone" ),
                            scal.ConstExpr ( "1", Type.INT ),
                            scal.ConstExpr ( "2", Type.INT )
                        ),
                        [
                            scal.ConstExpr ( "40", Type.STRING ),
                            scal.ConstExpr ( "50", Type.STRING ),
                            scal.ConstExpr ( "60", Type.STRING ),
                            scal.ConstExpr ( "70", Type.STRING ),
                            scal.ConstExpr ( "80", Type.STRING ),
                            scal.ConstExpr ( "33", Type.STRING ),
                            scal.ConstExpr ( "15", Type.STRING )
                        ]
                    ), None, True,
                    alg.scan ( "customer", "c2" )
                )
            )
        )
    )
)

