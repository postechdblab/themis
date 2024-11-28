alg.aggregation (
    [ "lshipmode" ],
    [ ( Reduction.SUM, "highline", "high_line_count" ), ( Reduction.SUM, "lowline", "low_line_count" ) ], 128, True,
    alg.map (
        "highline",
        scal.CaseExpr (
            [ 
                ( 
                    scal.OrExpr (
                        scal.EqualsExpr ( 
                            scal.AttrExpr ( "o_orderpriority"), 
                            scal.ConstExpr ( "1-URGENT", Type.STRING )
                        ),
                        scal.EqualsExpr ( 
                            scal.AttrExpr ( "o_orderpriority"), 
                            scal.ConstExpr ( "2-HIGH", Type.STRING )
                        )
                    ),
                    scal.ConstExpr ( "1", Type.INT ) 
                )
            ],
            scal.ConstExpr ( "0", Type.INT ) 
        ),
        alg.map (
            "lowline",
            scal.CaseExpr (
                [ 
                    ( 
                        scal.AndExpr (
                            scal.NotExpr ( scal.EqualsExpr ( 
                                scal.AttrExpr ( "o_orderpriority"), 
                                scal.ConstExpr ( "1-URGENT", Type.STRING )
                            ) ),
                            scal.NotExpr ( scal.EqualsExpr ( 
                                scal.AttrExpr ( "o_orderpriority"), 
                                scal.ConstExpr ( "2-HIGH", Type.STRING )
                            ) ),
                        ),
                        scal.ConstExpr ( "1", Type.INT ) 
                    )
                ],
                scal.ConstExpr ( "0", Type.INT ) 
            ),
            alg.indexjoin (
                scal.AttrExpr("lineitem.tid"), 'lineitem', ["l_orderkey"], "orders", None, ["o_orderkey"], None, True,
                alg.selection (
                    scal.InExpr (
                        scal.AttrExpr ( "lshipmode" ),
                        [
                            scal.ConstExpr ( "AIR", Type.STRING ),
                            scal.ConstExpr ( "REG AIR", Type.STRING )
                        ]
                    ), None, True,
                    #alg.multimap(
                    #    [['lshipmode', scal.AttrExpr('l_shipmode')]],
                    alg.selection(
                        scal.LargerEqualExpr (
                            scal.AttrExpr ( "lreceiptdate" ),
                            scal.ConstExpr ( "19940101", Type.DATE )
                        ), None, True,
                        alg.selection(
                            scal.SmallerExpr (
                                scal.AttrExpr ( "lreceiptdate" ),
                                scal.ConstExpr ( "19950101", Type.DATE )
                            ), None, True,
                            alg.selection(
                                scal.SmallerExpr (
                                    scal.AttrExpr ( "lcommitdate" ),
                                    scal.AttrExpr ( "lreceiptdate" )
                                ), None, True,
                                #alg.multimap([['lreceiptdate', scal.AttrExpr('l_receiptdate')]],
                                alg.selection(
                                    scal.SmallerExpr (
                                        scal.AttrExpr ( "l_shipdate" ),
                                        scal.AttrExpr ( "lcommitdate" )
                                    ), None, True,
                                    alg.multimap([["lcommitdate", scal.AttrExpr("l_commitdate")], ['lreceiptdate', scal.AttrExpr('l_receiptdate')],['lshipmode', scal.AttrExpr('l_shipmode')]],
                                        alg.scan ( "lineitem" )
                                    )
                                )
                                #)
                            )
                        )
                    )
                    #)
                )
            )
        )
    )
) 