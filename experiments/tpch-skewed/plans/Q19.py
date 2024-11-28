
alg.aggregation (
    [],
    [ ( Reduction.SUM, "rev", "revenue" ) ], 1, True,
    alg.map (
        "rev",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.SubExpr ( 
                scal.ConstExpr ( "1.0", Type.DOUBLE ),
                scal.AttrExpr ( "l_discount" )
            )
        ),
        alg.selection (
            scal.OrExpr (
                scal.OrExpr (
                    #or1
                    scal.AndExpr (
                        scal.AndExpr (
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.AndExpr (
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "p_brand" ),
                                            scal.ConstExpr ( "Brand#00", Type.STRING )
                                        ),
                                        scal.InExpr (
                                            scal.AttrExpr ( "p_container" ),
                                            [
                                                scal.ConstExpr ( "SM CASE", Type.STRING ),
                                                scal.ConstExpr ( "SM BOX", Type.STRING ),
                                                scal.ConstExpr ( "SM PACK", Type.STRING ),
                                                scal.ConstExpr ( "SM PKG", Type.STRING )
                                            ]
                                        )
                                    ),
                                    scal.LargerEqualExpr (
                                        scal.AttrExpr ( "l_quantity" ),
                                        scal.ConstExpr ( "2.0f", Type.DOUBLE )
                                    )
                                ),
                                scal.SmallerEqualExpr (
                                    scal.AttrExpr ( "l_quantity" ),
                                    scal.ConstExpr ( "12.0f", Type.DOUBLE )
                                )
                            ),
                            scal.LargerEqualExpr (
                                scal.AttrExpr ( "p_size" ),
                                scal.ConstExpr ( "1", Type.DOUBLE )
                            )
                        ),
                        scal.SmallerEqualExpr (
                            scal.AttrExpr ( "p_size" ),
                            scal.ConstExpr ( "5", Type.DOUBLE )
                        )
                    ),
                    #or2
                    scal.AndExpr (
                        scal.AndExpr (
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.AndExpr (
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "p_brand" ),
                                            scal.ConstExpr ( "Brand#00", Type.STRING )
                                        ),
                                        scal.InExpr (
                                            scal.AttrExpr ( "p_container" ),
                                            [
                                                scal.ConstExpr ( "MED BAG", Type.STRING ),
                                                scal.ConstExpr ( "MED BOX", Type.STRING ),
                                                scal.ConstExpr ( "MED PKG", Type.STRING ),
                                                scal.ConstExpr ( "MED PACK", Type.STRING )
                                            ]
                                        )
                                    ),
                                    scal.LargerEqualExpr (
                                        scal.AttrExpr ( "l_quantity" ),
                                        scal.ConstExpr ( "19.0f", Type.DOUBLE )
                                    )
                                ),
                                scal.SmallerEqualExpr (
                                    scal.AttrExpr ( "l_quantity" ),
                                    scal.ConstExpr ( "29.0f", Type.DOUBLE )
                                )
                            ),
                            scal.LargerEqualExpr (
                                scal.AttrExpr ( "p_size" ),
                                scal.ConstExpr ( "1", Type.DOUBLE )
                            )
                        ),
                        scal.SmallerEqualExpr (
                            scal.AttrExpr ( "p_size" ),
                            scal.ConstExpr ( "10", Type.DOUBLE )
                        )
                    )
                ),
                #or3
                scal.AndExpr (
                    scal.AndExpr (
                        scal.AndExpr (
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.EqualsExpr (
                                        scal.AttrExpr ( "p_brand" ),
                                        scal.ConstExpr ( "Brand#55", Type.STRING )
                                    ),
                                    scal.InExpr (
                                        scal.AttrExpr ( "p_container" ),
                                        [
                                            scal.ConstExpr ( "LG CASE", Type.STRING ),
                                            scal.ConstExpr ( "LG BOX", Type.STRING ),
                                            scal.ConstExpr ( "LG PACK", Type.STRING ),
                                            scal.ConstExpr ( "LG PKG", Type.STRING )
                                        ]
                                    )
                                ),
                                scal.LargerEqualExpr (
                                    scal.AttrExpr ( "l_quantity" ),
                                    scal.ConstExpr ( "50.0f", Type.DOUBLE )
                                )
                            ),
                            scal.SmallerEqualExpr (
                                scal.AttrExpr ( "l_quantity" ),
                                scal.ConstExpr ( "60.0f", Type.DOUBLE )
                            )
                        ),
                        scal.LargerEqualExpr (
                            scal.AttrExpr ( "p_size" ),
                            scal.ConstExpr ( "1", Type.DOUBLE )
                        )
                    ),
                    scal.SmallerEqualExpr (
                        scal.AttrExpr ( "p_size" ),
                        scal.ConstExpr ( "15", Type.DOUBLE )
                    )
                )
            ), None, True,
            alg.join(
                [("l_partkey", "p_partkey")], None, False, True,
                alg.selection(
                    scal.OrExpr (
                        scal.OrExpr (
                            #or1
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.AndExpr (
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "p_brand" ),
                                            scal.ConstExpr ( "Brand#00", Type.STRING )
                                        ),
                                        scal.InExpr (
                                            scal.AttrExpr ( "p_container" ),
                                            [
                                                scal.ConstExpr ( "SM CASE", Type.STRING ),
                                                scal.ConstExpr ( "SM BOX", Type.STRING ),
                                                scal.ConstExpr ( "SM PACK", Type.STRING ),
                                                scal.ConstExpr ( "SM PKG", Type.STRING )
                                            ]
                                        )
                                    ),
                                    scal.LargerEqualExpr (
                                        scal.AttrExpr ( "p_size" ),
                                        scal.ConstExpr ( "1", Type.DOUBLE )
                                    )
                                ),
                                scal.SmallerEqualExpr (
                                    scal.AttrExpr ( "p_size" ),
                                    scal.ConstExpr ( "5", Type.DOUBLE )
                                )
                            ),
                            #or2
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.AndExpr (
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "p_brand" ),
                                            scal.ConstExpr ( "Brand#00", Type.STRING )
                                        ),
                                        scal.InExpr (
                                            scal.AttrExpr ( "p_container" ),
                                            [
                                                scal.ConstExpr ( "MED BAG", Type.STRING ),
                                                scal.ConstExpr ( "MED BOX", Type.STRING ),
                                                scal.ConstExpr ( "MED PKG", Type.STRING ),
                                                scal.ConstExpr ( "MED PACK", Type.STRING )
                                            ]
                                        )
                                    ),
                                    scal.LargerEqualExpr (
                                        scal.AttrExpr ( "p_size" ),
                                        scal.ConstExpr ( "1", Type.DOUBLE )
                                    )
                                ),
                                scal.SmallerEqualExpr (
                                    scal.AttrExpr ( "p_size" ),
                                    scal.ConstExpr ( "10", Type.DOUBLE )
                                )
                            )
                        ),
                        #or3
                        scal.AndExpr (
                            scal.AndExpr (
                                scal.AndExpr (
                                    scal.EqualsExpr (
                                        scal.AttrExpr ( "p_brand" ),
                                        scal.ConstExpr ( "Brand#55", Type.STRING )
                                    ),
                                    scal.InExpr (
                                        scal.AttrExpr ( "p_container" ),
                                        [
                                            scal.ConstExpr ( "LG CASE", Type.STRING ),
                                            scal.ConstExpr ( "LG BOX", Type.STRING ),
                                            scal.ConstExpr ( "LG PACK", Type.STRING ),
                                            scal.ConstExpr ( "LG PKG", Type.STRING )
                                        ]
                                    )
                                ),
                                scal.LargerEqualExpr (
                                    scal.AttrExpr ( "p_size" ),
                                    scal.ConstExpr ( "1", Type.DOUBLE )
                                )
                            ),
                            scal.SmallerEqualExpr (
                                scal.AttrExpr ( "p_size" ),
                                scal.ConstExpr ( "15", Type.DOUBLE )
                            )
                        )
                    ), None, True,
                    alg.scan ( "part" )
                ),
                alg.selection (
                    scal.EqualsExpr (
                        scal.AttrExpr ( "l_shipinstruct" ),
                        scal.ConstExpr ( "DELIVER IN PERSON", Type.STRING )
                    ), None, True,
                    alg.selection(
                        scal.InExpr (
                            scal.AttrExpr ( "l_shipmode" ),
                            [
                                scal.ConstExpr ( "AIR", Type.STRING ),
                                scal.ConstExpr ( "AIR REG", Type.STRING ),
                            ]
                        ), None, True,
                        alg.scan("lineitem")
                        
                    )
                )
            )
        )
    )
)
