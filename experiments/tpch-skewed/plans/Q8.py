alg.projection ( 
    [ "o_year", "mkt_share" ],
    alg.map ( "mkt_share",
        scal.DivExpr (
            scal.AttrExpr ( "sum_volume_morocco" ),
            scal.AttrExpr ( "sum_volume" )
        ),
        alg.aggregation ( 
            [ "o_year" ], 
            [ ( Reduction.SUM, "case_volume", "sum_volume_morocco" ), ( Reduction.SUM, "volume", "sum_volume" ) ],
            32, True,
            alg.map ( "case_volume",
                scal.CaseExpr (
                    [ ( scal.EqualsExpr ( scal.AttrExpr ( "n2.n_name" ), scal.ConstExpr ( "MOROCCO", Type.STRING ) ), scal.AttrExpr ( "volume" ) ) ],
                    scal.ConstExpr ( "0", Type.FLOAT )
                ),
                alg.map ( "volume",
                    scal.MulExpr (
                        scal.AttrExpr ( "l_extendedprice" ),
                        scal.SubExpr (
                            scal.ConstExpr ( "1.0f", Type.FLOAT ), 
                            scal.AttrExpr ( "l_discount" )
                        )
                    ),
                    alg.indexjoin(
                        scal.AttrExpr("supplier.tid"), 'supplier', ["s_nationkey"],
                        "nation", "n2", ["n_nationkey"], None, True,
                        alg.indexjoin(
                            scal.AttrExpr("lineitem.tid"), 'lineitem', ["l_suppkey"],
                            "supplier", None, ["s_suppkey"], None, True,
                            alg.join(
                                [("l_orderkey", "o_orderkey")],
                                None, False, True,
                                alg.map ( "o_year",
                                    scal.ExtractExpr (
                                        scal.AttrExpr ( "o_orderdate" ), 
                                        scal.ExtractType.YEAR 
                                    ),
                                    alg.selection(
                                        scal.AndExpr (  
                                            scal.LargerEqualExpr (
                                                scal.AttrExpr ( "o_orderdate" ), scal.ConstExpr ( "19930101", Type.DATE )
                                            ),
                                            scal.SmallerEqualExpr (
                                                scal.AttrExpr ( "o_orderdate" ), scal.ConstExpr ( "19941231", Type.DATE )
                                            )
                                        ), None, True,
                                        alg.indexjoin(
                                            scal.AttrExpr("customer.tid"), 'customer', ["c_custkey"],
                                            "orders", None, ["o_custkey"], None, False,
                                            alg.indexjoin(
                                                scal.AttrExpr("n1.tid"), 'nation', ["n_nationkey"],
                                                "customer", None, ["c_nationkey"], None, False,
                                                alg.indexjoin(
                                                    scal.AttrExpr("region.tid"), 'region', ["r_regionkey"],
                                                    "nation", "n1", ["n_regionkey"], 5, False,
                                                    alg.selection(
                                                        scal.EqualsExpr( scal.AttrExpr("r_name"), scal.ConstExpr("AFRICA", Type.STRING) ),
                                                        None, True,
                                                        alg.scan("region")
                                                    )
                                                ), True
                                            ), True
                                        )
                                    ),
                                ),
                                alg.indexjoin(
                                    scal.AttrExpr("part.tid"), 'part', ["p_partkey"],
                                    "lineitem", None, ["l_partkey"], None, False,
                                    alg.selection(
                                        scal.EqualsExpr (
                                            scal.AttrExpr ( "p_type" ), 
                                            scal.ConstExpr ( "SHINY MINED GOLD", Type.STRING )
                                        ), None, True,
                                        alg.scan ( "part" )
                                    ), True
                                )
                            )
                        )
                    )
                )
            )
        )
    )
) 
