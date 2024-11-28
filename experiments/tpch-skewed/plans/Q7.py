alg.projection ( 
    [ "n1.n_name", "n3.n_name", "l_year", "sum_volume", "avg_volume" ],
    alg.aggregation ( 
        [ "n1.n_name", "n3.n_name", "l_year" ], 
        [ ( Reduction.SUM, "volume", "sum_volume" ),
        ( Reduction.AVG, "volume", "avg_volume" ) ],
        128, True,
        alg.map ( "l_year", 
            scal.ExtractExpr ( scal.AttrExpr ( "l_shipdate" ), 
            scal.ExtractType.YEAR 
        ),
            alg.map ( "volume", 
                scal.MulExpr ( 
                    scal.AttrExpr ( "l_extendedprice" ),
                    scal.SubExpr ( 
                        scal.ConstExpr ( "1", Type.INT ),
                        scal.AttrExpr ( "l_discount" )
                    )
                ),
                alg.join (
                    [ ("o_custkey", "c_custkey"), ("n2.n_nationkey", "n3.n_nationkey") ], None, False, True,
                    alg.indexjoin (
                        scal.AttrExpr("nation.tid"), 'nation', ["n_nationkey"],
                        "customer", None, ["c_nationkey"], 8 * 1024 * 1024, False,
                        alg.selection(
                            scal.OrExpr( 
                                scal.EqualsExpr ( 
                                    scal.AttrExpr ( "n3.n_name" ), 
                                    scal.ConstExpr ( "CHINA", Type.STRING ) ), 
                                scal.EqualsExpr ( 
                                    scal.AttrExpr ( "n3.n_name" ), 
                                    scal.ConstExpr ( "MOROCCO", Type.STRING ) ),
                            ), 2, True,
                            alg.scan("nation", "n3")
                        ), True
                    ),
                    alg.indexjoin(
                        scal.AttrExpr("lineitem.tid"), 'lineitem', ["l_orderkey"],
                        "orders", None, ["o_orderkey"], None, True,
                        alg.crossjoin(
                            scal.NotExpr(
                                scal.EqualsExpr(
                                    scal.AttrExpr("n1.n_nationkey"), scal.AttrExpr("n2.n_nationkey")
                                )
                            ), None,
                            alg.selection(
                                scal.OrExpr (  
                                    scal.EqualsExpr ( 
                                        scal.AttrExpr ( "n2.n_name" ), 
                                        scal.ConstExpr ( "MOROCCO", Type.STRING ) ), 
                                    scal.EqualsExpr ( 
                                        scal.AttrExpr ( "n2.n_name" ), 
                                        scal.ConstExpr ( "CHINA", Type.STRING ) ), 
                                ), 2, True,
                                alg.scan ( "nation", "n2" ),
                            ),
                            alg.join(
                                [("s_suppkey","l_suppkey")], None, False, True,
                                alg.indexjoin(
                                    scal.AttrExpr("n1.tid"), 'nation', ["n_nationkey"],
                                    "supplier", None, ["s_nationkey"], None, False,
                                    alg.selection (
                                        scal.OrExpr (  
                                            scal.EqualsExpr ( 
                                                scal.AttrExpr ( "n1.n_name" ), 
                                                scal.ConstExpr ( "MOROCCO", Type.STRING ) ), 
                                            scal.EqualsExpr ( 
                                                scal.AttrExpr ( "n1.n_name" ), 
                                                scal.ConstExpr ( "CHINA", Type.STRING ) ), 
                                        ), None, True,
                                        alg.scan ( "nation", "n1" )
                                    ), True
                                ),
                                alg.selection(
                                    scal.AndExpr (
                                        scal.LargerEqualExpr (
                                            scal.AttrExpr ( "l_shipdate" ),
                                            scal.ConstExpr ( "19930101", Type.DATE ) 
                                        ),
                                        scal.SmallerEqualExpr (
                                            scal.AttrExpr ( "l_shipdate" ),
                                            scal.ConstExpr ( "19941231", Type.DATE ) 
                                        )
                                    ), None, True,
                                    alg.scan("lineitem")
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
