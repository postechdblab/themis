alg.projection (
    [ "s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment" ],
#    alg.join(
#        [("r1.r_regionkey", "r3.r_regionkey")], None, False, True,
    alg.join(
        [("s1.s_nationkey", "n1.n_nationkey")], None, False, True,
        alg.indexjoin(
            scal.AttrExpr("r1.tid"), 'region', ["r_regionkey"], "nation", "n1", ["n_regionkey"], None, False,
            alg.selection (
                scal.EqualsExpr( scal.AttrExpr("r1.r_name"), scal.ConstExpr("MIDDLE EAST", Type.STRING) ), 
                None, False,
                alg.scan("region", "r1")
            ),
        ),
        alg.indexjoin(
            scal.AttrExpr("ps1.tid"), 'partsupp', ["ps_suppkey"], "supplier", "s1", ["s_suppkey"], None, True,
            alg.selection(
                scal.EqualsExpr( scal.AttrExpr("ps1.ps_supplycost"), scal.AttrExpr("min_supplycost") ), None, True,
                alg.indexjoin(
                    scal.AttrExpr("p1.tid"), 'part', ["p_partkey"], "partsupp", "ps1", ["ps_partkey"], None, False,
                    alg.join(
                        [("ps2.ps_partkey", "p1.p_partkey")], None, False, True,
                        alg.selection(
                            scal.AndExpr(
                                scal.LikeExpr( scal.AttrExpr("p1.p_type"), scal.ConstExpr("%NY MINED GOLD", Type.STRING) ),
                                scal.EqualsExpr( scal.AttrExpr("p1.p_size"), scal.ConstExpr("15", Type.INT) )
                            ), None, True,
                            alg.scan("part", "p1"),
                        ),
                        alg.aggregation(
                            [ "ps2.ps_partkey" ],
                            [ ( Reduction.MIN, "ps2.ps_supplycost", "min_supplycost" ) ], 10000000, True,
                            alg.semijoin(
                                [("ps2.ps_partkey", "p2.p_partkey")], None, None, True,
                                alg.selection (
                                    scal.AndExpr(
                                        scal.LikeExpr( scal.AttrExpr("p2.p_type"), scal.ConstExpr("%NY MINED GOLD", Type.STRING) ),
                                        scal.EqualsExpr( scal.AttrExpr("p2.p_size"), scal.ConstExpr("15", Type.INT) )
                                    ), None, True,
                                    alg.scan("part", "p2")
                                ),
                                alg.indexjoin(
                                    scal.AttrExpr("s2.tid"), 'supplier', ["s_suppkey"], "partsupp", "ps2", ["ps_suppkey"], None, False,
                                    alg.indexjoin(
                                        scal.AttrExpr("n2.tid"), 'nation', ["n_nationkey"], "supplier", "s2", ["s_nationkey"], None, False,
                                        alg.indexjoin(
                                            scal.AttrExpr("r2.tid"), 'region', ["r_regionkey"], "nation", "n2", ["n_regionkey"], None, False,
                                            alg.selection (
                                                scal.EqualsExpr( scal.AttrExpr("r2.r_name"), scal.ConstExpr("MIDDLE EAST", Type.STRING) ), 
                                                None, False,
                                                alg.scan("region", "r2")
                                            ), True
                                        ), True
                                    ), True
                                )
                            )
                        )
                    ), True,
                )
            )
        )
    )
)