alg.projection (
    [ "ps2.ps_partkey", "sum_suppval2" ],
    alg.crossjoin (
        scal.LargerExpr (
            scal.AttrExpr ( "sum_suppval2" ),
            scal.AttrExpr ( "lim_suppval" )
        ), 148873,
        alg.map (
            "lim_suppval",
            scal.MulExpr ( 
                scal.AttrExpr ( "sum_suppval" ),
                scal.ConstExpr ( "0.000002f", Type.DOUBLE )
            ),
            alg.aggregation (
                [],
                [ ( Reduction.SUM, "suppval", "sum_suppval" ) ], 1, False,
                alg.map (
                    "suppval",
                    scal.MulExpr ( 
                        scal.AttrExpr ( "ps1.ps_supplycost" ),
                        scal.AttrExpr ( "ps1.ps_availqty" )
                    ),
                    alg.indexjoin (
                        scal.AttrExpr("s1.tid"), 'supplier', ["s_suppkey"],
                        "partsupp", "ps1", ["ps_suppkey"], None, False,
                        alg.indexjoin (
                            scal.AttrExpr("n1.tid"), 'nation', ["n_nationkey"],
                            "supplier", "s1", ["s_nationkey"], None, False,
                            alg.selection (
                                scal.EqualsExpr ( 
                                    scal.AttrExpr ( "n1.n_name" ),
                                    scal.ConstExpr ( "UNITED STATES", Type.STRING )
                                ), None, True,
                                alg.scan ( "nation", "n1" )
                            ), True
                        ), True
                    )
                )
            )
        ),
        alg.aggregation (
            [ "ps2.ps_partkey" ],
            [ ( Reduction.SUM, "suppval2", "sum_suppval2" ) ], 16000000, True,
            alg.map (
                "suppval2",
                scal.MulExpr ( 
                    scal.AttrExpr ( "ps2.ps_supplycost" ),
                    scal.AttrExpr ( "ps2.ps_availqty" )
                ),
                alg.indexjoin (
                    scal.AttrExpr("s2.tid"), 'supplier', ["s_suppkey"],
                    "partsupp", "ps2", ["ps_suppkey"], None, False,
                    alg.indexjoin (
                        scal.AttrExpr("n2.tid"), 'nation', ["n_nationkey"],
                        "supplier", "s2", ["s_nationkey"], None, False,
                        alg.selection (
                            scal.EqualsExpr ( 
                                scal.AttrExpr ( "n2.n_name" ),
                                scal.ConstExpr ( "UNITED STATES", Type.STRING )
                            ), None, True,
                            alg.scan ( "nation", "n2" )
                        ), True,
                    ), True
                )
            )
        )
    )
)
