

alg.aggregation (
    [ "c_count" ],
    [ ( Reduction.COUNT, "c_count", "custdist" ) ], 1024, False,
    alg.aggregation (
        [ "ckey" ],
        [ ( Reduction.COUNT, "", "c_count" ) ],
        16  * 1024 * 1024, True,
        alg.join (
            [ ( "ckey", "o_custkey" ) ], None, True, True,
            alg.selection (
                scal.NotExpr (
                    scal.LikeExpr ( 
                        scal.AttrExpr ( "o_comment" ),
                        scal.ConstExpr ( "%special%deposits%", Type.STRING )
                    )
                ), 44563240 , True, #45,000,000
                alg.scan("orders")
            ),
            alg.map(
                "ckey", scal.AttrExpr("c_custkey"),
                alg.scan("customer")
            )
        )
    )
)

