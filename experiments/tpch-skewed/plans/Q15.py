[
alg.createtemp ( 
    "revenue",
    alg.aggregation (
        "l_suppkey",
        [ ( Reduction.SUM, "revenue", "sum_revenue" ) ],
        512 * 1024, True,
        alg.map ( 
            "revenue",
            scal.MulExpr ( 
                scal.AttrExpr ( "l_extendedprice" ),
                scal.SubExpr ( 
                    scal.ConstExpr ( "1.0", Type.FLOAT ),
                    scal.AttrExpr ( "l_discount" )
                )
            ),
            alg.selection (
                scal.LargerEqualExpr ( 
                    scal.AttrExpr ( "l_shipdate" ),
                    scal.ConstExpr ( "19940601", Type.DATE )
                ), None, True,
                alg.selection (
                    scal.SmallerExpr ( 
                        scal.AttrExpr ( "l_shipdate" ),
                        scal.ConstExpr ( "19940901", Type.DATE )
                    ), None, True,
                    alg.scan ( "lineitem" )
                )
            )
        )
    )
),
alg.projection (
    [ "s_suppkey", "s_name", "s_address", "s_phone", "total_revenue" ],
    alg.join (
        ( "l_suppkey", "s_suppkey" ), None, False, True,
        alg.join (
            ( "total_revenue", "sum_revenue" ), None, False, True,
            alg.aggregation (
                [],
                [( Reduction.MAX, "sum_revenue", "total_revenue" )], 1, False,
                alg.scan ( "revenue", "r1" )
            ),
            alg.scan ( "revenue", "r2" )
        ),
        alg.scan ( "supplier" )
    )
)
]
