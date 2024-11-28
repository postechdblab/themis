#todo: check result with double precision

alg.aggregation ( 
    [],
    [ ( Reduction.SUM, "rev", "revenue" ), ( Reduction.COUNT, "rev", "count" ) ], 1, True,
    alg.map (
        "rev",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.AttrExpr ( "l_discount" )
        ),
        alg.selection(
            scal.SmallerExpr (
                scal.AttrExpr ( "l_quantity" ),
                scal.ConstExpr ( "100", Type.DOUBLE )
            ), None, True,
            alg.selection(
                scal.AndExpr (
                    scal.LargerEqualExpr (
                        scal.AttrExpr ( "l_shipdate" ),
                        scal.ConstExpr ( "19930101", Type.DATE )
                    ),                
                    scal.SmallerExpr (
                        scal.AttrExpr ( "l_shipdate" ),
                        scal.ConstExpr ( "19940101", Type.DATE )
                    ),
                ), None, True,
                alg.selection(
                    scal.AndExpr (
                        scal.LargerEqualExpr (
                            scal.AttrExpr ( "l_discount" ),
                            scal.ConstExpr ( "0.05", Type.DOUBLE )
                        ),
                        scal.SmallerEqualExpr (
                            scal.AttrExpr ( "l_discount" ),
                            scal.ConstExpr ( "0.07", Type.DOUBLE )
                        )
                    ), None, True,
                    alg.scan ( "lineitem" ) # 180000000
                )   
            ),
        )
    )
)

