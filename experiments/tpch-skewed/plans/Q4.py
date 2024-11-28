
alg.aggregation (
    [ "o_orderpriority" ],
    [ ( Reduction.COUNT, "", "order_count" ) ], 32, False,
    alg.selection (
        scal.SmallerExpr ( 
            scal.AttrExpr ( "l_commitdate" ),
            scal.AttrExpr ( "l_receiptdate" )
        ), None, True,
        alg.indexjoin(
            scal.AttrExpr("orders.tid"), 'orders', ["o_orderkey"],
            "lineitem", None, ["l_orderkey"], None, False,
            alg.map(
                "opriority", scal.AttrExpr("o_orderpriority"),
                alg.selection (
                    scal.AndExpr (
                        scal.SmallerExpr ( 
                            scal.AttrExpr ( "o_orderdate" ),
                            scal.ConstExpr ( "19930801", Type.DATE )
                        ),
                        scal.LargerEqualExpr ( 
                            scal.AttrExpr ( "o_orderdate" ),
                            scal.ConstExpr ( "19930501", Type.DATE ) # 177829, 177829, # 177829, 281250, #103421
                        )
                    ), None, True,
                    alg.scan("orders")                        #26006352     
                )
            ), True
        )
    )
)
