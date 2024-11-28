alg.aggregation (
    [ "s_name" ],
    [ ( Reduction.COUNT, "", "numwait" ) ], 500000, True,
    alg.semijoin ( 
        ( "l2.l_orderkey", "l1.l_orderkey" ),
        scal.NotExpr ( 
            scal.EqualsExpr (
                scal.AttrExpr ( "l2.l_suppkey" ),
                scal.AttrExpr ( "l1.l_suppkey" )
            )
        ), None, True,
        alg.selection(
            scal.LargerExpr(scal.ConstExpr('1', Type.INT), scal.ConstExpr('0', Type.INT)), 240000000, True,
            alg.scan('lineitem', 'l2'),
        ),
        alg.antijoin ( 
            ( "l3.l_orderkey", "l1.l_orderkey" ), 
            scal.NotExpr ( 
                scal.EqualsExpr (
                    scal.AttrExpr ( "l3.l_suppkey" ),
                    scal.AttrExpr ( "l1.l_suppkey" )
                )
            ), None, True,
            alg.selection (
                scal.LargerExpr (
                    scal.AttrExpr ( "l_receiptdate" ),
                    scal.AttrExpr ( "l_commitdate" )
                ), 150000000, True,
                alg.scan ( "lineitem", "l3" )
            ),
            alg.join(
                [('l_orderkey', 'o_orderkey')], None, False, True,
                alg.selection ( 
                    scal.EqualsExpr (
                        scal.AttrExpr ( "o_orderstatus" ),
                        scal.ConstExpr ( "F", Type.CHAR )
                    ), 15000000, True,
                    alg.scan('orders')
                ),
                alg.selection (
                    scal.LargerExpr ( 
                        scal.AttrExpr ( "l1.l_receiptdate" ),
                        scal.AttrExpr ( "l1.l_commitdate" )
                    ), None, True,
                    alg.indexjoin(
                        scal.AttrExpr("supplier.tid"), 'supplier', ["s_suppkey"], "lineitem", "l1", ["l_suppkey"], None, False,
                        alg.indexjoin(
                            scal.AttrExpr("nation.tid"), 'nation', ["n_nationkey"], "supplier", None, ["s_nationkey"], None, False,
                            alg.selection (
                                scal.EqualsExpr (
                                    scal.AttrExpr ( "n_name" ),
                                    scal.ConstExpr ( "MOROCCO", Type.STRING )
                                ), None, False,
                                alg.scan ( "nation" )
                            )
                        )
                    )
                )
            )
        )
    )
)