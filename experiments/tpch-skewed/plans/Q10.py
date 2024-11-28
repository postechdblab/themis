alg.aggregation (
    [ "c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment" ],
    [ ( Reduction.SUM, "rev", "revenue" ) ], 4 * 1024 * 1024, True,
    alg.map (
        "rev",
        scal.MulExpr ( 
            scal.AttrExpr ( "l_extendedprice" ),
            scal.SubExpr ( 
                scal.ConstExpr ( "1.0f", Type.FLOAT ),
                scal.AttrExpr ( "l_discount" )
            )
        ),
        alg.selection (
            scal.EqualsExpr (
                scal.AttrExpr ( "l_returnflag" ),
                scal.ConstExpr ( "R", Type.CHAR )
            ), 84909981, True, 
            alg.indexjoin (
                scal.AttrExpr("orders.tid"), 'orders', ["o_orderkey"], "lineitem", None, ["l_orderkey"], None, False,
                #alg.multimap([["ccustkey", scal.AttrExpr("c_custkey")], ["cname", scal.AttrExpr("c_name")], ["cphone", scal.AttrExpr("c_phone")], ["nname", scal.AttrExpr("n_name")]],
                alg.indexjoin (
                    scal.AttrExpr("customer.tid"), 'customer', ["c_nationkey"], "nation", None, ["n_nationkey"], None, True,
                    alg.indexjoin(
                        scal.AttrExpr("orders.tid"), 'orders', ["o_custkey"], "customer", None, ["c_custkey"], None, True,
                        alg.selection (
                            scal.AndExpr (
                                scal.LargerEqualExpr (
                                    scal.AttrExpr ( "o_orderdate" ),
                                    scal.ConstExpr ( "19920525", Type.DATE )
                                ),
                                scal.SmallerExpr (
                                    scal.AttrExpr ( "o_orderdate" ),
                                    scal.ConstExpr ( "19920529", Type.DATE )
                                )
                            ), 3483619, True,
                            alg.scan ( "orders" )
                        ), True,
                    )
                )
                #)
            )
        )
    )
)
