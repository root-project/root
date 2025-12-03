foreach(dict Safr test0 test1 test2)
  ROOTTEST_GENERATE_DICTIONARY(${dict}
                               ${dict}.h
                               LINKDEF ${dict}LinkDef.h
                               NO_ROOTMAP NO_CXXMODULE)
endforeach()

foreach(dict map1 map2)
  ROOTTEST_GENERATE_DICTIONARY(${dict}
                               ${dict}.C
                               LINKDEF emptyLinkdef.h
                               NO_ROOTMAP NO_CXXMODULE)
endforeach()
