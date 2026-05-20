if(broken_tests)
  ROOTTEST_GENERATE_DICTIONARY(normal
    TFileInfo.h
    LINKDEF LinkDef.h
    FIXTURES_SETUP io-constmembers-normal)

  ROOTTEST_GENERATE_DICTIONARY(old
    TFileInfo.h
    LINKDEF oldLinkDef.h
    FIXTURES_REQUIRED io-constmembers-normal)
endif()