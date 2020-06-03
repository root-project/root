if(ROOT_pyroot_FOUND)
  if (ROOT_pyroot_legacy_FOUND)
    set(legacy_pyroot True)
  else()
    set(legacy_pyroot False)
  endif()
endif()

ROOTTEST_ADD_TESTDIRS()
