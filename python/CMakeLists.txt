if(ROOT_pyroot_FOUND)
  if (ROOT_pyroot_experimental_FOUND)
    set(exp_pyroot True)
  else()
    set(exp_pyroot False)
  endif()
endif()

ROOTTEST_ADD_TESTDIRS()
