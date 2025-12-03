if(ROOT_pyroot_FOUND)

  if(ROOT_roofit_FOUND)
    set(roofit True)
  else()
    set(roofit False)
  endif(ROOT_roofit_FOUND)

  find_python_module(pytest OPTIONAL)

  if (PY_PYTEST_FOUND)
    ROOTTEST_ADD_TESTDIRS()
  endif()
endif()
