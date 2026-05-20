if(pyroot)

  if(roofit)
    set(roofit True)
  else()
    set(roofit False)
  endif(roofit)

  find_python_module(pytest OPTIONAL)

  if (PY_PYTEST_FOUND)
    ROOTTEST_ADD_TESTDIRS()
  endif()
endif()
