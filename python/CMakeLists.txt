if(ROOT_pyroot_FOUND)
  if (ROOT_pyroot_legacy_FOUND)
    set(legacy_pyroot True)
  else()
    set(legacy_pyroot False)
  endif()

  find_python_module(pytest REQUIRED)

  if (PY_PYTEST_FOUND)
    ROOTTEST_ADD_TESTDIRS()
  endif()
endif()
