# FIXME: Temporary workaround for runtime_cxxmodules
if(ROOT_runtime_cxxmodules_FOUND)
  set(excluded tuple)
endif()
ROOTTEST_ADD_TESTDIRS(EXCLUDED_DIRS ${excluded})
