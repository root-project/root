# load the list of failed tests
IF (EXISTS ${_failed_tests_file})
   FILE(STRINGS ${_failed_tests_file} _failed_tests)
ELSE (EXISTS ${_failed_tests_file})
   SET(_failed_tests "0")
ENDIF (EXISTS ${_failed_tests_file})

# create ctest args for running only failed tests
FOREACH (_test ${_failed_tests})

   STRING(REGEX MATCH "^[0-9]+" _test ${_test})

   IF (NOT DEFINED _ctest_args)
      SET(_ctest_args "${_ctest_args} ${_test},${_test},")
   ENDIF (NOT DEFINED _ctest_args)

   SET(_ctest_args "${_ctest_args},${_test}")

ENDFOREACH (_test ${_failed_tests})

# clear failed test file
FILE(REMOVE ${_failed_tests_file})

# write out ctest arg file
FILE(WRITE ${_ctest_args_file} "${_ctest_args}\n")
