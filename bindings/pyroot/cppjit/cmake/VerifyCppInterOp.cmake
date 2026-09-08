# Runs between the CppInterOp ExternalProject's configure and build steps:
# the sub-configure records its project() name in its cache; reject any
# source tree that does not declare itself CppInterOp.
load_cache("${CPPINTEROP_BINARY_DIR}" READ_WITH_PREFIX _sub_ CMAKE_PROJECT_NAME)
if(NOT DEFINED _sub_CMAKE_PROJECT_NAME)
    message(FATAL_ERROR
        "No CMAKE_PROJECT_NAME in ${CPPINTEROP_BINARY_DIR}/CMakeCache.txt "
        "— the CppInterOp sub-configure did not complete")
endif()
if(NOT _sub_CMAKE_PROJECT_NAME STREQUAL "CppInterOp")
    message(FATAL_ERROR
        "The provided source path override declares project '${_sub_CMAKE_PROJECT_NAME}', "
        "not CppInterOp")
endif()
