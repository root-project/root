set(CTEST_BUILD_NAME
  ${CMAKE_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE})
enable_testing()

# LLVM builds (not installed llvm) provides gtest.
if (NOT TARGET gtest)
  include(GoogleTest)
endif()

set(gtest_libs gtest gtest_main)
# Clang prior than clang13 (I think) merges both gmock into gtest.
if (TARGET gmock)
  list(APPEND gtest_libs gmock gmock_main)
endif()

add_custom_target(CppInterOpUnitTests)
set_target_properties(CppInterOpUnitTests PROPERTIES FOLDER "CppInterOp tests")
add_dependencies(CppInterOpUnitTests clangCppInterOp)

set (TIMEOUT_VALUE 2400)
function(add_cppinterop_unittest name)
  add_executable(${name} EXCLUDE_FROM_ALL ${ARGN})
  add_dependencies(CppInterOpUnitTests ${name})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${GTEST_INCLUDE_DIR})
  set_property(TARGET ${name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(WIN32)
  target_link_libraries(${name} PUBLIC ${ARG_LIBRARIES} ${gtest_libs})
  set_property(TARGET ${name} APPEND_STRING PROPERTY LINK_FLAGS "${MSVC_EXPORTS}")
else()
  target_link_libraries(${name} PUBLIC ${ARG_LIBRARIES} ${gtest_libs} pthread)
endif()
  add_test(NAME cppinterop-${name} COMMAND ${name})
  set_tests_properties(cppinterop-${name} PROPERTIES
                       TIMEOUT "${TIMEOUT_VALUE}"
                       ENVIRONMENT "CPLUS_INCLUDE_PATH=${CMAKE_BINARY_DIR}/etc"
                       LABELS
                       DEPENDS)
  # FIXME: Just call gtest_add_tests this function is available.
  #gtest_add_tests(${name} "${Arguments}" AUTO)
endfunction()
 
add_subdirectory(CppInterOp)

add_custom_target(check-cppinterop COMMAND ${CMAKE_CTEST_COMMAND} -V --build-config $<CONFIG>
  DEPENDS CppInterOpUnitTests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(check-cppinterop PROPERTIES FOLDER "CppInterOp tests")
