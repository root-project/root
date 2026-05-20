set(CTEST_BUILD_NAME
  ${CMAKE_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE})
enable_testing()

# LLVM builds (not installed llvm) provides gtest.
if (NOT TARGET GTest::gtest AND NOT TARGET gtest)
  find_package(GTest)
  if (NOT GTest_FOUND)
    message(WARNING "CppInterOp could not find GTest. Provide the package or set CPPINTEROP_ENABLE_TESTING=OFF to disable this warning.")
    return()
  endif()
endif()

if(EMSCRIPTEN)
  if (TARGET GTest::gtest)
    # Target names in CMake >= v3.23
    set(gtest_libs GTest::gtest GTest::gmock)
  else()
    set(gtest_libs gtest gmock)
  endif()
else()
  if (TARGET GTest::gtest)
    # Target names in CMake >= v3.23
    set(gtest_libs GTest::gtest GTest::gmock GTest::gtest_main)
  else()
    set(gtest_libs gtest gtest_main gmock)
  endif()
  set(link_pthreads_lib pthread)
endif()

add_custom_target(CppInterOpUnitTests)
set_target_properties(CppInterOpUnitTests PROPERTIES FOLDER "CppInterOp tests")
add_dependencies(CppInterOpUnitTests clangCppInterOp)

set (TIMEOUT_VALUE 2400)
function(add_cppinterop_unittest name)
  cmake_parse_arguments(ARG
    "ENABLE_DISPATCH"
    ""
    ""
    ${ARGN}
  )
  add_executable(${name} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  add_dependencies(CppInterOpUnitTests ${name})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${GTEST_INCLUDE_DIR})
  set_property(TARGET ${name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  export_executable_symbols(${name})

  if(WIN32)
    target_link_libraries(${name} PUBLIC ${ARG_LIBRARIES} ${gtest_libs})
    set_property(TARGET ${name} APPEND_STRING PROPERTY LINK_FLAGS "${MSVC_EXPORTS}")
  else()
    target_link_libraries(${name} PRIVATE ${ARG_LIBRARIES} ${gtest_libs} ${link_pthreads_lib})
  endif()
  target_link_libraries(${name} PRIVATE clangCppInterOp)
  if(EMSCRIPTEN)
    if (BUILD_SHARED_LIBS)
      target_compile_definitions(${name} PRIVATE "EMSCRIPTEN_SHARED_LIBRARY")
    else()
      target_compile_definitions(${name} PRIVATE "EMSCRIPTEN_STATIC_LIBRARY")
    endif()
    # Without this cmake will try and get node to run the html file.
    # This guarantees that it runs the js file, and uses emsdks node.
    add_test(NAME cppinterop-${name} COMMAND $ENV{EMSDK_NODE} ${name}.js)
  else()
    add_test(NAME cppinterop-${name} COMMAND ${name})
    set(EXTRA_PATH_TEST_BINARIES /bin/$<CONFIG>/)
  endif()
  set_output_directory(${name}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
    LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${EXTRA_PATH_TEST_BINARIES}
  )

  set_tests_properties(cppinterop-${name} PROPERTIES
                        TIMEOUT "${TIMEOUT_VALUE}"
                        ENVIRONMENT "CPLUS_INCLUDE_PATH=${CMAKE_BINARY_DIR}/etc"
                        LABELS
                        DEPENDS)
  # Auto-generate the dispatch version if shared libs are built by recursively calling
  # add_cppinterop_unittest with the necessary compile definitions
  # FIXME: Enable for WASM builds
  if (NOT EMSCRIPTEN AND ARG_ENABLE_DISPATCH AND BUILD_SHARED_LIBS)
    add_cppinterop_unittest(${name}Dispatch ${ARG_UNPARSED_ARGUMENTS} DispatchTest.cpp)
    target_compile_definitions(${name}Dispatch PRIVATE 
      ENABLE_DISPATCH_TESTS
      CPPINTEROP_LIB_PATH="${CMAKE_BINARY_DIR}/lib/libclangCppInterOp${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
  endif()
endfunction()

add_subdirectory(CppInterOp)

add_custom_target(check-cppinterop COMMAND ${CMAKE_CTEST_COMMAND} -V --build-config $<CONFIG>
  DEPENDS CppInterOpUnitTests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(check-cppinterop PROPERTIES FOLDER "CppInterOp tests")
