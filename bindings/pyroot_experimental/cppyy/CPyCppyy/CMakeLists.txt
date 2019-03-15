set(headers
  include/TPyArg.h
  include/TPyException.h
  include/TPyReturn.h
  include/TPython.h
)

set(sources
  src/CallContext.cxx
  src/Converters.cxx
  src/CPPClassMethod.cxx
  src/CPPConstructor.cxx
  src/CPPDataMember.cxx
  src/CPPFunction.cxx
  src/CPPInstance.cxx
  src/CPPMethod.cxx
  src/CPPOverload.cxx
  src/CPPScope.cxx
  src/CPPSetItem.cxx
  src/CPyCppyyModule.cxx
  src/CustomPyTypes.cxx
  src/Dispatcher.cxx
  src/Executors.cxx
  src/LowLevelViews.cxx
  src/MemoryRegulator.cxx
  src/ProxyWrappers.cxx
  src/PyStrings.cxx
  src/Pythonize.cxx
  src/TemplateProxy.cxx
  src/TPyArg.cxx
  src/TPyClassGenerator.cxx
  src/TPyException.cxx
  src/TPyReturn.cxx
  src/TPython.cxx
  src/TupleOfInstances.cxx
  src/TypeManip.cxx
  src/Utility.cxx
)

add_library(cppyy SHARED ${headers} ${sources})
target_include_directories(cppyy PRIVATE ${CMAKE_BINARY_DIR}/include) # needed for string_view backport

target_compile_options(cppyy PRIVATE
  -Wno-shadow -Wno-strict-aliasing -Wno-unused-but-set-parameter)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 8)
  target_compile_options(cppyy PRIVATE -Wno-cast-function-type)
endif()

target_include_directories(cppyy PUBLIC ${PYTHON_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(cppyy cppyy_backend ${PYTHON_LIBRARIES})

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy)
install(TARGETS cppyy EXPORT ${CMAKE_PROJECT_NAME}Exports DESTINATION ${runtimedir})

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include)
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
