set(py_sources
  cppyy_backend/__init__.py
  cppyy_backend/loader.py
)

set(d $ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/${runtimedir})
foreach(py_source ${py_sources})
  file(COPY clingwrapper/python/${py_source} DESTINATION ${localruntimedir}/cppyy_backend)
  install(FILES clingwrapper/python/${py_source} DESTINATION ${runtimedir}/cppyy_backend)
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile ${d}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile ${d}/${py_source})")
endforeach()

add_library(cppyy_backend SHARED clingwrapper/src/clingwrapper.cxx)
target_include_directories(cppyy_backend PRIVATE clingwrapper/src)
target_link_libraries(cppyy_backend Core ${CMAKE_DL_LIBS})

# cppyy uses ROOT headers from binary directory
add_dependencies(cppyy_backend move_headers)
target_include_directories(cppyy_backend PRIVATE ${CMAKE_BINARY_DIR}/include)

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy_backend)
install(TARGETS cppyy_backend EXPORT ROOTExports DESTINATION ${runtimedir})
