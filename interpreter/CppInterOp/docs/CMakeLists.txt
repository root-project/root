find_package(Doxygen REQUIRED)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg)

set(docs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
set(docs_builddir ${CMAKE_CURRENT_BINARY_DIR})
set(cppinterop_srcdir ${CMAKE_SOURCE_DIR})
# file(READ ${CMAKE_SOURCE_DIR}/VERSION PACKAGE_VERSION)

configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)

add_custom_target(doxygen-cppinterop
                  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Generate CppInterOp documentation with Doxygen"
                  VERBATIM)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CreateSphinxTarget)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py 
               ${CMAKE_CURRENT_BINARY_DIR}/conf.py
               @ONLY
              )

create_sphinx_target(
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  TARGET_NAME sphinx-cppinterop
)