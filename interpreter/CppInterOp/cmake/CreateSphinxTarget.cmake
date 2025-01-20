# Implementation of 'create_sphinx_target' in this file is copied from 
# llvm implementation of 'AddSphinxTarget'.
# https://github.com/llvm/llvm-project/blob/main/llvm/cmake/modules/AddSphinxTarget.cmake

find_package(Sphinx REQUIRED)

function(create_sphinx_target)
  cmake_parse_arguments(SPHINX
                        "" # options
                        "SOURCE_DIR;TARGET_NAME"
                        "" # multi-value keywords
                        ${ARGN}
                      )
  set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)
  set(SPHINX_DOC_TREE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_doctrees)

  add_custom_target(${SPHINX_TARGET_NAME}
                    COMMAND
                    ${SPHINX_EXECUTABLE} -b html -d ${SPHINX_DOC_TREE_DIR} -q ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}
                    COMMENT
                    "Generating sphinx user documentation into \"${SPHINX_BUILD_DIR}\""
                    VERBATIM
                  )
  message(STATUS "Added ${SPHINX_TARGET_NAME} target")
endfunction()