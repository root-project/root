#execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated_selectors)
#execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/trees ${CMAKE_CURRENT_BINARY_DIR}/trees)

#ROOTTEST_COMPILE_MACRO(SampleClasses.h)
#ROOTTEST_ADD_TEST(Classes
#                  COPY_TO_BUILDDIR  SampleClasses.h
#                  MACRO runClasses.C
#                  OUTREF Classes.ref
#                  DEPENDS ${COMPILE_MACRO_TEST})

# It relies very heavily that these tests are run in the source directory
# so using the old method....

ROOTTEST_ADD_OLDTEST()
