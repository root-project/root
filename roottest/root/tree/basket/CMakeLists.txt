#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_TEST(basket
                  MACRO rundropbasket.C
                  COPY_TO_BUILDDIR DataTest3_cel.root
                  OUTREF dropbasket.ref)
