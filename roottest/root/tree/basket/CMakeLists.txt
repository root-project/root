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

# corrupted.root contains a TTree "tree" with a corrupted TBasket that has fNevBufSize == 4 and fNevBuf == 4210752.
# Since fNevBuf is much larger than fNevBufSize this would cause oob reads unless TBasket properly validates this.
# Verify that it does.
ROOTTEST_ADD_TEST(corrupted_basket
                  COPY_TO_BUILDDIR corrupted.root
                  COMMAND ${ROOT_root_CMD} -q -l -e "(new TFile(\"corrupted.root\"))->template Get<TTree>(\"tree\")->GetEntry(0)"
                  PASSREGEX "Inconsistent length for the entry offset buffer \\(4210752 events for a buffer size of 4\\)")
