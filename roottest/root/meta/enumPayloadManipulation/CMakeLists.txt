# This particular test has to be put in a separate directory, otherwise, for a reason
# which I cannot figure out, CMake will pass multiple times the "-D"s relative to the
# cling workarounds to genreflex. Given that other tests check some properties of the
# fwd declaration payload in the dictionaries, this "feature" makes them fail.
# In a sense, this is a workaround to have a sane reproducer of ROOT-9660 which does
# not interfere with the other tests.

# ROOT-9660
ROOTTEST_GENERATE_REFLEX_DICTIONARY(checkEnumFwdDeclxDict
                                    x.h
                                    SELECTION selection_x.xml
                                    FIXTURES_SETUP root-meta-enumPayloadManipulation-x-fixture)
ROOTTEST_GENERATE_REFLEX_DICTIONARY(checkEnumFwdDeclyDict
                                    y.h
                                    SELECTION selection_y.xml
                                    FIXTURES_SETUP root-meta-enumPayloadManipulation-y-fixture)

if(pyroot)
  ROOTTEST_ADD_TEST(checkEnumFwdDecl
                    MACRO checkEnumFwdDecl.py
                    OUTREF empty.ref
                    ERRREF empty.ref
                    FIXTURES_REQUIRED root-meta-enumPayloadManipulation-x-fixture
                                      root-meta-enumPayloadManipulation-y-fixture)
endif()

