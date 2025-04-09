# The rootmap file adds the required fwd declsas a side-effect, so disable that.
# Using cxxmodule prevents all fwd decls and thus doesn't exhibit the "dangling" fwd decl
# of the using declaration.
ROOTTEST_GENERATE_DICTIONARY(issue8499_dict issue8499.h LINKDEF LinkDef.h NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_ADD_TEST(issue8499
                  MACRO runissue8499.C
                  DEPENDS roottest-cling-dict-issue8499-issue8499_dict-build)
