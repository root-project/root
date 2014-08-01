# Add tests for XML parsing. The files are divided in bunches of 30
# selection xmls. One bunch is analysed per test.

SET(XMLPATH ${CMAKE_CURRENT_SOURCE_DIR}/experimentsSelectionXMLs)
SET(THESCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/parseXMLs.py)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-0
                  MACRO ${THESCRIPT} OPTS 30 0 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-1
                  MACRO ${THESCRIPT} OPTS 30 1 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-2
                  MACRO ${THESCRIPT} OPTS 30 2 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-3
                  MACRO ${THESCRIPT} OPTS 30 3 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-4
                  MACRO ${THESCRIPT} OPTS 30 4 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-5
                  MACRO ${THESCRIPT} OPTS 30 5 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-6
                  MACRO ${THESCRIPT} OPTS 30 6 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-7
                  MACRO ${THESCRIPT} OPTS 30 7 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-8
                  MACRO ${THESCRIPT} OPTS 30 8 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-9
                  MACRO ${THESCRIPT} OPTS 30 9 ${XMLPATH} )
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-XMLParsing-10
                  MACRO ${THESCRIPT} OPTS 30 10 ${XMLPATH} )

