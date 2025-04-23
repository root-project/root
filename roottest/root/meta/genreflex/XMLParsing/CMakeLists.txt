# Add tests for XML parsing. The files are divided in bunches of 30
# selection xmls. One bunch is analysed per test.

SET(XMLPATH ${CMAKE_CURRENT_SOURCE_DIR}/experimentsSelectionXMLs)
SET(THESCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/parseXMLs.py)

ROOTTEST_ADD_TEST(0
                  MACRO ${THESCRIPT} OPTS 6 0 ${XMLPATH} )
ROOTTEST_ADD_TEST(1
                  MACRO ${THESCRIPT} OPTS 6 1 ${XMLPATH} )
ROOTTEST_ADD_TEST(2
                  MACRO ${THESCRIPT} OPTS 6 2 ${XMLPATH} )
ROOTTEST_ADD_TEST(3
                  MACRO ${THESCRIPT} OPTS 6 3 ${XMLPATH} )
ROOTTEST_ADD_TEST(4
                  MACRO ${THESCRIPT} OPTS 6 4 ${XMLPATH} )
