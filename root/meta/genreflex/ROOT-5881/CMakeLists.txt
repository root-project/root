
ROOTTEST_GENERATE_REFLEX_DICTIONARY(StringDict 
   stringIO_allClasses.h SELECTION stringIO_allClasses_selection.xml)


ROOTTEST_GENERATE_EXECUTABLE(writeTreeBinary writeTree.cpp
                             LIBRARIES Core RIO Tree)




ROOTTEST_ADD_TEST(execStringInputFiles
                  EXEC writeTreeBinary
                  DEPENDS ${GENERATE_REFLEX_TEST}
                  FIXTURES_SETUP execStringInputFiles)

ROOTTEST_ADD_TEST(execstringI
  MACRO execstringI.C
  OUTREF execstringI.ref
  DEPENDS execStringInputFiles
  FIXTURES_REQUIRED execStringInputFiles
)
