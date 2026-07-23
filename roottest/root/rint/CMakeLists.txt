if (NOT MSVC)
# ROOT-5843
ROOTTEST_ADD_TEST(dotLibraries
                  COMMAND $<TARGET_FILE:root> -b -q -e ".libraries"
                  PASSREGEX "libRint")

ROOTTEST_ADD_TEST(TabCom
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                  INPUT TabCom_input.txt
                  OUTCNV filterOpt.sh
                  OUTREF TabCom.oref
                  ERRREF TabCom.eref
                  COPY_TO_BUILDDIR MyClass.h)

ROOTTEST_ADD_TEST(BackslashNewline
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                  INPUT BackslashNewline_input.txt)

# issue 6439
if(dataframe)
  ROOTTEST_ADD_TEST(failedCompilationBadState_1
                    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                    INPUT 6439_1_input.txt
                    FAILREGEX "You are probably missing the definition|Assertion")
endif()

ROOTTEST_ADD_TEST(failedCompilationBadState_2
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                  INPUT 6439_2_input.txt
                  FAILREGEX "You are probably missing the definition|Assertion")

# issue 7970: The test disabled since the bug is still there on many platforms
#ROOTTEST_ADD_TEST(failedCompilationBadState_3
#                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
#                  INPUT 7970_input.txt
#                  FAILREGEX "You are probably missing the definition|Assertion")
endif()

# ROOT-10425
ROOTTEST_ADD_TEST(missingSymbol
                 MACRO missingSymbol.C
                 PASSREGEX "You are probably missing the definition of.*MissingSymbol.*")

# issue 11707
ROOTTEST_ADD_TEST(macroCalledMain
                  COPY_TO_BUILDDIR main.cpp
                  MACRO main.cpp)
