if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*|x86.*|amd64.*|AMD64.*|i686.*|i386.*")
  # All platforms except of ARM/AARCH64
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
endif()

# ROOT-10425
ROOTTEST_ADD_TEST(missingSymbol
                 MACRO missingSymbol.C
                 PASSREGEX "You are probably missing the definition of.*MissingSymbol.*")

if (NOT MSVC)
# ROOT-5843
ROOTTEST_ADD_TEST(dotLibraries
                  COMMAND root -b -q -l -e ".libraries"
                  PASSREGEX "libRint")
endif()

# issue 11707
ROOTTEST_ADD_TEST(macroCalledMain
                  COPY_TO_BUILDDIR main.cpp
                  MACRO main.cpp)
