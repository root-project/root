ROOTTEST_ADD_TEST(TabCom
                  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                  INPUT TabCom_input.txt
                  OUTCNV filterOpt.sh
                  OUTREF TabCom.oref
                  ERRREF TabCom.eref
                  COPY_TO_BUILDDIR MyClass.h)
