ROOTTEST_ADD_TEST(TabCom
                  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                  INPUT TabCom_input.txt
                  OUTREF TabCom.oref
                  ERRREF TabCom.eref
                  COPY_TO_BUILDDIR MyClass.h)
