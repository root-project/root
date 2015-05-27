ROOTTEST_ADD_TEST(roottest-root-rint-TabCom
                  COMMAND root.exe -l -b
                  INPUT TabCom_input.txt
                  OUTREF TabCom.oref
                  ERRREF TabCom.eref
                  COPY_TO_BUILDDIR MyClass.h)

