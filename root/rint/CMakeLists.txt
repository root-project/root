if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64.*|x86.*|amd64.*|AMD64.*|i686.*|i386.*")
  # All platforms except of ARM/AARCH64
  ROOTTEST_ADD_TEST(TabCom
                    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driveTabCom.py
                    INPUT TabCom_input.txt
                    OUTCNV filterOpt.sh
                    OUTREF TabCom.oref
                    ERRREF TabCom.eref
                    COPY_TO_BUILDDIR MyClass.h)
endif()
