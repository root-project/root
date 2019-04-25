############################################################################
# CMakeLists.txt file for building ROOT build package
# @author Pere Mato, CERN
############################################################################

if(WIN32)
  ROOT_EXECUTABLE(bindexplib
    win/bindexplib/bindexplib.cxx
  )
endif()

ROOT_EXECUTABLE(rmkdepend
  rmkdepend/cppsetup.c
  rmkdepend/ifparser.c
  rmkdepend/include.c
  rmkdepend/main.c
  rmkdepend/mainroot.cxx
  rmkdepend/parse.c
  rmkdepend/pr.c
)

IF(CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
  file(COPY "gdbPrinters/"
       DESTINATION ${CMAKE_BINARY_DIR}/lib
       FILES_MATCHING PATTERN "*-gdb.py")

  file(GLOB PRETTY_PRINTERS "gdbPrinters/*-gdb.py")
  install(FILES ${PRETTY_PRINTERS}
          DESTINATION lib
          CONFIGURATIONS Debug RelWithDebInfo)
ENDIF()
