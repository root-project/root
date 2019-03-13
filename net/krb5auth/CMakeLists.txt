############################################################################
# CMakeLists.txt file for building ROOT net/krb5auth package
# @author Pere Mato, CERN
############################################################################

include_directories(SYSTEM ${KRB5_INCLUDE_DIRS})
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS R__KRB5INIT="${KRB5_INIT}")

if(MACOSX_KRB5_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Krb5Auth
  HEADERS
    TKSocket.h
  SOURCES
    src/Krb5Auth.cxx
    src/TKSocket.cxx
  LIBRARIES
    ${KRB5_LIBRARIES}
    ${COMERR_LIBRARIES}
  DEPENDENCIES
    RootAuth
    Net
)
