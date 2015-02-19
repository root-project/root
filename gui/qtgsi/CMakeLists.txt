############################################################################
# CMakeLists.txt file for building ROOT gui/qtgsi package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(gui/gui)

include(${QT_USE_FILE})
add_definitions(-DQT3_SUPPORT= -DQT_DLL -DQT_THREAD_SUPPORT )
include_directories(${QT_QT3SUPPORT_INCLUDE_DIR})

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wno-deprecated-register -Wno-uninitialized)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  add_definitions(-Wno-deprecated -Wno-uninitialized)
endif()

QT4_WRAP_CPP(mocfiles inc/TQCanvasMenu.h inc/TQRootApplication.h
                      inc/TQRootCanvas.h inc/TQRootDialog.h)

ROOT_GENERATE_DICTIONARY(G__QtGSI *.h MODULE QtGSI LINKDEF LinkDef.h OPTIONS -DQTVERS=0x${QT_VERSION_NUM} -Wno-deprecated)

ROOT_LINKER_LIBRARY(QtGSI *.cxx ${mocfiles} G__QtGSI.cxx LIBRARIES ${QT_LIBRARIES} ${QT_QT3SUPPORT_LIBRARY} DEPENDENCIES Gui Gpad)

ROOT_INSTALL_HEADERS()

