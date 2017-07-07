############################################################################
# CMakeLists.txt file for building ROOT gui/qtgsi package
# @author Pere Mato, CERN
############################################################################

include(${QT_USE_FILE})
add_definitions(-DQT3_SUPPORT= -DQT3_SUPPORT_CONSTRUCTOR= -DQT_DLL -DQT_THREAD_SUPPORT)
if(QT_QT3SUPPORT_INCLUDE_DIR)
  include_directories(${QT_QT3SUPPORT_INCLUDE_DIR})
endif()

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-uninitialized)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-shadow)

QT4_WRAP_CPP(mocfiles inc/TQCanvasMenu.h inc/TQRootApplication.h
                      inc/TQRootCanvas.h inc/TQRootDialog.h)

ROOT_GENERATE_DICTIONARY(G__QtGSI *.h MODULE QtGSI LINKDEF LinkDef.h OPTIONS -DQTVERS=0x${QT_VERSION_NUM} -Wno-deprecated DEPENDENCIES Gui Gpad)

ROOT_LINKER_LIBRARY(QtGSI *.cxx ${mocfiles} G__QtGSI.cxx LIBRARIES ${QT_LIBRARIES} ${QT_QT3SUPPORT_LIBRARY} DEPENDENCIES Gui Gpad)

ROOT_INSTALL_HEADERS()

