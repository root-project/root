############################################################################
# CMakeLists.txt file for building ROOT bindings/pyroot package
############################################################################
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(tree/tree)
include_directories(${PYTHON_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__PyROOT *.h MODULE PyROOT LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

if(GCC_MAJOR EQUAL 4)
  add_definitions(-fno-strict-aliasing)
endif()

ROOT_LINKER_LIBRARY(PyROOT *.cxx G__PyROOT.cxx LIBRARIES RIO Net Hist Graf Graf3d Gpad Tree Matrix Thread ${PYTHON_LIBRARIES})
if(WIN32)
  set_target_properties(PyROOT PROPERTIES SUFFIX ".pyd")
endif()
if(MACOSX_MINOR EQUAL 5)
  ROOT_EXECUTABLE(python64 python64.c LIBRARIES  ${PYTHON_LIBRARIES})
endif()

#---Install python modules--------------------------------------------------
file(GLOB pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(pyfile ${pyfiles})
  install(FILES ${pyfile} DESTINATION ${runtimedir})
  install(CODE "execute_process(COMMAND python -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/${pyfile} )")
  install(CODE "execute_process(COMMAND python -O -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/${pyfile} )")
  if(testing)
    file(COPY ${pyfile} DESTINATION ${CMAKE_BINARY_DIR}/${runtimedir})
  endif()
endforeach()

#---Install headers----------------------------------------------------------
ROOT_INSTALL_HEADERS()

