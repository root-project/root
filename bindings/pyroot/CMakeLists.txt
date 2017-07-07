############################################################################
# CMakeLists.txt file for building ROOT bindings/pyroot package
############################################################################
include_directories(${PYTHON_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__PyROOT *.h MODULE PyROOT LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Core MathCore)

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -fno-strict-aliasing)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-parentheses-equality)

if(WIN32)
  set(ROOTULIBS -include:_G__cpp_setupG__Net
                -include:_G__cpp_setupG__IO
                -include:_G__cpp_setupG__Tree
                -include:_G__cpp_setupG__Thread
                -include:_G__cpp_setupG__MathCore)
endif()
ROOT_LINKER_LIBRARY(PyROOT *.cxx G__PyROOT.cxx LIBRARIES Core Net Tree MathCore Rint ${PYTHON_LIBRARIES})
ROOT_LINKER_LIBRARY(JupyROOT ../JupyROOT/src/*.cxx LIBRARIES Core)


if(MSVC)
  add_custom_command(TARGET PyROOT POST_BUILD
                     COMMAND link -dll -nologo -IGNORE:4001 -machine:ix86 -export:initlibPyROOT
                    $<TARGET_LINKER_FILE:PyROOT> -nodefaultlib kernel32.lib msvcrt.lib
                    -out:$<TARGET_FILE_DIR:PyROOT>/libPyROOT.pyd)
  install(FILES  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libPyROOT.pyd
                 DESTINATION ${CMAKE_INSTALL_BINDIR}
                 COMPONENT libraries)
endif()
if(MACOSX_MINOR EQUAL 5)
  ROOT_EXECUTABLE(python64 python64.c LIBRARIES  ${PYTHON_LIBRARIES})
endif()

#---Install python modules--------------------------------------------------
file(GLOB pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(pyfile ${pyfiles})
  install(FILES ${pyfile} DESTINATION ${runtimedir})
  if(IS_ABSOLUTE ${runtimedir})
    set(absruntimedir ${runtimedir})
  else()
    set(absruntimedir \${CMAKE_INSTALL_PREFIX}/${runtimedir})
  endif()
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile \$ENV{DESTDIR}${absruntimedir}/${pyfile})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile \$ENV{DESTDIR}${absruntimedir}/${pyfile})")
  file(COPY ${pyfile} DESTINATION ${localruntimedir})
endforeach()

set(JupyROOTDirName "JupyROOT")
install (DIRECTORY ${JupyROOTDirName} DESTINATION ${runtimedir})
file(COPY ${JupyROOTDirName} DESTINATION ${localruntimedir})

set( JsMVADirName "JsMVA")
install (DIRECTORY ${JsMVADirName} DESTINATION ${runtimedir})
file(COPY ${JsMVADirName} DESTINATION ${localruntimedir})

#---Install headers----------------------------------------------------------
ROOT_INSTALL_HEADERS()
