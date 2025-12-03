# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT main package
# @author Pere Mato, CERN
############################################################################

if(NOT WIN32)
  ROOT_EXECUTABLE(rootn.exe rmain.cxx LIBRARIES New Core MathCore Rint CMAKENOEXPORT)
  if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    SET_TARGET_PROPERTIES(rootn.exe PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
  endif()
  ROOT_EXECUTABLE(roots.exe roots.cxx LIBRARIES Core MathCore CMAKENOEXPORT)
endif()
ROOT_EXECUTABLE(root.exe rmain.cxx LIBRARIES Core Rint)
if(MSVC)
  set(root_exports "/EXPORT:_Init_thread_abort /EXPORT:_Init_thread_epoch \
      /EXPORT:_Init_thread_footer /EXPORT:_Init_thread_header /EXPORT:_tls_index \
      /STACK:4000000 setargv.obj")
  if(MSVC_VERSION GREATER_EQUAL 1933)
    set(root_exports "${root_exports} /EXPORT:__std_find_trivial_1 \
        /EXPORT:__std_find_trivial_2 /EXPORT:__std_find_trivial_4 \
        /EXPORT:__std_find_trivial_8")
  endif()
  set_property(TARGET root.exe APPEND_STRING PROPERTY LINK_FLAGS ${root_exports})
  add_custom_command(TARGET root.exe POST_BUILD
                     COMMAND mt -manifest ${ROOT_MANIFEST} -outputresource:${CMAKE_BINARY_DIR}/bin/root.exe)
endif()

if(MSVC)
  ROOT_EXECUTABLE(hadd hadd.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore CMAKENOEXPORT)
  set_property(TARGET hadd APPEND_STRING PROPERTY LINK_FLAGS "setargv.obj")
else()
  ROOT_EXECUTABLE(hadd hadd.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore MultiProc CMAKENOEXPORT)
  if(ROOT_NEED_STDCXXFS)
    target_link_libraries(hadd PRIVATE stdc++fs)
  endif()
endif()
ROOT_EXECUTABLE(rootnb.exe nbmain.cxx LIBRARIES Core CMAKENOEXPORT)

#---ReadSpeed-------------------------------------------------------------------------------------
ROOT_EXECUTABLE(rootreadspeed src/readspeed.cxx LIBRARIES RIO Tree TreePlayer ReadSpeed CMAKENOEXPORT)

#---CreateHaddCommandLineOptions------------------------------------------------------------------
generateHeader(hadd
  ${CMAKE_SOURCE_DIR}/main/src/hadd-argparse.py
  ${CMAKE_BINARY_DIR}/ginclude/haddCommandLineOptionsHelp.h
)

if(fortran AND CMAKE_Fortran_COMPILER)
  ROOT_EXECUTABLE(g2root g2root.f LIBRARIES minicern CMAKENOEXPORT)
  set_target_properties(g2root PROPERTIES COMPILE_FLAGS "-w")
  ROOT_EXECUTABLE(h2root h2root.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread minicern CMAKENOEXPORT)

  # This is needed in particular for macOS, where the path of the GNU Fortran
  # library might not be in the default DYLD_LIBRARY_PATH, but it gets picked
  # up with enable_language(Fortran).
  if(APPLE AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    # Compute path to quadmath
    execute_process(
      COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgfortran.dylib
      OUTPUT_VARIABLE LIBGFORTRAN_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(LIBGFORTRAN_DIR "${LIBGFORTRAN_PATH}" DIRECTORY)
    set_property(TARGET h2root APPEND PROPERTY BUILD_RPATH "${LIBGFORTRAN_DIR}")
    set_property(TARGET h2root APPEND PROPERTY INSTALL_RPATH "${LIBGFORTRAN_DIR}")
  endif()
endif()

file(GLOB utils RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} python/root*)
foreach(rawUtilName ${utils})
  get_filename_component(utilName ${rawUtilName} NAME)
  if(NOT WIN32)
    # We need the .py only on Windows
    string(REPLACE ".py" "" utilName ${utilName})
    set(python python3)
  else()
    set(python python)
  endif()
  configure_file(${rawUtilName} ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${utilName} @ONLY)

  install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${utilName}
                               DESTINATION ${CMAKE_INSTALL_BINDIR}
                               RENAME ${utilName}
                               PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                                           GROUP_EXECUTE GROUP_READ
                                           WORLD_EXECUTE WORLD_READ
                               COMPONENT applications)
endforeach()

install(FILES python/cmdLineUtils.py DESTINATION ${runtimedir})
if(IS_ABSOLUTE ${runtimedir})
  set(absruntimedir ${runtimedir})
else()
  set(absruntimedir \${CMAKE_INSTALL_PREFIX}/${runtimedir})
endif()
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m py_compile \$ENV{DESTDIR}${absruntimedir}/cmdLineUtils.py)")
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -O -m py_compile \$ENV{DESTDIR}${absruntimedir}/cmdLineUtils.py)")
configure_file(python/cmdLineUtils.py ${localruntimedir}/cmdLineUtils.py @ONLY)


set_source_files_properties(src/rootcling.cxx PROPERTIES
  COMPILE_FLAGS "${CLING_CXXFLAGS}"
  VISIBILITY_INLINES_HIDDEN "ON"
)

set(rootcling_exe_names rootcling genreflex rootcint)

foreach(exe_name IN LISTS rootcling_exe_names)
  ROOT_EXECUTABLE(${exe_name} src/rootcling.cxx LIBRARIES RIO Cling Core Rint)

  # rootcling includes the ROOT complex header which would build the complex
  # dictionary with modules. To make sure that rootcling_stage1 builds this
  # dict before we use it, we add a dependency here.
  add_dependencies(${exe_name} complexDict)

  target_include_directories(${exe_name} PRIVATE
          ${CMAKE_SOURCE_DIR}/core/metacling/res
          ${CMAKE_SOURCE_DIR}/core/dictgen/res
          ${CMAKE_SOURCE_DIR}/io/rootpcm/res)
  set_property(TARGET ${exe_name} PROPERTY ENABLE_EXPORTS 1)
  if(WIN32)
    set_target_properties(${exe_name} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
    set_property(TARGET ${exe_name} APPEND_STRING PROPERTY LINK_FLAGS " -STACK:4000000")
  endif()
endforeach()

# To inherit the dependencies from rootcling
add_dependencies(genreflex rootcling)
add_dependencies(rootcint rootcint)

if (TARGET Gui)
  ROOT_EXECUTABLE(rootbrowse src/rootbrowse.cxx LIBRARIES RIO Core Rint Gui)
endif()
ROOT_EXECUTABLE(rootls src/rootls.cxx LIBRARIES RIO Tree Core Rint ROOTNTuple)

ROOT_ADD_TEST_SUBDIRECTORY(test)
