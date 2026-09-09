# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1925 AND MSVC_VERSION LESS 1929)
  # FIXME: since Visual Studio v16.5.0 the /O2 optimization flag makes most of the roofit/roostats tests failing
  # Try to re-enable /O2 after the upgrade of llvm & clang, or when upgrading Visual Studio
  string(REPLACE "-O2" "-O1 -Oi" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(REPLACE "-O2" "-O1 -Oi" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

set(BASE_HEADERS
  ROOT/RCryptoRandom.hxx
  ROOT/RFloat16.hxx
  ROOT/TErrorDefaultHandler.hxx
  ROOT/TExecutorCRTP.hxx
  ROOT/TSequentialExecutor.hxx
  ROOT/StringConv.hxx
  ROOT/Buttons.h
  ROOT/Bytes.h
  ROOT/Byteswap.h
  ROOT/KeySymbols.h
  ROOT/MessageTypes.h
  Riostream.h
  ROOT/Rtypes.h
  ROOT/TApplication.h
  ROOT/TAtt3D.h
  ROOT/TAttAxis.h
  ROOT/TAttBBox2D.h
  ROOT/TAttBBox.h
  ROOT/TAttFill.h
  ROOT/TAttLine.h
  ROOT/TAttMarker.h
  ROOT/TAttPad.h
  ROOT/TAttText.h
  ROOT/TBase64.h
  ROOT/TBenchmark.h
  ROOT/TBuffer3D.h
  ROOT/TBuffer3DTypes.h
  ROOT/TBuffer.h
  ROOT/TColor.h
  ROOT/TColorGradient.h
  ROOT/TDatime.h
  ROOT/TDirectory.h
  ROOT/TEnv.h
  ROOT/TException.h
  ROOT/TExec.h
  ROOT/TFileCollection.h
  ROOT/TFileInfo.h
  ROOT/TFolder.h
  ROOT/TInetAddress.h
  ROOT/TMacro.h
  ROOT/TMathBase.h
  ROOT/TMD5.h
  ROOT/TMemberInspector.h
  ROOT/TMessageHandler.h
  ROOT/TNamed.h
  ROOT/TNotifyLink.h
  ROOT/TObject.h
  ROOT/TObjString.h
  ROOT/TParameter.h
  ROOT/TPluginManager.h
  ROOT/TPoint.h
  ROOT/TPRegexp.h
  ROOT/TProcessID.h
  ROOT/TProcessUUID.h
  ROOT/TQClass.h
  ROOT/TQCommand.h
  ROOT/TQConnection.h
  ROOT/TQObject.h
  ROOT/TRedirectOutputGuard.h
  ROOT/TRefCnt.h
  ROOT/TRef.h
  ROOT/TRegexp.h
  ROOT/TRemoteObject.h
  ROOT/TROOT.h
  ROOT/TRootIOCtor.h
  ROOT/TStopwatch.h
  ROOT/TStorage.h
  ROOT/TString.h
  ROOT/TStringLong.h
  ROOT/TStyle.h
  ROOT/TSysEvtHandler.h
  ROOT/TSystemDirectory.h
  ROOT/TSystemFile.h
  ROOT/TSystem.h
  ROOT/TTask.h
  ROOT/TThreadSlots.h
  ROOT/TTime.h
  ROOT/TTimer.h
  ROOT/TTimeStamp.h
  ROOT/TUri.h
  ROOT/TUrl.h
  ROOT/TUUID.h
  ROOT/TVersionCheck.h
  ROOT/TVirtualFFT.h
  ROOT/TVirtualGL.h
  ROOT/TVirtualMonitoring.h
  ROOT/TVirtualMapFile.h
  ROOT/TVirtualMutex.h
  ROOT/TVirtualPadEditor.h
  ROOT/TVirtualPad.h
  ROOT/TVirtualPadPainter.h
  ROOT/TVirtualPerfStats.h
  ROOT/TVirtualPS.h
  ROOT/TVirtualQConnection.h
  ROOT/TVirtualRWMutex.h
  ROOT/TVirtualTableInterface.h
  ROOT/TVirtualViewer3D.h
  ROOT/TVirtualX.h
)

set(BASE_SOURCES
  src/Match.cxx
  src/RCryptoRandom.cxx
  src/String.cxx
  src/Stringio.cxx
  src/TApplication.cxx
  src/TAtt3D.cxx
  src/TAttAxis.cxx
  src/TAttBBox2D.cxx
  src/TAttBBox.cxx
  src/TAttFill.cxx
  src/TAttLine.cxx
  src/TAttMarker.cxx
  src/TAttPad.cxx
  src/TAttText.cxx
  src/TBase64.cxx
  src/TBenchmark.cxx
  src/TBuffer3D.cxx
  src/TBuffer.cxx
  src/TColor.cxx
  src/TColorGradient.cxx
  src/TDatime.cxx
  src/TDirectory.cxx
  src/TEnv.cxx
  src/TErrorDefaultHandler.cxx
  src/TException.cxx
  src/TExec.cxx
  src/TFileCollection.cxx
  src/TFileInfo.cxx
  src/TFolder.cxx
  src/TInetAddress.cxx
  src/TListOfTypes.cxx
  src/TListOfTypes.h
  src/TMacro.cxx
  src/TMathBase.cxx
  src/TMD5.cxx
  src/TMemberInspector.cxx
  src/TMessageHandler.cxx
  src/TNamed.cxx
  src/TObject.cxx
  src/TObjString.cxx
  src/TParameter.cxx
  src/TPluginManager.cxx
  src/TPRegexp.cxx
  src/TProcessID.cxx
  src/TProcessUUID.cxx
  src/TQCommand.cxx
  src/TQConnection.cxx
  src/TQObject.cxx
  src/TRefCnt.cxx
  src/TRef.cxx
  src/TRegexp.cxx
  src/TRemoteObject.cxx
  src/TROOT.cxx
  src/TStopwatch.cxx
  src/TStorage.cxx
  src/TString.cxx
  src/TStringLong.cxx
  src/TStyle.cxx
  src/TSysEvtHandler.cxx
  src/TSystem.cxx
  src/TSystemDirectory.cxx
  src/TSystemFile.cxx
  src/TTask.cxx
  src/TTime.cxx
  src/TTimer.cxx
  src/TTimeStamp.cxx
  src/TUri.cxx
  src/TUrl.cxx
  src/TUUID.cxx
  src/TVirtualFFT.cxx
  src/TVirtualGL.cxx
  src/TVirtualMonitoring.cxx
  src/TVirtualMutex.cxx
  src/TVirtualPad.cxx
  src/TVirtualPadEditor.cxx
  src/TVirtualPadPainter.cxx
  src/TVirtualPerfStats.cxx
  src/TVirtualPS.cxx
  src/TVirtualViewer3D.cxx
  src/TVirtualX.cxx
)

set(RELATIVE_BASE_INC_HEADERS ${BASE_HEADERS})
list(TRANSFORM RELATIVE_BASE_INC_HEADERS PREPEND inc/)

# only here complete list of headers can be propogated to parent cmake file
set_property(TARGET Core APPEND PROPERTY DICT_HEADERS ${BASE_HEADERS})

target_sources(Core PRIVATE ${BASE_SOURCES})

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0") # https://discourse.cmake.org/t/file-set-xyz-is-listed-in-interface-file-sets-of-w-but-has-not-been-exported/9131/3
  target_sources(
      Core
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/ src/
      FILES
          ${RELATIVE_BASE_INC_HEADERS}
          src/TListOfTypes.h
  )
endif()

target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT> # bw compatibility
)
target_include_directories(Core PRIVATE ${CMAKE_SOURCE_DIR}/core/utils/inc)

if(ROOT_NEED_STDCXXFS)
    target_link_libraries(Core PRIVATE stdc++fs)
endif()

# This code about the LIB_CORE_NAME define is important for TROOT::GetSharedLibDir()
# On Linux, dl_iterate_phdr reports loaded libraries by their SONAME
# (e.g. libCore.so.6.38), not their full filename (libCore.so.6.38.02).
# On macOS, _dyld_get_image_name returns the actual filename.
# LIB_CORE_NAME must match what the dynamic linker reports.
if(soversion AND NOT APPLE)
  set(full_core_filename $<TARGET_SONAME_FILE_NAME:Core>)
else()
  set(full_core_filename $<TARGET_FILE_NAME:Core>)
endif()

# Absolue CMAKE_INSTALL_<dir> paths are discouraged in CMake, but some
# packagers use them anyway. So we support it.
if(IS_ABSOLUTE ${CMAKE_INSTALL_INCLUDEDIR})
  set(install_lib_to_include "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
    set(libdir "${CMAKE_INSTALL_LIBDIR}")
  else()
    set(libdir "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  endif()
  file(RELATIVE_PATH install_lib_to_include "${libdir}" "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
  unset(libdir)
endif()

target_compile_options(Core PRIVATE -DLIB_CORE_NAME=${full_core_filename}
                                    -DINSTALL_LIB_TO_INCLUDE="${install_lib_to_include}"
)
add_dependencies(Core ensure_build_tree_marker)

if(PCRE2_FOUND)
  target_link_libraries(Core PRIVATE PCRE2::PCRE2)
  set_source_files_properties(src/TPRegexp.cxx
    TARGET_DIRECTORY Core
    PROPERTIES COMPILE_DEFINITIONS USE_PCRE2)
else()
  target_link_libraries(Core PRIVATE PCRE::PCRE)
endif()

ROOT_INSTALL_HEADERS(${BASE_HEADER_DIRS})

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/man)

if(NOT MSVC)
#---addRootC++CmdMan-------------------------------------------------------------------------
generateManual(rootMan ${CMAKE_CURRENT_SOURCE_DIR}/src/root-argparse.py ${CMAKE_BINARY_DIR}/man/root.1)
generateManual(haddMan ${CMAKE_SOURCE_DIR}/main/src/hadd-argparse.py ${CMAKE_BINARY_DIR}/man/hadd.1)
generateManual(rootclingMan ${CMAKE_SOURCE_DIR}/core/dictgen/src/rootcling-argparse.py ${CMAKE_BINARY_DIR}/man/rootcling.1)

#---addRootPyCmdMan---------------------------------------------------------------------------
#generateManual(rootcpMan ${CMAKE_SOURCE_DIR}/main/python/rootcp.py ${CMAKE_BINARY_DIR}/man/rootcp.1)
#generateManual(rootdrawtreeMan ${CMAKE_SOURCE_DIR}/main/python/rootdrawtree.py ${CMAKE_BINARY_DIR}/man/rootdrawtree.1)
#generateManual(rooteventselectorMan ${CMAKE_SOURCE_DIR}/main/python/rooteventselector.py ${CMAKE_BINARY_DIR}/man/rooteventselector.1)
#generateManual(rootmkdirMan ${CMAKE_SOURCE_DIR}/main/python/rootmkdir.py ${CMAKE_BINARY_DIR}/man/rootmkdir.1)
#generateManual(rootmvMan ${CMAKE_SOURCE_DIR}/main/python/rootmv.py ${CMAKE_BINARY_DIR}/man/rootmv.1)
#generateManual(rootprintMan ${CMAKE_SOURCE_DIR}/main/python/rootprint.py ${CMAKE_BINARY_DIR}/man/rootprint.1)
#generateManual(rootrmMan ${CMAKE_SOURCE_DIR}/main/python/rootrm.py ${CMAKE_BINARY_DIR}/man/rootrm.1)
#generateManual(rootslimtreeMan ${CMAKE_SOURCE_DIR}/main/python/rootslimtree.py ${CMAKE_BINARY_DIR}/man/rootslimtree.1)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)

if(UNIX)
  # Determine cryptographic random number generator

  CHECK_CXX_SOURCE_COMPILES("#include <cstdlib>
  int main() { char buf[32]; arc4random_buf(buf, 32); return 0;}" found_arc4)

  if(found_arc4)
     message(STATUS "Found arc4random_buf in stdlib.h")
     target_compile_definitions(Core PRIVATE R__ARC4_STDLIB)
  else()
    set(OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
    set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    if(DEFINED LIBBSDROOT)
       set(CMAKE_REQUIRED_INCLUDES ${LIBBSDROOT}/include)
       set(CMAKE_REQUIRED_LIBRARIES ${LIBBSDROOT}/lib/libbsd.so)
    endif()
    CHECK_CXX_SOURCE_COMPILES("#include <bsd/stdlib.h>
    int main() { char buf[32]; arc4random_buf(buf, 32); return 0;}" found_arc4_bsd)
    set(CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})
    set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
    if(found_arc4_bsd)
      message(STATUS "Found arc4random_buf in bsd/stdlib.h")
      target_compile_definitions(Core PRIVATE R__ARC4_BSDLIB)
      if(DEFINED LIBBSDROOT)
        target_include_directories(Core PRIVATE ${LIBBSDROOT}/include)
        target_link_libraries(Core PRIVATE ${LIBBSDROOT}/lib/libbsd.so)
      endif()
    else()
      CHECK_CXX_SOURCE_COMPILES("#include <sys/random.h>
      int main() { char buf[32]; int res = getrandom(buf, 32, GRND_NONBLOCK); return 0;}" found_getrandom)
      if(found_getrandom)
        message(STATUS "Found getrandom in sys/random.h")
        target_compile_definitions(Core PRIVATE R__GETRANDOM_CLIB)
      else()
        CHECK_CXX_SOURCE_RUNS("
        #include <fstream>

        int main() {
            std::ifstream urandom{\"/dev/urandom\"};
            if (!urandom) {
                // This will make the CMake command fail
                return 1;
            }

            constexpr int len{32};
            char buf[len];
            for (int n = 0; n < len; n++) buf[n] = 0;
            urandom.read(buf, len);

            int nmatch = 0;
            for (int n = 0; n < len; n++)
                if (buf[n] == 0) nmatch++;

            // Fail if no values have changed
            return nmatch != len ? 0 : 1;
        }" found_urandom)
        if(found_urandom)
          message(STATUS "Found random device in /dev/urandom")
          target_compile_definitions(Core PRIVATE R__USE_URANDOM)
        else()
          message(FATAL_ERROR "Fail to detect cryptographic random generator")
        endif()
      endif()
    endif()
  endif()
endif(UNIX)
