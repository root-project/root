# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(RRawFile RRawFile.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TFile TFileTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TBufferFile TBufferFileTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TBufferMerger TBufferMerger.cxx LIBRARIES RIO Imt Tree)
ROOT_ADD_GTEST(TBufferJSON TBufferJSONTests.cxx LIBRARIES RIO)
ROOT_ADD_GTEST(TFileMerger TFileMergerTests.cxx LIBRARIES RIO Tree Hist)
if(ROOT_NEED_STDCXXFS)
  target_link_libraries(TFileMerger PUBLIC stdc++fs)
endif()
ROOT_ADD_GTEST(TROMemFile TROMemFileTests.cxx LIBRARIES RIO Tree)
if(uring AND NOT DEFINED ENV{ROOTTEST_IGNORE_URING})
  ROOT_ADD_GTEST(RIoUring RIoUring.cxx LIBRARIES RIO)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RFileTestIncludes
                              NO_INSTALL_HEADERS
                              HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/RFileTestIncludes.hxx
                              SOURCES RFileTestIncludes.cxx
                              LINKDEF RFileTestIncludesLinkDef.h
                              DEPENDENCIES RIO Tree)
configure_file(RFileTestIncludes.hxx . COPYONLY)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET RFileTestIncludes POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libRFileTestIncludes.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libRFileTestIncludes.dll)
endif()
ROOT_ADD_GTEST(rfile rfile.cxx LIBRARIES RIO Hist ROOTNTuple Tree RFileTestIncludes)
if(pyroot)
  ROOT_ADD_PYUNITTEST(rfile_py rfile.py)
endif()

ROOT_ADD_GTEST(ZipHeader ZipHeader.cxx LIBRARIES RIO Tree ROOTNTuple)

# Temporarily disabled.  Test routinely fails on MacOS and some Linuxes.
#if(NOT WIN32 AND (NOT MACOS_VERSION OR NOT MACOSX_VERSION VERSION_LESS 13.00))
#  ROOT_EXECUTABLE(TMapFileTest TMapFileTest.cxx LIBRARIES RIO Hist New)
#  ROOT_ADD_TEST(io-io-test-TMapFileTest COMMAND TMapFileTest complete)
#endif()

# Detect bitness.
# FIXME: This is also done in roottest/CMakeLists.txt which is included 'too late' for
# the tests here.  We should unify this logic and do it only once.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_64BIT 1)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(_32BIT 1)
endif()

if(NOT _32BIT)
  ROOT_ADD_GTEST(TBufferFileInnerReferences InnerReferencesTests.cxx LIBRARIES RIO)

  ROOTTEST_COMPILE_MACRO(testByteCount.cxx
                        FIXTURES_SETUP io-io-tobj-fixture)

  ROOTTEST_ADD_TEST(testLargeByteCounts
                    MACRO testByteCount.cxx+
                    OUTREF testByteCount.ref
                    FIXTURES_REQUIRED io-io-tobj-fixture)

  ROOTTEST_COMPILE_MACRO(testLargeCollection.cxx
                        FIXTURES_SETUP io-io-testcoll-fixture)

  # Memory-check script: exits 0 when enough RAM+swap is available, 77 (skip) otherwise.
  # RootTestDriver.cmake propagates exit code 77 from PRECMD to the test's own exit code,
  # and SKIP_RETURN_CODE 77 on the main test turns that into a proper CTest "Skipped".
  set(_memcheck_script "${CMAKE_CURRENT_BINARY_DIR}/check_mem_for_large_coll.sh")
  file(WRITE ${_memcheck_script}
    "#!/bin/sh\n"
    "# Exit 77 (CTest SKIP via SKIP_RETURN_CODE) if total RAM+swap < 32 GB.\n"
    "# We use TOTAL (not free) because the OS reclaims cached pages on demand.\n"
    "THRESHOLD_KB=33554432\n"
    "if [ -r /proc/meminfo ]; then\n"
    "  # Linux: use MemTotal + SwapTotal (kernel will reclaim caches as needed)\n"
    "  RAM_KB=\$(awk '/^MemTotal:/{print \$2}' /proc/meminfo)\n"
    "  SWAP_KB=\$(awk '/^SwapTotal:/{print \$2}' /proc/meminfo)\n"
    "  RAM_KB=\${RAM_KB:-0}\n"
    "  SWAP_KB=\${SWAP_KB:-0}\n"
    "  TOTAL_KB=\$(( RAM_KB + SWAP_KB ))\n"
    "elif [ \"\$(uname)\" = \"Darwin\" ]; then\n"
    "  # macOS: total physical RAM via sysctl hw.memsize (bytes -> kB)\n"
    "  RAM_BYTES=\$(sysctl -n hw.memsize 2>/dev/null)\n"
    "  RAM_KB=\$(( \${RAM_BYTES:-0} / 1024 ))\n"
    "  # sysctl output: 'total = NNN.NNM  used = ...  free = ...'\n"
    "  SWAP_TOTAL_MB=\$(sysctl -n vm.swapusage 2>/dev/null | sed 's/total = \\([0-9.]*\\)M.*/\\1/')\n"
    "  if [ -n \"\$SWAP_TOTAL_MB\" ]; then\n"
    "    SWAP_KB=\$(awk \"BEGIN{printf \\\"%d\\\", \$SWAP_TOTAL_MB * 1024}\")\n"
    "  else\n"
    "    SWAP_KB=0\n"
    "  fi\n"
    "  TOTAL_KB=\$(( RAM_KB + SWAP_KB ))\n"
    "else\n"
    "  echo \"testLargeCollections memory check: unsupported OS, assuming enough memory\"\n"
    "  exit 0\n"
    "fi\n"
    "echo \"testLargeCollections memory check: total \${TOTAL_KB} kB (RAM \${RAM_KB} + swap \${SWAP_KB}), need \${THRESHOLD_KB} kB\"\n"
    "if [ \"\$TOTAL_KB\" -lt \"\$THRESHOLD_KB\" ]; then\n"
    "  echo \"Insufficient memory -- skipping testLargeCollections\"\n"
    "  exit 77\n"
    "fi\n"
    "exit 0\n"
  )
  execute_process(COMMAND chmod +x ${_memcheck_script})

  ROOTTEST_ADD_TEST(testLargeCollections
                    TIMEOUT 1200
                    MACRO testLargeCollection.cxx+
                    OUTREF testLargeCollection.ref
                    FIXTURES_REQUIRED io-io-testcoll-fixture
                    PRECMD sh ${_memcheck_script}
                    PROPERTIES "SKIP_REGULAR_EXPRESSION;ROOTTEST_SKIP")
endif()
