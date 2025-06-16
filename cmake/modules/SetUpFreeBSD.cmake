# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(ROOT_PLATFORM freebsd)

if(CMAKE_SYSTEM_PROCESSOR MATCHES x86_64 OR CMAKE_SYSTEM_PROCESSOR MATCHES amd64)
    set(ROOT_ARCHITECTURE freebsdamd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i686)
  set(FP_MATH_FLAGS "-msse2 -mfpmath=sse")
  set(ROOT_ARCHITECTURE freebsdi686)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i386) # FreeBSD port maintainer note: Treating i386 as i686 works
  set(FP_MATH_FLAGS "-msse2 -mfpmath=sse")
  set(ROOT_ARCHITECTURE freebsdi386)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
  set(ROOT_ARCHITECTURE freebsdarm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES arm)
  set(ROOT_ARCHITECTURE freebsdarm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES ppc64 OR
       CMAKE_SYSTEM_PROCESSOR MATCHES powerpc64 OR
       CMAKE_SYSTEM_PROCESSOR MATCHES powerpc64le)
  set(ROOT_ARCHITECTURE freebsdppc64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES s390x)
  set(ROOT_ARCHITECTURE freebsds390x)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES s390)
  set(ROOT_ARCHITECTURE freebsds390)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES riscv64)
  set(ROOT_ARCHITECTURE freebsdriscv64)
else()
  message(FATAL_ERROR "Unknown processor: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# JIT must be able to resolve symbols from all ROOT binaries.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# Set developer flags
if(dev)
  # Warnings are errors.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

  # Do not relink just because a dependent .so has changed.
  # I.e. relink only if a header included by the libs .o-s has changed,
  # whether or not that header "belongs" to a different .so.
  set(CMAKE_LINK_DEPENDS_NO_SHARED On)

  # Split debug info for faster builds.
  if(NOT gnuinstall)
    # We won't install DWARF files.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gsplit-dwarf")
  endif()

  if(_BUILD_TYPE_UPPER MATCHES "DEB")
    message(STATUS "Using ld.lld linker with gdb-index")
    set(GDBINDEX "-Wl,--gdb-index")
  else()
    message(STATUS "Using ld.lld linker without gdb-index")
  endif()

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${GDBINDEX}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GDBINDEX}")
  set(LLVM_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${GDBINDEX}")
  set(LLVM_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GDBINDEX}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe ${FP_MATH_FLAGS} -Wshadow -Wall -W -Woverloaded-virtual -fsigned-char -fsized-deallocation")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -W")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -Wl,--hash-style=\"both\"")

  if(asan)
    # See also core/sanitizer/README.md for what's happening.
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so OUTPUT_VARIABLE ASAN_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ASAN_EXTRA_CXX_FLAGS -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address)
    set(ASAN_EXTRA_SHARED_LINKER_FLAGS "-fsanitize=address -z undefs")
    set(ASAN_EXTRA_EXE_LINKER_FLAGS "-fsanitize=address -z undefs -Wl,--undefined=__asan_default_options -Wl,--undefined=__lsan_default_options -Wl,--undefined=__lsan_default_suppressions")
  endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe ${FP_MATH_FLAGS} -Wall -W -Woverloaded-virtual -fsigned-char -fsized-deallocation")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -W")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
  endif()

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

  if(asan)
    # See also core/sanitizer/README.md for what's happening.
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so OUTPUT_VARIABLE ASAN_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ASAN_EXTRA_CXX_FLAGS -fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
    set(ASAN_EXTRA_SHARED_LINKER_FLAGS "-fsanitize=address -static-libsan -z undefs")
    set(ASAN_EXTRA_EXE_LINKER_FLAGS "-fsanitize=address -static-libsan -z undefs -Wl,--undefined=__asan_default_options -Wl,--undefined=__lsan_default_options -Wl,--undefined=__lsan_default_suppressions")
  endif()
endif()
