# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(ROOT_PLATFORM win32)

#----Check the compiler that is used-----------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)

  set(ROOT_ARCHITECTURE win32gcc)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe  -Wall -W -Woverloaded-virtual")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -W")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

elseif(MSVC)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH "-wd4267")
    set(ROOT_ARCHITECTURE win64)
    set(MACHINE_ARCH X64)
  else()
    set(ROOT_ARCHITECTURE win32)
    set(MACHINE_ARCH X86)
  endif()

  math(EXPR VC_MAJOR "${MSVC_VERSION} / 100")
  math(EXPR VC_MINOR "${MSVC_VERSION} % 100")

  #---Select compiler flags----------------------------------------------------------------
  if(winrtdebug AND dev)
    set(ROOT_RTC_FLAGS "-RTC1")
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-MD -O2 -Ob2 -Z7 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "-MD -O1 -Ob1 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "-MD -O2 -Ob2 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG          "-MDd -Od -Ob0 -Z7 ${ROOT_RTC_FLAGS}")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-MD -O2 -Ob2 -Z7 -DNDEBUG")
  set(CMAKE_C_FLAGS_MINSIZEREL       "-MD -O1 -Ob1 -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE          "-MD -O2 -Ob2 -DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG            "-MDd -Od -Ob0 -Z7 ${ROOT_RTC_FLAGS}")

  if(winrtdebug)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      string(REGEX MATCH "-D_ITERATOR_DEBUG_LEVEL=0" result ${CMAKE_CXX_FLAGS_DEBUG})
      if(NOT result MATCHES "-D_ITERATOR_DEBUG_LEVEL=0")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_ITERATOR_DEBUG_LEVEL=0")
      endif()
    endif()
    set(BLDCXXFLAGS "-MDd")
    set(BLDCFLAGS "-MDd")
  else()
    set(BLDCXXFLAGS "-MD")
    set(BLDCFLAGS "-MD")
    string(REPLACE "-MDd" "-MD" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "-MDd" "-MD" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  endif()

  if(asan)
    set(ASAN_EXTRA_CXX_FLAGS /fsanitize=address /wd5072)
    set(ASAN_EXTRA_SHARED_LINKER_FLAGS "/InferASanLibs /incremental:no /DEBUG")
    set(ASAN_EXTRA_EXE_LINKER_FLAGS "/InferASanLibs /incremental:no /DEBUG")
  endif()

  if(CMAKE_PROJECT_NAME STREQUAL ROOT)
    set(CMAKE_CXX_FLAGS "-nologo -I${CMAKE_SOURCE_DIR}/cmake/win -Zc:__cplusplus -std:c++${CMAKE_CXX_STANDARD} -GR -FIw32pragma.h -FIsehmap.h ${BLDCXXFLAGS} -EHsc -W3 -wd4141 -wd4291 -wd4244 -wd4049 -wd4146 -wd4250 -wd4624 ${ARCH} -D_XKEYCHECK_H -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_C_FLAGS   "-nologo -I${CMAKE_SOURCE_DIR}/cmake/win -FIw32pragma.h -FIsehmap.h -GR ${BLDCFLAGS} -EHsc -W3 ${ARCH} -DNOMINMAX")
    if(CMAKE_CXX_STANDARD GREATER_EQUAL 17)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING -D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING")
    endif()
    if(win_broken_tests)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DR__ENABLE_BROKEN_WIN_TESTS")
    endif()
    if(llvm13_broken_tests)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DR__ENABLE_LLVM13_BROKEN_TESTS")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "-nologo -FIw32pragma.h -FIsehmap.h -Zc:__cplusplus -std:c++${CMAKE_CXX_STANDARD} -GR ${BLDCXXFLAGS} -EHsc -W3 -wd4244 ${ARCH}")
    set(CMAKE_C_FLAGS   "-nologo -FIw32pragma.h -FIsehmap.h -GR ${BLDCFLAGS} -EHsc -W3 ${ARCH}")
  endif()

  #---Set Linker flags----------------------------------------------------------------------
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ignore:4049,4206,4217,4221 -incremental:no")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -ignore:4049,4206,4217,4221 -incremental:no")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ignore:4049,4206,4217,4221 -incremental:no")

  string(TIMESTAMP CURRENT_YEAR "%Y")
  set(ROOT_RC_SCRIPT ${CMAKE_BINARY_DIR}/etc/root.rc)
  set(ROOT_MANIFEST ${CMAKE_BINARY_DIR}/etc/root-manifest.xml)

  foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )
  endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
else()
  message(FATAL_ERROR "There is no setup for compiler '${CMAKE_CXX_COMPILER}' on this Windows system up to now. Stop cmake at this point.")
endif()
