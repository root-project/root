# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Helper macro for checking binutils available in OS
macro(root_check_assembler)
   exec_program(${CMAKE_CXX_COMPILER} ARGS -print-prog-name=as OUTPUT_VARIABLE _as)
   mark_as_advanced(_as)
   if(NOT _as)
      message(WARNING "Could not find 'as', the assembler used by GCC.")
   else()
      exec_program(${_as} ARGS --version OUTPUT_VARIABLE _as_version)
      string(REGEX REPLACE "\\([^\\)]*\\)" "" _as_version "${_as_version}")
      string(REGEX MATCH "[1-9]\\.[0-9]+(\\.[0-9]+)?" _as_version "${_as_version}")
      if(_as_version VERSION_LESS "2.18.93")
         message(STATUS "OS binutils is too old (${_as_version}). Some ZLIB optimizations will be disabled.")
         set(ROOT_DEFINITIONS ${ROOT_DEFINITIONS} -DROOT_NO_AVX)
         set(ZLIB_AVX_INTRINSICS_BROKEN true)
      elseif(_as_version VERSION_LESS "2.21.0")
         message(STATUS "OS binutils is too old (${_as_version}) for AVX2 instructions.")
         set(ROOT_DEFINITIONS ${ROOT_DEFINITIONS} -DROOT_NO_AVX2)
         set(ZLIB_AVX2_INTRINSICS_BROKEN true)
      else()
         message(STATUS "Binutils as version: ${_as_version}")
      endif()
   endif()
endmacro()
