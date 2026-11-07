# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_library(BUILTIN_MATHTEXT STATIC
  fontembed.cc
  fontembedpdf.cc
  fontembedps.cc
  geometry.cc
  mathrender.cc
  mathrenderstyle.cc
  mathrendertoken.cc
  mathtext.cc
  mathtextencode.cc
  mathtextparse.cc
  mathtextview.cc
)
target_include_directories(BUILTIN_MATHTEXT PUBLIC ./) # for mathtext/fontembed.h  mathtext/mathrender.h  mathtext/mathtext.h mathtext/geometry.h

check_cxx_compiler_flag(-Wbidi-chars=none GCC_HAS_BIDI_CHARS_FLAG)
if(GCC_HAS_BIDI_CHARS_FLAG)
    set_source_files_properties(fontembed.cc COMPILE_FLAGS "-Wbidi-chars=none")
endif()
if (WIN32)
  if (CMAKE_C_BYTE_ORDER STREQUAL "LITTLE_ENDIAN" OR CMAKE_CXX_BYTE_ORDER STREQUAL "LITTLE_ENDIAN")
    target_compile_definitions(BUILTIN_MATHTEXT PRIVATE LITTLE_ENDIAN=1)
  endif()
endif()
set_property(TARGET BUILTIN_MATHTEXT PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(mathtext::mathtext ALIAS BUILTIN_MATHTEXT)
