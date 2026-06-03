# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_LINKER_LIBRARY(mathtext
  fontembed.cc
  fontembedps.cc
  mathrender.cc
  mathrenderstyle.cc
  mathrendertoken.cc
  mathtext.cxx
  mathtextencode.cc
  mathtextparse.cc
  mathtextview.cc
  TYPE
    STATIC
  NOINSTALL
)

check_cxx_compiler_flag(-Wbidi-chars=none GCC_HAS_BIDI_CHARS_FLAG)
if(GCC_HAS_BIDI_CHARS_FLAG)
    set_source_files_properties(src/fontembed.cc COMPILE_FLAGS "-Wbidi-chars=none")
endif()
set_property(TARGET mathtext PROPERTY POSITION_INDEPENDENT_CODE ON)
