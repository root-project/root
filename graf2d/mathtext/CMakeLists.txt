# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_LINKER_LIBRARY(mathtext
  fontembed.cxx
  fontembedps.cxx
  mathrender.cxx
  mathrenderstyle.cxx
  mathrendertoken.cxx
  mathtext.cxx
  mathtextencode.cxx
  mathtextparse.cxx
  mathtextview.cxx
  TYPE
    STATIC
  NOINSTALL
)

check_cxx_compiler_flag(-Wbidirectional=none GCC_HAS_BIDIRECTIONAL_FLAG)
if(GCC_HAS_BIDIRECTIONAL_FLAG)
    set_source_files_properties(src/fontembed.cxx COMPILE_FLAGS "-Wbidirectional=none")
endif()
set_property(TARGET mathtext PROPERTY POSITION_INDEPENDENT_CODE ON)
