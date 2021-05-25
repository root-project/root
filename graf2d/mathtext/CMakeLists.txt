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

set_property(TARGET mathtext PROPERTY POSITION_INDEPENDENT_CODE ON)
