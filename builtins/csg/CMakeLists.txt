# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# hard-coded builtin, forked first from SOLID in early 2000s (first by a Blender developer, then by Timur)
# See https://github.com/root-project/root/blob/813cca51897056cd187a63f1e8b5a8a8601d7605/gl/src/CsgOps.cxx
# vs https://sourceforge.net/projects/solid/files/
# Upstream is now https://github.com/dtecta/solid3
# not yet in Debian https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=500616

add_library(ROOTCsg STATIC)
target_sources(ROOTCsg PRIVATE src/CsgOps.cxx)
target_include_directories(ROOTCsg PUBLIC inc/)
set_property(TARGET ROOTCsg PROPERTY POSITION_INDEPENDENT_CODE ON)
