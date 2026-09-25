# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_EXECUTABLE(rmkdepend
# This was forked in 1996 and slightly adapated for ROOT, see https://github.com/root-project/root/tree/852600061bcacd9b255d44f6312c96b6b1e00a2d/build/rmkdepend vs https://ftp.xfree86.org/pub/XFree86/3.3.6/source/X336src-1.tgz (xc/config/makedepend subfolder)
  cppsetup.c
  ifparser.c
  include.c
  main.c
  mainroot.cxx
  parse.c
  pr.c
)
