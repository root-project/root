# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/x3d package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(X3d
  HEADERS
    TViewerX3D.h
    TX3DFrame.h
  SOURCES
    src/TViewerX3D.cxx
    src/TX3DFrame.cxx
    ../../builtins/x3d/x3d.c
  LIBRARIES
    ${X11_LIBRARIES}
  DEPENDENCIES
    Graf3d
    Gui
)

# This ROOT package depends on a hard-coded builtin C library (x3d) that was forked in 1993 and slightly adapted for ROOT. See
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/inc/x3d.h
# https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/x3d/src/x3d.c
# vs http://hpux.connect.org.uk/hppd/hpux/X11/Viewers/x3d-2.2/ download source and rename from .tar.gz to just .tar

target_include_directories(X3d PRIVATE ${X11_INCLUDE_DIR} src/ ../../builtins/x3d/) # src for src/X3DDefs.h, ../../builtins/x3d/ for x3d.h
