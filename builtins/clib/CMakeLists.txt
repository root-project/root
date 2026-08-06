# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  strlcpy.h # custom ROOT file (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/inc/strlcpy.h)
  snprintf.h # custom ROOT file (https://github.com/root-project/root/blob/3c0f42cd0d4f7aeab78e01533afefeb6f1248946/clib/inc/snprintf.h vs https://github.com/daztucker/historic-ssh/blob/658f5a469b028b54a9934db11fb43f65319bb811/snprintf.h)
  strtok.h # custom ROOT file (https://github.com/root-project/root/blob/d39c9211ba045c0b9aaa780ec76a32666e04cd1e/core/clib/inc/strtok.h)
)

ROOT_OBJECT_LIBRARY(Clib
  # All this code below is LGPL (GNU Lesser General Public License was called at that time GNU Library General Public License)
  src/attach.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/attach.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/attach.c)
  src/detach.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/detach.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/detach.c)
  src/getpagesize.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/getpagesize.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mvalloc.c)
  src/keys.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/keys.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/keys.c)
  src/mcalloc.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mcalloc.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mcalloc.c)
  src/mfree.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mfree.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mfree.c)
  src/mmalloc.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mmalloc.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmalloc.c)
  src/mmapsup.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mmapsup.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmap-sup.c)
  src/mmcheck.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mmcheck.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmcheck.c)
  src/mrealloc.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/mrealloc.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mrealloc.c)
  src/sbrksup.c # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/src/sbrksup.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/sbrk-sup.c)
  src/snprintf.c # forked in ~1998 maybe from Revision 1.19 / ssh 1.2.26 (https://github.com/root-project/root/blob/3c0f42cd0d4f7aeab78e01533afefeb6f1248946/clib/src/snprintf.c vs https://github.com/daztucker/historic-ssh/blob/ssh/snprintf.c)
  src/strlcat.c # forked in 2010 from OpenBSD (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/src/strlcat.c vs https://github.com/libressl/openbsd/blob/032d9a965adfb74e42b18dc939b34fcda2cf740e/src/lib/libc/string/strlcat.c)
  src/strlcpy.c # forked in 2010 from OpenBSD (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/src/strlcpy.c vs https://github.com/libressl/openbsd/blob/032d9a965adfb74e42b18dc939b34fcda2cf740e/src/lib/libc/string/strlcpy.c)
  # res/mmalloc.h (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmalloc.h vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmalloc.h)
  # res/mmconfig.h # custom ROOT file (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmconfig.h)
  # res/mmprivate.h # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmprivate.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmprivate.c)
)

target_include_directories(Clib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    ${CMAKE_BINARY_DIR}/ginclude
    ${CMAKE_CURRENT_SOURCE_DIR}/res
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/foundation/inc # for RConfig.hxx
)

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_sources(Core PRIVATE $<TARGET_OBJECTS:Clib>)

ROOT_INSTALL_HEADERS()
