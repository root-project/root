# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

add_library(ROOTclib STATIC)

target_sources(ROOTclib PUBLIC
  inc/strlcpy.h # custom ROOT file (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/inc/strlcpy.h)
  inc/strtok.h # custom ROOT file (https://github.com/root-project/root/blob/d39c9211ba045c0b9aaa780ec76a32666e04cd1e/core/clib/inc/strtok.h)
)

target_sources(ROOTclib PRIVATE
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
  src/strlcat.c # forked in 2010 from OpenBSD (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/src/strlcat.c vs https://github.com/libressl/openbsd/blob/032d9a965adfb74e42b18dc939b34fcda2cf740e/src/lib/libc/string/strlcat.c)
  src/strlcpy.c # forked in 2010 from OpenBSD (https://github.com/root-project/root/blob/b48975de4dcbcd5f1ecf7f60c3f8e39c38a1f6e3/core/clib/src/strlcpy.c vs https://github.com/libressl/openbsd/blob/032d9a965adfb74e42b18dc939b34fcda2cf740e/src/lib/libc/string/strlcpy.c)
  # res/mmalloc.h (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmalloc.h vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmalloc.h)
  # res/mmconfig.h # custom ROOT file (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmconfig.h)
  # res/mmprivate.h # forked in 1999 from gdb-4 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/clib/inc/mmprivate.c vs https://github.com/gnutools/binutils-gdb/blob/c906108c21474dfb4ed285bcc0ac6fe02cd400cc/mmalloc/mmprivate.c)
)

include(CheckSourceCompiles)
check_source_compiles(C
    "#include <string.h>
     int main() {
         char dst[10];
         strlcpy(dst, \"a\", 10);
         return 0;
     }"
    HAS_STRLCPY
)


target_compile_definitions(ROOTclib PUBLIC $<$<BOOL:${HAS_STRLCPY}>:HAS_STRLCPY>)
if (NOT WIN32)
  target_compile_definitions(ROOTclib PRIVATE INVALID_HANDLE_VALUE=-1)
endif()
if (UNIX)
target_compile_definitions(ROOTclib PRIVATE R__HAVE_UNISTD_H)
endif()
target_compile_definitions(ROOTclib PRIVATE R__HAVE_MMAP)
target_compile_definitions(ROOTclib PRIVATE NO_SBRK_MALLOC)
target_compile_definitions(ROOTclib PRIVATE $<$<PLATFORM_ID:LynxOS>:R__LYNXOS>)
target_compile_definitions(ROOTclib PRIVATE $<$<PLATFORM_ID:GNU>:R__HURD>)

target_include_directories(ROOTclib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/res> # New requires those publicly
)
set_property(TARGET ROOTclib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_options(ROOTclib PRIVATE -fvisibility=hidden)
