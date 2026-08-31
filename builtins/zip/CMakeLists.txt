# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

add_library(ROOTzip STATIC)
target_sources(ROOTzip PUBLIC
  Bits.h # was renamed from Bits.c at https://github.com/root-project/root/commit/3b2f05c8ddf1001cc8eb05ad60b07b10af6ac293#diff-ff111e76990c56e7019ed0d837f1f615e280dc679ed393200c4f75b61094c5b7 and later split back to Bits.c in https://github.com/root-project/root/commit/25d5805abac4d408554e99aec6e49b8a671a2d96
  Tailor.h # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/inc/Tailor.h vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/tailor.h)
  ZIP.h  # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/inc/ZIP.h vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/zip.h)
PRIVATE
  Bits.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Bits.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/bits.c)
  ZDeflate.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Deflate.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/deflate.c)
  ZInflate.c # forked from 1994: version c14o / unzip-5.12 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Inflate.c vs https://github.com/freebsd/freebsd-src/commit/37c32e9f3a0dc8ba9d0b7cd06742502dcba1b457 or https://stuff.mit.edu/afs/sipb.mit.edu/project/sipbsrc/i386_nbsd1/unzip-5.12/inflate.c)
  ZTrees.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Trees.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/trees.c)
  ZipLZMA.c # ROOT
  ZipLZMA.h # ROOT, public
  ZipLZ4.c # ROOT
  ZipLZ4.h # ROOT, public
  ZipZSTD.cxx # ROOT
  ZipZSTD.h # ROOT, public
)

include(CheckSourceCompiles)
check_source_compiles(CXX "
#if !defined(__linux) && defined(__linux__) && !defined(linux) && !(defined(__CYGWIN__) && defined(__GNUC__))
#error \"This is not Linux\"
#endif
int main() { return 0; }
" IS_LINUX)
add_library(ROOTtls INTERFACE)
target_sources(ROOTtls PUBLIC ${CMAKE_SOURCE_DIR}/core/foundation/inc) # TODO move to builtins directory
target_compile_definitions(ROOTtls INTERFACE
$<$<BOOL:${CMAKE_USE_PTHREADS_INIT}>:R__HAS_PTHREAD>
$<$<BOOL:${IS_LINUX}>:R__LINUX>
$<$<PLATFORM_ID:FreeBSD>:R__FBSD>
$<$<PLATFORM_ID:Windows>:R__WIN32>
$<$<PLATFORM_ID:SunOS>:R__SOLARIS>
)
target_include_directories(ROOTtls INTERFACE ${CMAKE_SOURCE_DIR}/core/foundation/inc) # TODO move to builtins. could be instead a fileSET.

target_include_directories(ROOTzip PUBLIC .)
target_link_libraries(ROOTzip PRIVATE LibLZMA::LibLZMA xxHash::xxHash LZ4::LZ4 ZSTD::ZSTD ROOTtls)
set_target_properties(ROOTzip PROPERTIES POSITION_INDEPENDENT_CODE ON)
