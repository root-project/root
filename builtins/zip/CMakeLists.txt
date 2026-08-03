# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

target_sources(Core PRIVATE
  Bits.h # was renamed from Bits.c at https://github.com/root-project/root/commit/3b2f05c8ddf1001cc8eb05ad60b07b10af6ac293#diff-ff111e76990c56e7019ed0d837f1f615e280dc679ed393200c4f75b61094c5b7 and later split back to Bits.c in https://github.com/root-project/root/commit/25d5805abac4d408554e99aec6e49b8a671a2d96
  Tailor.h # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/inc/Tailor.h vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/tailor.h)
  ZIP.h  # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/inc/ZIP.h vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/zip.h)
  Bits.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Bits.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/bits.c)
  ZDeflate.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Deflate.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/deflate.c)
  ZInflate.c # forked from 1994: version c14o / unzip-5.12 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Inflate.c vs https://github.com/freebsd/freebsd-src/commit/37c32e9f3a0dc8ba9d0b7cd06742502dcba1b457 or https://stuff.mit.edu/afs/sipb.mit.edu/project/sipbsrc/i386_nbsd1/unzip-5.12/inflate.c)
  ZTrees.c # forked from 1993: zip 2.0 (https://github.com/root-project/root/blob/852600061bcacd9b255d44f6312c96b6b1e00a2d/zip/src/Trees.c vs https://github.com/pmqs/Info-ZIP-Family-Tree-for-Zip/blob/d25ed515bc56fae136b635d764ce8b363041aef5/trees.c)
)
