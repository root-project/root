// @(#)root/zip:$Id$
// Author: Sergey Linev   7 July 2014

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RZip
#define ROOT_RZip

#include "Rtypes.h"

extern "C" ULong_t R__crc32(ULong_t crc, const UChar_t* buf, UInt_t len);

extern "C" ULong_t R__memcompress(Char_t* tgt, ULong_t tgtsize, Char_t* src, ULong_t srcsize);

extern "C" void R__zipMultipleAlgorithm(Int_t cxlevel, Int_t *srcsize, Char_t *src, Int_t *tgtsize, Char_t *tgt, Int_t *irep, Int_t compressionAlgorithm);

extern "C" void R__unzip(Int_t *nin, UChar_t *bufin, Int_t *lout, char *bufout, Int_t *nout);

extern "C" int R__unzip_header(Int_t *nin, UChar_t *bufin, Int_t *lout);

enum { kMAXZIPBUF = 0xffffff };

#endif
