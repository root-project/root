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

extern "C" unsigned long R__crc32(unsigned long crc, const unsigned char* buf, unsigned int len);

extern "C" unsigned long R__memcompress(char *tgt, unsigned long tgtsize, char *src, unsigned long srcsize);

extern "C" void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, int compressionAlgorithm);

extern "C" void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep);

extern "C" void R__unzip(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

extern "C" int R__unzip_header(int *srcsize, unsigned char *src, int *tgtsize);

enum { kMAXZIPBUF = 0xffffff };

#endif
