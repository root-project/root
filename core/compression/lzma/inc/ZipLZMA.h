// @(#)root/lzma:$Id$
// Author: David Dagenhart   May 2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ZipLZMA
#define ROOT_ZipLZMA

#ifdef __cplusplus
extern "C" {
#endif

void R__zipLZMA(int cxlevel, int *srcsize, const char *src, int *tgtsize, char *tgt, int *irep);

void R__unzipLZMA(int *srcsize, const unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

#ifdef __cplusplus
}
#endif

#endif
