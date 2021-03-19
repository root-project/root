// Original Author: Oksana Shadura
/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ZipLZMAFL2
#define ROOT_ZipLZMAFL2

#ifdef __cplusplus
extern "C" {
#endif

void R__zipFLZMA2(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep);

void R__unzipFLZMA2(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

#ifdef __cplusplus
}
#endif

#endif
