// Author: Brian Bockelman March 2015

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

void R__zipLZ4(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep);

void R__unzipLZ4(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);
