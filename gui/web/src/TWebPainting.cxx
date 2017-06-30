// $Id$
// Author:  Sergey Linev  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TWebPainting.h"

ClassImp(TWebPainterAttributes)

ClassImp(TWebPainting)

Float_t *TWebPainting::Reserve(Int_t sz)
{
   if (fSize + sz > fBuf.GetSize()) {
      Int_t nextsz = fBuf.GetSize() + TMath::Max(1024, (sz/128 + 1) * 128);
      fBuf.Set(nextsz);
   }

   Float_t *res = fBuf.GetArray() + fSize;
   fSize += sz;
   return res; // return size where drawing can start
}

