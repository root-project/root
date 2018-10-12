// Author:  Sergey Linev, GSI  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWebPainting
#define ROOT_TWebPainting

#include "TList.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"
#include "TAttMarker.h"
#include "TArrayF.h"

/** Class to store actual drawing attributes */
class TWebPainterAttributes : public TObject, public TAttFill, public TAttLine, public TAttMarker, public TAttText  {
   public:
      virtual ~TWebPainterAttributes() = default;

      ClassDef(TWebPainterAttributes,1) // different draw attributes used by the painter
};

/** Object used to store paint operations and deliver them to JSROOT */
class TWebPainting : public TObject {

   protected:
      TList   fOper;                /// list of last draw operations
      Int_t   fSize{0};             ///!< filled buffer size
      TArrayF fBuf;                 /// array of points for all operations

   public:

      TWebPainting()  { fOper.SetOwner(kTRUE); }
      virtual ~TWebPainting() { fOper.Delete(); }

      void Add(TObject *obj, const char *opt) { fOper.Add(obj, opt); }
      Float_t *Reserve(Int_t sz);

      // Set actual filled size
      void FixSize() { fBuf.Set(fSize); }

   ClassDef(TWebPainting, 1)// store for all paint operation of TVirtualPadPainter
};

#endif
