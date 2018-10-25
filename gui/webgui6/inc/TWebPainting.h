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
#include "TColor.h"

#include <vector>
#include <string>

/** Object used to store paint operations and deliver them to JSROOT */
class TWebPainting : public TObject {

   protected:
      std::vector<std::string> fOper; /// list of operations
      Int_t fSize{0};                 ///<! filled buffer size
      TArrayF fBuf;                   /// array of points for all operations

      TAttLine fLastLine;             ///<! last line attributes
      TAttFill fLastFill;             ///<! last fill attributes
      TAttMarker fLastMarker;         ///<! last marker attributes

   public:

      TWebPainting();
      virtual ~TWebPainting() = default;

      Bool_t IsEmpty() const { return (fOper.size() == 0) && (fBuf.GetSize() == 0); }

      void AddOper(const std::string &oper) { fOper.emplace_back(oper); }

      void AddLineAttr(const TAttLine &attr);
      void AddFillAttr(const TAttFill &attr);
      void AddTextAttr(const TAttText &attr);
      void AddMarkerAttr(const TAttMarker &attr);

      Float_t *Reserve(Int_t sz);

      void AddColor(Int_t indx, TColor *col);

      // Set actual filled size
      void FixSize() { fBuf.Set(fSize); }

   ClassDef(TWebPainting, 1)// store for all paint operation of TVirtualPadPainter
};

#endif
