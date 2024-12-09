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

#include "TObject.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttText.h"
#include "TAttMarker.h"
#include "TArrayF.h"

class TColor;

#include <string>

class TWebPainting : public TObject {

   protected:
      std::string fClassName; ///< class name of object produced this painting
      std::string fObjectName; ///< object name
      std::string fOper;      ///< list of operations, separated by semicolons
      Int_t fSize{0};         ///<! filled buffer size
      TArrayF fBuf;           ///< array of points for all operations
      TAttLine fLastLine;     ///<! last line attributes
      TAttFill fLastFill;     ///<! last fill attributes
      TAttMarker fLastMarker; ///<! last marker attributes

   public:

      TWebPainting();
      ~TWebPainting() override = default;

      void SetClassName(const std::string &classname) { fClassName = classname; }
      const std::string &GetClassName() const { return fClassName; }

      void SetObjectName(const std::string &objname) { fObjectName = objname; }
      const std::string &GetObjectName() const { return fObjectName; }

      Bool_t IsEmpty() const { return fOper.empty() && (fBuf.GetSize() == 0); }

      void AddOper(const std::string &oper);

      void AddLineAttr(const TAttLine &attr);
      void AddFillAttr(const TAttFill &attr);
      void AddTextAttr(const TAttText &attr);
      void AddMarkerAttr(const TAttMarker &attr);

      Float_t *Reserve(Int_t sz);

      void AddColor(Int_t indx, TColor *col);

      // Set actual filled size
      void FixSize() { fBuf.Set(fSize); }

      static std::string MakeTextOper(const char *str);


   ClassDefOverride(TWebPainting, 2) // store for all paint operation of TVirtualPadPainter
};

#endif
