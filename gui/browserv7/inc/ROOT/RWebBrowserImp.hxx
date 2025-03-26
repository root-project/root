// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2021-02-11

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebBrowserImp
#define ROOT7_RWebBrowserImp

#include "TBrowserImp.h"

#include "ROOT/RBrowser.hxx"

namespace ROOT {

class RWebBrowserImp : public TBrowserImp {

   std::shared_ptr<RBrowser> fWebBrowser;  ///< actual browser used
   Int_t fX{-1}, fY{-1}, fWidth{0}, fHeight{0}; ///< window coordinates

   void ShowWarning();

public:
   RWebBrowserImp(TBrowser *b = nullptr);
   RWebBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt = "");
   RWebBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt = "");
   virtual ~RWebBrowserImp();

   void      Iconify() final;
   void      Refresh(Bool_t = kFALSE) final;
   void      Show() final;
   void      BrowseObj(TObject *) final;
   Bool_t    IsWeb() const final { return kTRUE; }

   static TBrowserImp *NewBrowser(TBrowser *b = nullptr, const char *title = "ROOT Browser", UInt_t width = 800, UInt_t height = 500, Option_t *opt = "");
   static TBrowserImp *NewBrowser(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt = "");

   ClassDefOverride(RWebBrowserImp,0)  // browser implementation for RBrowser
};

} // namespace ROOT

#endif
