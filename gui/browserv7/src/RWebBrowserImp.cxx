// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2021-02-11
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RWebBrowserImp.hxx>

#include "TROOT.h"
#include "TSeqCollection.h" // needed in gROOT->GetListOfFiles()->FindObject

#include <iostream>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RWebBrowserImp::RWebBrowserImp(TBrowser *b) : TBrowserImp(b)
{
   ShowWarning();

   fWebBrowser = std::make_shared<RBrowser>();
   fWebBrowser->AddTCanvas();
}

////////////////////////////////////////////////////////////////////////////////////////
/// Constructor with width and height parameters

RWebBrowserImp::RWebBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt) : TBrowserImp(b,title, width, height, opt)
{
   ShowWarning();

   fWidth = width;
   fHeight = height;
   fWebBrowser = std::make_shared<RBrowser>();
   fWebBrowser->AddTCanvas();
}

////////////////////////////////////////////////////////////////////////////////////////
/// Constructor with x,y, width and height parameters

RWebBrowserImp::RWebBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt) : TBrowserImp(b,title, x, y, width, height, opt)
{
   ShowWarning();

   fX = x;
   fY = y;
   fWidth = width;
   fHeight = height;
   fWebBrowser = std::make_shared<RBrowser>();
   fWebBrowser->AddTCanvas();
}

////////////////////////////////////////////////////////////////////////////////////////
/// Constructor with width and height parameters

RWebBrowserImp::~RWebBrowserImp()
{
}

////////////////////////////////////////////////////////////////////////////////////////
/// Show warning that RBrowser will be shown

void RWebBrowserImp::ShowWarning()
{
   static bool show_warn = true;
   if (!show_warn) return;
   show_warn = false;

   std::cout << "\n"
                "ROOT comes with a web-based browser, which is now being started. \n"
                "Revert to TBrowser by setting \"Browser.Name: TRootBrowser\" in rootrc file or\n"
                "by starting \"root --web=off\"\n"
                "Web-based TBrowser can be used in batch mode when starting with \"root -b --web=server:8877\"\n"
                "Find more info on https://root.cern/for_developers/root7/#rbrowser\n";
}

////////////////////////////////////////////////////////////////////////////////////////
/// Iconify browser

void RWebBrowserImp::Iconify()
{
}

////////////////////////////////////////////////////////////////////////////////////////
/// Refresh browser

void RWebBrowserImp::Refresh(Bool_t)
{
}

////////////////////////////////////////////////////////////////////////////////////////
/// Show browser

void RWebBrowserImp::Show()
{
   fWebBrowser->Show();
}


////////////////////////////////////////////////////////////////////////////////////////
/// Browse specified object

void RWebBrowserImp::BrowseObj(TObject *obj)
{
   if (obj == gROOT) return;

   if (gROOT->GetListOfFiles()->FindObject(obj))
      fWebBrowser->SetWorkingPath("ROOT Files");
}

////////////////////////////////////////////////////////////////////////////////////////
/// Factory method to create RWebBrowserImp via plugin

TBrowserImp *RWebBrowserImp::NewBrowser(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt)
{
   return new RWebBrowserImp(b, title, width, height, opt);
}

////////////////////////////////////////////////////////////////////////////////////////
/// Factory method to create RWebBrowserImp via plugin

TBrowserImp *RWebBrowserImp::NewBrowser(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt)
{
   return new RWebBrowserImp(b, title, x, y, width, height, opt);
}
