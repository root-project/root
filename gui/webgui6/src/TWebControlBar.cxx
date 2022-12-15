// Author: Sergey Linev, GSI   15/12/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebControlBar.h"

#include "TROOT.h"


/** \class TWebControlBar
\ingroup webgui6

Web-based implementation for TControlBar class

*/

using namespace std::string_literals;

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TWebControlBar::TWebControlBar(TControlBar *bar, const char *title, Int_t x, Int_t y)
   : TControlBarImp(bar, title, x, y)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Send initial message with buttons configuration

void TWebControlBar::SendInitMsg(unsigned connid)
{
   if (!fWindow)
      return;

   std::string buf = "INIT";

   if (!buf.empty())
      fWindow->Send(connid, buf);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Handle data from web browser
/// Returns kFALSE if message was not processed

Bool_t TWebControlBar::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg.empty())
      return kTRUE;

   printf("Get msg %s from conn %u\n", arg.c_str(), connid);

   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Hide control bar

void TWebControlBar::Hide()
{
   if (fWindow)
      fWindow->CloseConnections();
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Show canvas in browser window

void TWebControlBar::Show()
{
   if (gROOT->IsWebDisplayBatch())
      return;

   if (!fWindow) {
      fWindow = ROOT::Experimental::RWebWindow::Create();

      fWindow->SetConnLimit(1); // configure connections limit

      fWindow->SetDefaultPage("file:rootui5sys/canv/ctrlbar.html");

      fWindow->SetCallBacks(
         // connection
         [this](unsigned connid) {
            SendInitMsg(connid);
         },
         // data
         [this](unsigned connid, const std::string &arg) {
            ProcessData(connid, arg);
         },
         // disconnect
         [this](unsigned) {
         });
   }

   ROOT::Experimental::RWebDisplayArgs args;
   args.SetWidgetKind("TControlBar");

   fWindow->Show(args);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static method to create TWebControlBar instance
/// Used by plugin manager

TControlBarImp *TWebControlBar::NewControlBar(TControlBar *bar, const char *title, Int_t x, Int_t y)
{
   return new TWebControlBar(bar, title, x, y);
}

