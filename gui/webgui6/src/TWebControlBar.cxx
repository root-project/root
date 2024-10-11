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
#include "TBufferJSON.h"
#include "TControlBar.h"
#include "TControlBarButton.h"
#include "TList.h"

#include <vector>
#include <string>

/** \class TWebControlBar
\ingroup webgui6
\ingroup webwidgets

\brief Web-based implementation for TControlBar class

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

   auto lst = fControlBar->GetListOfButtons();

   std::vector<std::string> btns;

   if (fControlBar->GetOrientation() == TControlBar::kHorizontal)
      btns.emplace_back("horizontal");
   else
      btns.emplace_back("vertical");

   btns.emplace_back(fControlBar->GetName());

   TIter iter(lst);
   while (auto btn = iter()) {
      btns.emplace_back(btn->GetName());
      btns.emplace_back(btn->GetTitle());
   }

   if (btns.size() == 0)
      return;

   std::string buf = "BTNS:";
   buf.append(TBufferJSON::ToJSON(&btns).Data());

   fWindow->Send(connid, buf);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Handle data from web browser
/// Returns kFALSE if message was not processed

Bool_t TWebControlBar::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg.empty())
      return kTRUE;

   if (arg.compare(0, 6, "CLICK:") == 0) {
      auto id = std::stoi(arg.substr(6));

      auto lst = fControlBar->GetListOfButtons();

      auto btn = dynamic_cast<TControlBarButton *>(lst->At(id));

      if (btn) {
         printf("Click btn %s act %s\n", btn->GetName(), btn->GetAction());
         btn->Action();
      }

   } else {
      printf("Get msg %s from conn %u\n", arg.c_str(), connid);
   }

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
      fWindow = ROOT::RWebWindow::Create();

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
         });
   }

   ROOT::RWebDisplayArgs args;
   args.SetWidgetKind("TControlBar");

   auto lst = fControlBar->GetListOfButtons();
   int nbtns = 0, maxlen = 0, totallen = 0;
   TIter iter(lst);
   while (auto btn = iter()) {
      nbtns++;
      int len = strlen(btn->GetName());
      totallen += len;
      if (len > maxlen)
         maxlen = len;
   }

   int w = 100, h = 50;

   if (nbtns > 0) {
      if (fControlBar->GetOrientation() == TControlBar::kHorizontal) {
         w = totallen*10 + nbtns*20;
         h = 35;
      } else {
         w = maxlen*10 + 10;
         h = nbtns*30 + 30;
      }
   }
   fWindow->SetGeometry(w, h);

   fWindow->Show(args);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static method to create TWebControlBar instance
/// Used by plugin manager

TControlBarImp *TWebControlBar::NewControlBar(TControlBar *bar, const char *title, Int_t x, Int_t y)
{
   return new TWebControlBar(bar, title, x, y);
}

