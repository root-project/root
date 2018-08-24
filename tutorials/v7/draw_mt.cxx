/// \file
/// \ingroup tutorial_v7
///
/// This macro demonstrate usage of ROOT7 graphics from many threads
/// Three different canvases in three different threads are started and regularly updated.
/// Extra thread created in background and used to run http protocol, in/out websocket communications and process http requests
/// Main application thread (CLING interactive session) remains fully functional
///
/// \macro_code
///
/// \date 2018-08-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

R__LOAD_LIBRARY(libROOTGpadv7);
// R__LOAD_LIBRARY(libROOTGraphicsPrimitives);
// R__LOAD_LIBRARY(libROOTHistDraw);

#include "ROOT/RHist.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/TWebWindowsManager.hxx"

#include "TRandom3.h"
#include "TEnv.h"

#include <thread>
// #include <chrono>

using namespace ROOT::Experimental;

void draw_canvas(const std::string &title, RColor col)
{
   // Create histograms
   RAxisConfig xaxis(100, -10., 10.);
   auto pHist = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   TRandom3 random;
   Float_t px, py;

   for(int n=0;n<10000;++n) {
      random.Rannor(px,py);
      pHist->Fill(px-2);
      pHist2->Fill(py+2);
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create(title);
   canvas->Draw(pHist)->SetLineColor(col);
   canvas->Draw(pHist2)->SetLineColor(RColor::kBlue);

   canvas->Show();

   for (int loop=0;loop<50;++loop) {

      for(int n=0;n<10000;++n) {
         random.Rannor(px,py);
         pHist->Fill(px-2);
         pHist2->Fill(py+2);
      }

      canvas->Modified();
      canvas->Update();

      // std::this_thread::sleep_for(std::chrono::milliseconds(500));
      canvas->Run(0.5); // let run canvas code for next 0.5 seconds
   }

   //   canvas->SaveAs("th1.png");

   printf("%s completed\n", title.c_str());

   // remove from global list, will be destroyed with thread exit
   canvas->Remove();
}

void draw_mt()
{
   gEnv->SetValue("WebGui.HttpThrd","yes");
   gEnv->SetValue("WebGui.SenderThrds","yes");

   std::thread thrd1(draw_canvas, "First canvas", RColor::kRed);
   std::thread thrd2(draw_canvas, "Second canvas", RColor::kBlue);
   std::thread thrd3(draw_canvas, "Third canvas", RColor::kGreen);

   thrd1.detach();
   thrd2.detach();
   thrd3.detach();
}
