/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro demonstrate usage of ROOT7 graphics from many threads
/// Three different canvases in three different threads are started and regularly updated.
/// Extra thread created in background and used to run http protocol, in/out websocket communications and process http
/// requests
/// Main application thread (CLING interactive session) remains fully functional
///
/// \macro_code
///
/// \date 2018-08-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RWebWindowsManager.hxx"
#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"

#include "TRandom3.h"
#include "TEnv.h"
#include "TROOT.h"

#include <thread>
#include <iostream>

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void draw_canvas(const std::string &title, RColor col)
{
   // Create histograms
   RAxisConfig xaxis(100, -10., 10.);
   auto pHist = std::make_shared<RH1D>(xaxis);
   auto pHist2 = std::make_shared<RH1D>(xaxis);

   TRandom3 random;
   Float_t px, py;

   for (int n = 0; n < 10000; ++n) {
      random.Rannor(px, py);
      pHist->Fill(px - 2);
      pHist2->Fill(py + 2);
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create(title);

   canvas->Draw<RFrameTitle>(title);
   canvas->Draw(pHist)->line.color = col;
   canvas->Draw(pHist2)->line.color = RColor::kBlue;

   int maxloop = 100;

   canvas->Show();

   std::cout << title << " started" <<std::endl;

   for (int loop = 0; loop < maxloop; ++loop) {

      for (int n = 0; n < 10000; ++n) {
         random.Rannor(px, py);
         pHist->Fill(px - 2);
         pHist2->Fill(py + 2);
      }

      canvas->Modified();

      canvas->Update();
      canvas->Run(0.2); // let run canvas code for next 0.5 seconds

      // if (loop == 0)
      //    canvas->SaveAs(title + "_first.png");
      // if (loop == maxloop - 1)
      //    canvas->SaveAs(title + "_last.png");
   }

   std::cout << title << " completed" <<std::endl;

   // remove from global list, will be destroyed with thread exit
   canvas->Remove();
}

void rcanvas_mt(bool block_main_thread = true)
{
   if (block_main_thread) {
      // let use special http thread to process requests, do not need main thread
      // required while gSystem->ProcessEvents() will be blocked
      gEnv->SetValue("WebGui.HttpThrd", "yes");

      // let create special threads for data sending, optional
      gEnv->SetValue("WebGui.SenderThrds", "yes");
   }

   ROOT::EnableThreadSafety();

   // create instance in main thread, used to assign thread id as well
   RWebWindowsManager::Instance();

   std::thread thrd1(draw_canvas, "First canvas", RColor::kRed);
   std::thread thrd2(draw_canvas, "Second canvas", RColor::kBlue);
   std::thread thrd3(draw_canvas, "Third canvas", RColor::kGreen);

   if (block_main_thread) {
      // wait until threads execution finished
      thrd1.join();
      thrd2.join();
      thrd3.join();
   } else {
      // detach threads and return to CLING
      thrd1.detach();
      thrd2.detach();
      thrd3.detach();
   }
}
