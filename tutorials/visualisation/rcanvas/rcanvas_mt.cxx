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
/// \author Sergey Linev

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RWebWindowsManager.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/TObjectDrawable.hxx"

#include "TRandom3.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TH1.h"
#include "TColor.h"

#include <thread>
#include <iostream>

using namespace ROOT::Experimental;

void draw_canvas(const std::string &title, Int_t col1)
{
   // Create histograms
   auto hist1 = new TH1D("hist1", "hist1", 100, -10, 10);
   auto hist2 = new TH1D("hist2", "hist2", 100, -10, 10);

   hist1->SetLineColor(col1);
   hist2->SetLineColor(kBlue);

   TRandom3 random;
   Float_t px, py;

   for (int n = 0; n < 10000; ++n) {
      random.Rannor(px, py);
      hist1->Fill(px - 2);
      hist2->Fill(py + 2);
   }

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create(title);

   canvas->Draw<TObjectDrawable>(hist1, "");
   canvas->Draw<TObjectDrawable>(hist2, "same");

   int maxloop = 100;

   canvas->Show();

   std::cout << title << " started" << std::endl;

   for (int loop = 0; loop < maxloop; ++loop) {

      for (int n = 0; n < 10000; ++n) {
         random.Rannor(px, py);
         hist1->Fill(px - 2);
         hist2->Fill(py + 2);
      }

      canvas->Modified();

      canvas->Update();
      canvas->Run(0.2); // let run canvas code for next 0.5 seconds

      // if (loop == 0)
      //    canvas->SaveAs(title + "_first.png");
      // if (loop == maxloop - 1)
      //    canvas->SaveAs(title + "_last.png");
   }

   std::cout << title << " completed" << std::endl;

   // remove from global list, will be destroyed with thread exit
   canvas->Remove();
}

void rcanvas_mt(bool block_main_thread = true)
{
   TH1::AddDirectory(false);

   if (block_main_thread) {
      // let use special http thread to process requests, do not need main thread
      // required while gSystem->ProcessEvents() will be blocked
      gEnv->SetValue("WebGui.HttpThrd", "yes");

      // let create special threads for data sending, optional
      gEnv->SetValue("WebGui.SenderThrds", "yes");
   }

   ROOT::EnableThreadSafety();

   // create instance in main thread, used to assign thread id as well
   ROOT::RWebWindowsManager::Instance();

   std::thread thrd1(draw_canvas, "First canvas", kRed);
   std::thread thrd2(draw_canvas, "Second canvas", kBlue);
   std::thread thrd3(draw_canvas, "Third canvas", kGreen);

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
