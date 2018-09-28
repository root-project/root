/// \file ROOT/TFitPanel.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-10-24
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFitPanel.hxx"

#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/TLogger.hxx>
#include "ROOT/TDirectory.hxx"

#include "TString.h"
#include "TROOT.h"
#include "TBufferJSON.h"

/** \class ROOT::Experimental::TFitPanel
\ingroup webdisplay

web-based FitPanel prototype.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Returns RWebWindow instance, used to display FitPanel

std::shared_ptr<ROOT::Experimental::RWebWindow> ROOT::Experimental::TFitPanel::GetWindow()
{
   if (!fWindow) {
      fWindow = RWebWindowsManager::Instance()->CreateWindow();

      fWindow->SetPanelName("FitPanel");

      fWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); });

      fWindow->SetGeometry(300, 500); // configure predefined geometry
   }

   return fWindow;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Show FitPanel

void ROOT::Experimental::TFitPanel::Show(const std::string &where)
{
   GetWindow()->Show(where);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide FitPanel

void ROOT::Experimental::TFitPanel::Hide()
{
   if (!fWindow)
      return;

   fWindow->CloseConnections();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from FitPanel
/// OpenUI5-based FitPanel sends commands or status changes

void ROOT::Experimental::TFitPanel::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      fConnId = connid;
      printf("FitPanel connection established %u\n", fConnId);
      fWindow->Send(fConnId, "INITDONE");

      TFitPanelModel model;
      model.fDataNames.push_back(ComboBoxItem("1", "RootData1"));
      model.fDataNames.push_back(ComboBoxItem("2", "RootData2"));
      model.fDataNames.push_back(ComboBoxItem("3", "RootData3"));
      model.fSelectDataId = "1";

      model.fModelNames.push_back(ComboBoxItem("1", "RootModel1"));
      model.fModelNames.push_back(ComboBoxItem("2", "RootModel2"));
      model.fModelNames.push_back(ComboBoxItem("3", "RootModel3"));
      model.fSelectModelId = "3";

      TString json = TBufferJSON::ToJSON(&model);

      fWindow->Send(fConnId, std::string("MODEL:") + json.Data());

      return;
   }

   if (arg == "CONN_CLOSED") {
      printf("FitPanel connection closed\n");
      fConnId = 0;
      return;
   }

   if (arg.find("DOFIT:") == 0) {
      TString exec;
      exec.Form("((ROOT::Experimental::TFitPanel *) %p)->DoFit(%s);", this, arg.c_str() + 6);
      printf("Execute %s\n", exec.Data());
      gROOT->ProcessLine(exec.Data());
      return;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Let use canvas to display fit results

void ROOT::Experimental::TFitPanel::UseCanvas(std::shared_ptr<RCanvas> &canv)
{
   if (!fCanvas) {
      fCanvas = canv;
   } else {
      R__ERROR_HERE("ShowIn") << "FitPanel already bound to the canvas - change is not yet supported";
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dummy function, called when "Fit" button pressed in UI

void ROOT::Experimental::TFitPanel::DoFit(const std::string &dname, const std::string &mname)
{
   printf("DoFit %s %s\n", dname.c_str(), mname.c_str());

   bool first_time = false;

   if (!fCanvas) {
      fCanvas = Experimental::RCanvas::Create("FitPanel Canvas");
      first_time = true;
   }

   if (!fFitHist) {

      // Create the histogram.
      auto xaxis = std::make_shared<ROOT::Experimental::RAxisConfig>(10, 0., 10.);

      fFitHist = std::make_shared<ROOT::Experimental::RH1D>(*xaxis.get());

      // Fill a few points.
      fFitHist->Fill(5);
      fFitHist->Fill(6);
      fFitHist->Fill(6);
      fFitHist->Fill(7);

      fCanvas->Draw(fFitHist)->SetLineColor(Experimental::RColor::kBlue);

      // workaround to keep histogram in the lists
      ROOT::Experimental::TDirectory::Heap().Add("fitaxis", xaxis);

      if (first_time) {
         fCanvas->Show();
         // fCanvas->Update();
      } else {
         fCanvas->Modified();
         // fCanvas->Update();
      }
   }
}
