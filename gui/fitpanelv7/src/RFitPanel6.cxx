/// \file RFitPanel6.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \author Iliana Betsou <Iliana.Betsou@cern.ch>
/// \date 2019-04-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFitPanel6.hxx>

#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/TLogger.hxx>

#include "TString.h"
#include "TBackCompFitter.h"
#include "TGraph.h"
#include "TROOT.h"
#include "TF1.h"
#include "TList.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TDirectory.h"
#include "TBufferJSON.h"
#include "TMath.h"
#include "Math/Minimizer.h"
#include "TColor.h"
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std::string_literals;

/** \class ROOT::Experimental::RFitPanel
\ingroup webdisplay

web-based FitPanel prototype.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Returns RWebWindow instance, used to display FitPanel

std::shared_ptr<ROOT::Experimental::RWebWindow> ROOT::Experimental::RFitPanel6::GetWindow()
{
   if (!fWindow) {
      fWindow = RWebWindowsManager::Instance()->CreateWindow();

      fWindow->SetPanelName("rootui5.fitpanel.view.FitPanel");

      fWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); });

      fWindow->SetGeometry(400, 650); // configure predefined geometry
   }

   return fWindow;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Assign histogram to use with fit panel - without ownership

void ROOT::Experimental::RFitPanel6::AssignHistogram(TH1 *hist)
{
   fHist = hist;
   model().SelectHistogram("", fHist);
   SendModel();
}

/// Assign histogram name to use with fit panel - it should be available in gDirectory

void ROOT::Experimental::RFitPanel6::AssignHistogram(const std::string &hname)
{
   fHist = nullptr;
   model().SelectHistogram(hname, nullptr);
   SendModel();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Show FitPanel

void ROOT::Experimental::RFitPanel6::Show(const std::string &where)
{
   GetWindow()->Show(where);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide FitPanel

void ROOT::Experimental::RFitPanel6::Hide()
{
   if (fWindow)
      fWindow->CloseConnections();
}

ROOT::Experimental::RFitPanel6Model &ROOT::Experimental::RFitPanel6::model()
{
   if (!fModel)
      fModel = std::make_unique<RFitPanel6Model>();

   return *fModel.get();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Send model object to the client

void ROOT::Experimental::RFitPanel6::SendModel()
{
   if (fWindow && (fConnId > 0)) {
      TString json = TBufferJSON::ToJSON(&model());
      fWindow->Send(fConnId, "MODEL:"s + json.Data());
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from FitPanel
/// OpenUI5-based FitPanel sends commands or status changes

void ROOT::Experimental::RFitPanel6::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      fConnId = connid;
      fWindow->Send(fConnId, "INITDONE");

      if (!model().IsSelectedHistogram())
         model().SelectHistogram("", fHist);

      SendModel();

   } else if (arg == "CONN_CLOSED") {

      fConnId = 0;

   } else if (arg.compare(0, 7, "UPDATE:") == 0) {

      if (UpdateModel(arg.substr(7)) > 0)
         SendModel();

   } else if (arg.compare(0, 6, "DOFIT:") == 0) {

      DoFit(arg.substr(6));

   } else if (arg.compare(0, 11, "SETCONTOUR:") == 0) {

      DrawContour(arg.substr(11));

   } else if (arg.compare(0, 8, "SETSCAN:") == 0) {

      DrawScan(arg.substr(8));

   } else if (arg.compare(0, 8, "GETPARS:") == 0) {

      auto &m = model();

      m.SelectFunc(arg.substr(8), m.GetSelectedHistogram(fHist));

      TString json = TBufferJSON::ToJSON(&m.fFuncPars);
      fWindow->Send(fConnId, "PARS:"s + json.Data());

   } else if (arg.compare(0, 8, "SETPARS:") == 0) {

      auto info = TBufferJSON::FromJSON<RFitFuncParsList>(arg.substr(8));

      if (info) {
         TF1 *func = model().FindFunction(info->name, fHist);

         // copy all parameters back to the function
         if (func)
            info->SetParameters(func);
      }

   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dummy function, called when "Fit" button pressed in UI
void ROOT::Experimental::RFitPanel6::DrawContour(const std::string &model)
{
   // FIXME: do not use static!!!
   static TGraph *graph = nullptr;
   std::string options;
   // TBackCompFitter *fFitter = nullptr;
   auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanel6Model>(model);

   if (!obj->fContourImpose) {
      if (graph) {
         delete graph;
         options = "ALF";
         graph = nullptr;
      }
   } else {
      options = "LF";
   }

   if (!graph)
      graph = new TGraph(static_cast<int>(obj->fContourPoints));

   auto colorid = TColor::GetColor(std::stoi(obj->fColorContour[0]), std::stoi(obj->fColorContour[1]),
                                   std::stoi(obj->fColorContour[2]));
   graph->SetLineColor(colorid);

   if (obj->fContourPar1 == obj->fContourPar2) {
      Error("DrawContour", "Parameters cannot be the same");
      return;
   }

   // fFitter->Contour(obj->fContourPar1, obj->fContourPar2, graph, obj->fConfLevel);
   // graph->GetXaxis()->SetTitle( fFitter->GetParName(obj->fContourPar1) );
   // graph->GetYaxis()->SetTitle( fFitter->GetParName(obj->fContourPar2) );
   // graph->Draw( options.c_str() );
   gPad->Update();

   // printf("Points %d Contour1 %d Contour2 %d ConfLevel %f\n", obj->fContourPoints, obj->fContourPar1,
   // obj->fContourPar2, obj->fConfLevel);
}

void ROOT::Experimental::RFitPanel6::DrawScan(const std::string &model)
{

   auto obj = TBufferJSON::FromJSON<RFitPanel6Model>(model);
   static TGraph *graph = nullptr;
   // TBackCompFitter *fFitter = nullptr;

   // FIXME: do not use static!!!
   if (graph) {
      delete graph;
   }
   graph = new TGraph(static_cast<int>(obj->fScanPoints));
   // fFitter->Scan(obj->fScanPar, graph, obj->fScanMin, obj->fScanMax);

   graph->SetLineColor(kBlue);
   graph->SetLineWidth(2);
   // graph->GetXaxis()->SetTitle(fFitter->GetParName(obj->fScanPar)); ///???????????
   graph->GetYaxis()->SetTitle("FCN");
   graph->Draw("APL");
   gPad->Update();

   // printf("SCAN Points %d, Par %d, Min %d, Max %d\n", obj->fScanPoints, obj->fScanPar, obj->fScanMin, obj->fScanMax);
}

/// Returns pad where histogram should be drawn
/// Ensure that histogram is on the first place

TPad *ROOT::Experimental::RFitPanel6::GetDrawPad(TH1 *hist)
{
   if (fCanvName.empty()) {
      gPad = nullptr;
      return nullptr;
   }

   TCanvas *canv = (TCanvas *) gROOT->GetListOfCanvases()->FindObject(fCanvName.c_str());

   if (!canv) {
      canv = gROOT->MakeDefCanvas();
      canv->SetName(fCanvName.c_str());
      canv->SetTitle("Fit panel drawings");
   }

   canv->cd();

   // TODO: provide proper draw options
   if (hist && !canv->FindObject(hist)) {
      canv->Clear();
      hist->Draw();
   }

   return canv;
}

////////////////////////////////////////////
/// Update fit model
/// returns -1 if JSON fails
/// return 0 if nothing large changed
/// return 1 if important selection are changed and client need to be updated

int ROOT::Experimental::RFitPanel6::UpdateModel(const std::string &json)
{
   auto m = TBufferJSON::FromJSON<RFitPanel6Model>(json);

   if (!m) {
      R__ERROR_HERE("webgui") << "Fail to parse JSON for RFitPanel6Model";
      return -1;
   }

   int res = 0; // nothing changed

   auto selected = m->GetSelectedHistogram(fHist);

   TF1 *hfunc = nullptr;

   if (model().fSelectedData != m->fSelectedData) {
      res = 1;
      m->UpdateRange(selected);
      hfunc = m->UpdateFuncList(selected, true); // try to get existing function
      m->UpdateAdvanced(hfunc);
      if (!hfunc) m->fSelectedFunc = ""; // reset func selection
   }

   if ((model().fSelectedFunc != m->fSelectedFunc) && !hfunc) {
      res = 1;
      m->SelectFunc(m->fSelectedFunc, selected);
   }

   std::swap(fModel, m); // finally replace model

   return res;
}


void ROOT::Experimental::RFitPanel6::DoFit(const std::string &json)
{
   if (UpdateModel(json) < 0) return;

   auto &m = model();

   ROOT::Math::MinimizerOptions minOption;

   // Fitting Options
   if (gDebug > 0)
      ::Info("RFitPanel6::DoFit", "range %f %f select %s function %s", m.fRangeX[0], m.fRangeX[1],
             m.fSelectedData.c_str(), m.fSelectedFunc.c_str());

   if (m.fSelectedFunc.empty())
      m.fSelectedFunc = "gaus";

   if (!m.fMinLibrary.empty())
      minOption.SetMinimizerAlgorithm(m.fMinLibrary.c_str());

   if (m.fErrorDef == 0)
      minOption.SetErrorDef(1.00);
   else
      minOption.SetErrorDef(m.fErrorDef);

   if (m.fMaxTol == 0)
      minOption.SetTolerance(0.01);
   else
      minOption.SetTolerance(m.fMaxTol);

   minOption.SetMaxIterations(m.fMaxInter);

   std::string opt = m.GetFitOption();

   TH1 *h1 = m.GetSelectedHistogram(fHist);

   auto pad = GetDrawPad(h1);

   // Assign the options to Fitting function
   if (h1 && !m.fSelectedFunc.empty() && (m.fSelectedFunc != "none")) {
      h1->Fit(m.fSelectedFunc.c_str(), opt.c_str(), "*", m.fRangeX[0], m.fRangeX[1]);

      if (pad) pad->Update();

      auto *fres = m.UpdateFuncList(h1, true);

      m.UpdateAdvanced(fres);
   }

   SendModel(); // provide client with latest changes
}
