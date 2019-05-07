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

#include "TString.h"
#include "TBackCompFitter.h"
#include "TGraph.h"
#include "TROOT.h"
#include "TF1.h"
#include "TList.h"
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

      fWindow->SetGeometry(380, 750); // configure predefined geometry
   }

   return fWindow;
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
   if (!fWindow)
      return;

   fWindow->CloseConnections();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from FitPanel
/// OpenUI5-based FitPanel sends commands or status changes

void ROOT::Experimental::RFitPanel6::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      fConnId = connid;
      fWindow->Send(fConnId, "INITDONE");
      if (!fModel) {
         fModel = std::make_unique<RFitPanel6Model>();
         fModel->Initialize(fHist);
      }

      // Communication with the JSONModel in JS
      TString json = TBufferJSON::ToJSON(fModel.get());
      fWindow->Send(fConnId, "MODEL:"s + json.Data());

   } else if (arg == "CONN_CLOSED") {
      printf("FitPanel connection closed\n");
      fConnId = 0;
   } else if (arg.compare(0, 6, "DOFIT:") == 0) {

      DoFit(arg.substr(6));
   } else if (arg.compare(0, 11, "SETCONTOUR:") == 0) {

      DrawContour(arg.substr(11));
   } else if (arg.compare(0, 8, "SETSCAN:") == 0) {

      DrawScan(arg.substr(8));
   } else if (arg.compare(0, 8, "GETPARS:") == 0) {

      RFitFunc info;
      // ROOT::Experimental::RFitPanel6Model model;

      info.name = arg.substr(8);
      TF1 *func = dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(info.name.c_str()));

      printf("Found func %s %p\n", info.name.c_str(), func);

      if (func) {
         for (int n = 0; n < func->GetNpar(); ++n) {
            info.pars.emplace_back(n, func->GetParName(n));
            auto &par = info.pars.back();

            par.value = func->GetParameter(n);
            par.error = func->GetParError(n);
            func->GetParLimits(n, par.min, par.max);
            if ((par.min >= par.max) && ((par.min != 0) || (par.max != 0)))
               par.fixed = true;
         }
      } else {
         info.name = "<not exists>";
      }
      TString json = TBufferJSON::ToJSON(&info);

      fWindow->Send(fConnId, "PARS:"s + json.Data());

   } else if (arg.compare(0, 8, "SETPARS:") == 0) {

      auto info = TBufferJSON::FromJSON<RFitFunc>(arg.substr(8));

      if (info) {
         TF1 *func = dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(info->name.c_str()));

         if (func) {
            printf("Found func1 %s %p %d %d\n", info->name.c_str(), func, func->GetNpar(), (int)info->pars.size());
            // copy all parameters back to the function
            for (int n = 0; n < func->GetNpar(); ++n) {
               func->SetParameter(n, info->pars[n].value);
               func->SetParError(n, info->pars[n].error);
               func->SetParLimits(n, info->pars[n].min, info->pars[n].max);
               if (info->pars[n].fixed)
                  func->FixParameter(n, info->pars[n].value);
            }
         }
      }

   } else if (arg.compare(0, 12, "GETADVANCED:") == 0) {

      RFitFunc info;
      RFitPanel6Model modelAdv;

      info.name = arg.substr(12);
      TF1 *func = dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(info.name.c_str()));

      // printf("Found func1 %s %p\n", info.name.c_str(), func);

      if (func) {
         for (int n = 0; n < func->GetNpar(); ++n) {
            modelAdv.fContour1.emplace_back(std::to_string(n), func->GetParName(n));
            modelAdv.fContourPar1Id = "0";
            modelAdv.fContour2.emplace_back(std::to_string(n), func->GetParName(n));
            modelAdv.fContourPar2Id = "0";
            modelAdv.fScan.emplace_back(std::to_string(n), func->GetParName(n));
            modelAdv.fScanId = "0";
         }
      } else {
         info.name = "<not exists>";
      }
      TString jsonModel = TBufferJSON::ToJSON(&modelAdv);

      fWindow->Send(fConnId, "ADVANCED:"s + jsonModel.Data());
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

void ROOT::Experimental::RFitPanel6::DoFit(const std::string &model)
{
   // printf("DoFit %s\n", model.c_str());
   auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanel6Model>(model);

   ROOT::Math::MinimizerOptions minOption;

   // Fitting Options
   if (obj) {

      if (gDebug > 0)
         ::Info("RFitPanel6::DoFit", "range %f %f select %s function %s", obj->fUpdateRange[0], obj->fUpdateRange[1],
                obj->fSelectDataId.c_str(), obj->fSelectedFunc.c_str());

      if (obj->fSelectedFunc.empty())
         obj->fSelectedFunc = "gaus";

      if (!obj->fMinLibrary.empty())
         minOption.SetMinimizerAlgorithm(obj->fMinLibrary.c_str());

      if (obj->fErrorDef == 0)
         minOption.SetErrorDef(1.00);
      else
         minOption.SetErrorDef(obj->fErrorDef);

      if (obj->fMaxTol == 0)
         minOption.SetTolerance(0.01);
      else
         minOption.SetTolerance(obj->fMaxTol);

      minOption.SetMaxIterations(obj->fMaxInter);

      std::string opt = obj->GetFitOption();

      TH1 *h1 = obj->FindHistogram(obj->fSelectDataId, fHist);

      // Assign the options to Fitting function
      if (h1 && !obj->fSelectedFunc.empty() && (obj->fSelectedFunc!="none")) {
         h1->Fit(obj->fSelectedFunc.c_str(), opt.c_str(), "*", obj->fUpdateRange[0], obj->fUpdateRange[1]);
         gPad->Update();
      }
   }
}
