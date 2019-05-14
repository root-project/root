/// \file RFitPanel.cxx
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

#include <ROOT/RFitPanel.hxx>

#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/TLogger.hxx>

#include "TString.h"
#include "TBackCompFitter.h"
#include "TGraph.h"
#include "TROOT.h"
#include "TH1.h"
#include "TF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TList.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TDirectory.h"
#include "TBufferJSON.h"
#include "TMath.h"
#include "Math/Minimizer.h"
#include "HFitInterface.h"
#include "TColor.h"

#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std::string_literals;

/** \class ROOT::Experimental::RFitPanel
\ingroup webdisplay

web-based FitPanel prototype.
*/

ROOT::Experimental::RFitPanel::RFitPanel(const std::string &title)
{
   model().fTitle = title;

   GetFunctionsFromSystem();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Returns RWebWindow instance, used to display FitPanel

std::shared_ptr<ROOT::Experimental::RWebWindow> ROOT::Experimental::RFitPanel::GetWindow()
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

void ROOT::Experimental::RFitPanel::AssignHistogram(TH1 *hist)
{
   fHist = hist;
   model().SelectHistogram("", fHist);
   SendModel();
}

/// Assign histogram name to use with fit panel - it should be available in gDirectory

void ROOT::Experimental::RFitPanel::AssignHistogram(const std::string &hname)
{
   fHist = nullptr;
   model().SelectHistogram(hname, nullptr);
   SendModel();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// assign canvas to use for drawing results of fitting or showing fitpanel itself

void ROOT::Experimental::RFitPanel::AssignCanvas(std::shared_ptr<RCanvas> &canv)
{
   if (!fCanvas) {
      fCanvas = canv;
   } else {
      R__ERROR_HERE("webgui") << "FitPanel already bound to the canvas - change is not yet supported";
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// assign histogram for fitting

void ROOT::Experimental::RFitPanel::AssignHistogram(std::shared_ptr<RH1D> &hist)
{
   fFitHist = hist;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Show FitPanel

void ROOT::Experimental::RFitPanel::Show(const std::string &where)
{
   GetWindow()->Show(where);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Hide FitPanel

void ROOT::Experimental::RFitPanel::Hide()
{
   if (fWindow)
      fWindow->CloseConnections();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Return reference on model object
/// Model created if was not exists before


ROOT::Experimental::RFitPanelModel &ROOT::Experimental::RFitPanel::model()
{
   if (!fModel)
      fModel = std::make_unique<RFitPanelModel>();

   return *fModel.get();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Send model object to the client

void ROOT::Experimental::RFitPanel::SendModel()
{
   if (fWindow && (fConnId > 0)) {
      TString json = TBufferJSON::ToJSON(&model());
      fWindow->Send(fConnId, "MODEL:"s + json.Data());
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from FitPanel
/// OpenUI5-based FitPanel sends commands or status changes

void ROOT::Experimental::RFitPanel::ProcessData(unsigned connid, const std::string &arg)
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

      if (UpdateModel(arg.substr(6)) >= 0)
         if (DoFit())
            SendModel();

   } else if (arg.compare(0, 11, "SETCONTOUR:") == 0) {

      DrawContour(arg.substr(11));

   } else if (arg.compare(0, 8, "SETSCAN:") == 0) {

      DrawScan(arg.substr(8));

   } else if (arg.compare(0, 8, "GETPARS:") == 0) {

      auto &m = model();

      m.SelectedFunc(arg.substr(8), FindFunction(arg.substr(8)));

      TString json = TBufferJSON::ToJSON(&m.fFuncPars);
      fWindow->Send(fConnId, "PARS:"s + json.Data());

   } else if (arg.compare(0, 8, "SETPARS:") == 0) {

      auto info = TBufferJSON::FromJSON<RFitPanelModel::RFitFuncParsList>(arg.substr(8));

      if (info) {
         TF1 *func = FindFunction(info->name);

         // copy all parameters back to the function
         if (func)
            info->SetParameters(func);
      }

   }
}

/////////////////////////////////////////////////////////////////////////////////////
/// Search for existing functions, ownership still belongs to FitPanel or global lists

TF1 *ROOT::Experimental::RFitPanel::FindFunction(const std::string &funcname)
{
   if (funcname.compare(0,8,"system::")) {
      std::string name = funcname.substr(8);

      for (auto &item : fSystemFuncs)
         if (name.compare(item->GetName()) == 0)
            return item.get();
   }

   return nullptr;
}

//////////////////////////////////////////////////////////
/// Creates new instance to make fitting

std::unique_ptr<TF1> ROOT::Experimental::RFitPanel::GetFitFunction(const std::string &funcname)
{
   std::unique_ptr<TF1> res;

   TF1 *func = FindFunction(funcname);

   if (func) {
      // Now we make a copy.
      res.reset((TF1*) func->IsA()->New());
      func->Copy(*res);
   } else if (funcname.compare(0,6,"dflt::") == 0) {

      std::string formula = funcname.substr(6);

      ROOT::Fit::DataRange drange;
      model().GetRanges(drange);

      double xmin, xmax, ymin, ymax, zmin, zmax;
      drange.GetRange(xmin, xmax, ymin, ymax, zmin, zmax);

      if ( model().fDim == 1 || model().fDim == 0 ) {
         res.reset(new TF1("PrevFitTMP", formula.c_str(), xmin, xmax));
      } else if ( model().fDim == 2 ) {
         res.reset(new TF2("PrevFitTMP", formula.c_str(), xmin, xmax, ymin, ymax));
      } else if ( model().fDim == 3 ) {
         res.reset(new TF3("PrevFitTMP", formula.c_str(), xmin, xmax, ymin, ymax, zmin, zmax));
      }
   }

   return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dummy function, called when "Fit" button pressed in UI

void ROOT::Experimental::RFitPanel::DrawContour(const std::string &model)
{
   // FIXME: do not use static!!!
   static TGraph *graph = nullptr;
   std::string options;
   // TBackCompFitter *fFitter = nullptr;
   auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanelModel>(model);

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

void ROOT::Experimental::RFitPanel::DrawScan(const std::string &model)
{

   auto obj = TBufferJSON::FromJSON<RFitPanelModel>(model);
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

TPad *ROOT::Experimental::RFitPanel::GetDrawPad(TH1 *hist)
{
   if (model().fNoDrawing || model().fNoStoreDraw)
      return nullptr;


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

int ROOT::Experimental::RFitPanel::UpdateModel(const std::string &json)
{
   auto m = TBufferJSON::FromJSON<RFitPanelModel>(json);

   if (!m) {
      R__ERROR_HERE("webgui") << "Fail to parse JSON for RFitPanelModel";
      return -1;
   }

   int res = 0; // nothing changed

   auto selected = m->GetSelectedHistogram(fHist);

   if (model().fSelectedData != m->fSelectedData) {
      res = 1;
      m->UpdateRange(selected);
      m->UpdateFuncList(); // try to get existing function
      m->UpdateAdvanced(nullptr);
      m->fSelectedFunc = ""; // reset func selection
   }

   if (model().fSelectedFunc != m->fSelectedFunc) {
      res = 1;
      m->SelectedFunc(m->fSelectedFunc, FindFunction(m->fSelectedFunc));
   }

   std::swap(fModel, m); // finally replace model

   return res;
}


////////////////////////////////////////////////////////////////////////////////
///Copies f into a new TF1 to be stored in the fitpanel with it's
///own ownership. This is taken from Fit::StoreAndDrawFitFunction in
///HFitImpl.cxx

TF1* ROOT::Experimental::RFitPanel::copyTF1(TF1* f)
{
   double xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;

   // no need to use kNotGlobal bit. TF1::Copy does not add in the list by default
   if ( dynamic_cast<TF3*>(f)) {
      TF3* fnew = (TF3*)f->IsA()->New();
      f->Copy(*fnew);
      f->GetRange(xmin,ymin,zmin,xmax,ymax,zmax);
      fnew->SetRange(xmin,ymin,zmin,xmax,ymax,zmax);
      fnew->SetParent( nullptr );
      fnew->AddToGlobalList(false);
      return fnew;
   } else if ( dynamic_cast<TF2*>(f) != 0 ) {
      TF2* fnew = (TF2*)f->IsA()->New();
      f->Copy(*fnew);
      f->GetRange(xmin,ymin,xmax,ymax);
      fnew->SetRange(xmin,ymin,xmax,ymax);
      fnew->Save(xmin,xmax,ymin,ymax,0,0);
      fnew->SetParent( nullptr );
      fnew->AddToGlobalList(false);
      return fnew;
   }

   TF1* fnew = (TF1*)f->IsA()->New();
   f->Copy(*fnew);
   f->GetRange(xmin,xmax);
   fnew->SetRange(xmin,xmax);
   // This next line is added, as fnew-Save fails with gausND! As
   // the number of dimensions is unknown...
   if ( '\0' != fnew->GetExpFormula()[0] )
      fnew->Save(xmin,xmax,0,0,0,0);
   fnew->SetParent( nullptr );
   fnew->AddToGlobalList(false);
   return fnew;
}

void ROOT::Experimental::RFitPanel::GetFunctionsFromSystem()
{
   // Looks for all the functions registered in the current ROOT
   // session.

   fSystemFuncs.clear();

   // Be carefull not to store functions that will be in the
   // predefined section
   std::vector<std::string> fnames = { "gaus" ,   "gausn", "expo", "landau",
                                       "landaun", "pol0",  "pol1", "pol2",
                                       "pol3",    "pol4",  "pol5", "pol6",
                                       "pol7",    "pol8",  "pol9", "user" };

   // No go through all the objects registered in gROOT
   TIter functionsIter(gROOT->GetListOfFunctions());
   TObject* obj;
   while( (obj = functionsIter()) != nullptr ) {
      // And if they are TF1s
      if ( TF1* func = dynamic_cast<TF1*>(obj) ) {
         bool addFunction = true;
         // And they are not already registered in fSystemFunc
         for ( auto &name : fnames) {
            if ( name.compare(func->GetName()) == 0 ) {
               addFunction = false;
               break;
            }
         }
         // Add them.
         if ( addFunction )
            fSystemFuncs.emplace_back( copyTF1(func) );
      }
   }
}

///////////////////////////////////////////////
/// Perform fitting using current model settings
/// Returns true if any action was done

bool ROOT::Experimental::RFitPanel::DoFit()
{
   auto &m = model();

   TH1 *h1 = m.GetSelectedHistogram(fHist);

   if (!h1) return false;

   if (m.fSelectedFunc.empty())
      m.fSelectedFunc = "dflt::gaus";

   auto f1 = GetFitFunction(m.fSelectedFunc);
   if (!f1) return false;

   ROOT::Fit::DataRange drange;
   ROOT::Math::MinimizerOptions minOption;
   Foption_t fitOpts;

   m.GetRanges(drange);
   m.GetFitOptions(fitOpts);
   m.GetMinimizerOptions(minOption);

   // std::string drawOpts = m.GetDrawOption();

   TString strDrawOpts;

   TVirtualPad *save = gPad;

   auto pad = GetDrawPad(h1);

   ROOT::Fit::FitObject(h1, f1.get(), fitOpts, minOption, strDrawOpts, drange);

   if (m.fSame && f1)
      f1->Draw("same");

   if (pad) {
      pad->Modified();
      pad->Update();
   }

   // TODO: save fitting to be able comeback to previous fits

   m.UpdateAdvanced(f1.get());

   f1->SetName(Form("PrevFit-%d", (int) (fPrevFuncs.size() + 1)));
   fPrevFuncs.emplace(m.fSelectedData, std::move(f1));

   if (save && (gPad != save))
      gPad = save;

   return true; // provide client with latest changes
}
