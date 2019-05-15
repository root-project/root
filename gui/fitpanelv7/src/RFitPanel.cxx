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
#include "TGraph2D.h"
#include "TMultiGraph.h"
#include "THStack.h"
#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
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

enum EFitObjectType {
   kObjectNone,
   kObjectHisto,
   kObjectGraph,
   kObjectGraph2D,
   kObjectHStack,
//   kObjectTree,
   kObjectMultiGraph,
   kObjectNotSupported
};




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
/// Update list of avaliable data

void ROOT::Experimental::RFitPanel::UpdateDataSet()
{
   auto &m = model();

   m.fDataSet.clear();

   for (auto &obj : fObjects)
      m.fDataSet.emplace_back("Panel", "panel::"s + obj->GetName(), Form("%s::%s", obj->ClassName(), obj->GetName()));

   if (gDirectory) {
      TIter iter(gDirectory->GetList());
      TObject *obj = nullptr;
      while ((obj = iter()) != nullptr) {
         if (obj->InheritsFrom(TH1::Class()) ||
             obj->InheritsFrom(TGraph::Class()) ||
             obj->InheritsFrom(TGraph2D::Class()) ||
             obj->InheritsFrom(THStack::Class()) ||
             obj->InheritsFrom(TMultiGraph::Class())) {
            m.fDataSet.emplace_back("gDirectory", "gdir::"s + obj->GetName(), Form("%s::%s", obj->ClassName(), obj->GetName()));
         }
      }
   }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Select object for fitting

void ROOT::Experimental::RFitPanel::SelectObject(const std::string &objid)
{
   UpdateDataSet();

   auto &m = model();

   std::string id = objid;
   if (id.compare("$$$") == 0) {
      if (m.fDataSet.size() > 0)
         id = m.fDataSet[0].id;
      else
         id.clear();
   }

   int kind{0};
   TObject *obj = GetSelectedObject(id, &kind);

   TH1 *hist = nullptr;
   switch (kind) {
      case kObjectHisto:
         hist = (TH1*)obj;
         break;

      case kObjectGraph:
         hist = ((TGraph*)obj)->GetHistogram();
         break;

      case kObjectMultiGraph:
         hist = ((TMultiGraph*)obj)->GetHistogram();
         break;

      case kObjectGraph2D:
         hist = ((TGraph2D*)obj)->GetHistogram("empty");
         break;

      case kObjectHStack:
         hist = (TH1 *)((THStack *)obj)->GetHists()->First();
         break;

      default:
         break;
   }

   if (!obj)
      m.fSelectedData = "";
   else
      m.fSelectedData = id;

   m.fInitialized = true;

   // update list of data

   m.UpdateRange(hist);

   UpdateFunctionsList();

   if (!m.HasFunction(m.fSelectedFunc)) {
      if (m.fFuncList.size() > 0)
         m.fSelectedFunc = m.fFuncList[0].id;
      else
         m.fSelectedFunc.clear();
   }

   m.UpdateAdvanced(nullptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Assign histogram to use with fit panel - without ownership

TObject *ROOT::Experimental::RFitPanel::GetSelectedObject(const std::string &objid, int *kind)
{
   TObject *res = nullptr;

   if (objid.compare(0,6,"gdir::") == 0) {
      std::string name = objid.substr(6);
      if (gDirectory)
         res = gDirectory->GetList()->FindObject(name.c_str());
   } else if (objid.compare(0,7,"panel::") == 0) {
      std::string name = objid.substr(7);
      for (auto &item : fObjects)
         if (name.compare(item->GetName()) == 0) {
            res = item;
            break;
         }
   }

   if (res && kind) {
      if (res->InheritsFrom(TH1::Class()))
         *kind = kObjectHisto;
      else if (res->InheritsFrom(TGraph::Class()))
         *kind = kObjectGraph;
      else if (res->InheritsFrom(TGraph2D::Class()))
         *kind = kObjectGraph2D;
      else if (res->InheritsFrom(THStack::Class()))
         *kind = kObjectGraph2D;
      else if (res->InheritsFrom(TMultiGraph::Class()))
         *kind = kObjectGraph2D;
      else
         *kind = kObjectNotSupported;
   } else if (kind) {
      *kind = kObjectNone;
   }

   return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Update list of available functions

void ROOT::Experimental::RFitPanel::UpdateFunctionsList()
{
   auto &m = model();

   m.fFuncList.clear();

   if (m.fDim == 1) {
      m.fFuncList = { {"gaus"}, {"gausn"}, {"expo"}, {"landau"}, {"landaun"},
                    {"pol0"},{"pol1"},{"pol2"},{"pol3"},{"pol4"},{"pol5"},{"pol6"},{"pol7"},{"pol8"},{"pol9"},
                    {"cheb0"}, {"cheb1"}, {"cheb2"}, {"cheb3"}, {"cheb4"}, {"cheb5"}, {"cheb6"}, {"cheb7"}, {"cheb8"}, {"cheb9"} };
   } else if (m.fDim == 2) {
      m.fFuncList = { {"xygaus"}, {"bigaus"}, {"xyexpo"}, {"xylandau"}, {"xylandaun"} };
   }

   for (auto &func : fSystemFuncs) {
      m.fFuncList.emplace_back("System", "system::"s + func->GetName(), func->GetName());
   }

   for (auto &pair : fPrevFuncs) {
      if (pair.first == m.fSelectedData)
         m.fFuncList.emplace_back("Previous", "previous::"s + pair.second->GetName(), pair.second->GetName());
   }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Select function

void ROOT::Experimental::RFitPanel::SelectFunction(const std::string &funcid)
{
   model().SelectedFunc(funcid, FindFunction(funcid));
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Assign histogram to use with fit panel - without ownership

void ROOT::Experimental::RFitPanel::AssignHistogram(TH1 *hist)
{
   fObjects.emplace_back(hist);
   SelectObject("panel::"s + hist->GetName());
   SendModel();
}

/// Assign histogram name to use with fit panel - it should be available in gDirectory

void ROOT::Experimental::RFitPanel::AssignHistogram(const std::string &hname)
{
   SelectObject("gdir::" + hname);
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
   if (!fModel) {
      fModel = std::make_unique<RFitPanelModel>();
      fModel->Initialize();
   }

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
      if (!model().fInitialized)
         SelectObject("$$$");

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

      auto info = TBufferJSON::FromJSON<RFitPanelModel::RFuncParsList>(arg.substr(8));

      if (info) {
         TF1 *func = FindFunction(info->id);

         // copy all parameters back to the function
         if (func)
            info->SetParameters(func);
      }

   }
}

/////////////////////////////////////////////////////////////////////////////////////
/// Search for existing functions, ownership still belongs to FitPanel or global lists

TF1 *ROOT::Experimental::RFitPanel::FindFunction(const std::string &id)
{
   if (id.compare(0,8,"system::") == 0) {
      std::string name = id.substr(8);

      for (auto &item : fSystemFuncs)
         if (name.compare(item->GetName()) == 0)
            return item.get();
   }

   if (id.compare(0,10,"previous::") == 0) {
      std::string name = id.substr(10);

      for (auto &pair : fPrevFuncs)
         if (name.compare(pair.second->GetName()) == 0)
            return pair.second.get();
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
         res.reset(new TF1(formula.c_str(), formula.c_str(), xmin, xmax));
      } else if ( model().fDim == 2 ) {
         res.reset(new TF2(formula.c_str(), formula.c_str(), xmin, xmax, ymin, ymax));
      } else if ( model().fDim == 3 ) {
         res.reset(new TF3(formula.c_str(), formula.c_str(), xmin, xmax, ymin, ymax, zmin, zmax));
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

TPad *ROOT::Experimental::RFitPanel::GetDrawPad(TObject *obj)
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
   if (obj && !canv->FindObject(obj)) {
      canv->Clear();
      obj->Draw();
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

   m->fInitialized = true;

   int res = 0; // nothing changed

   if (model().fSelectedData != m->fSelectedData) {
      res |= 1;
   }

   if (model().fSelectedFunc != m->fSelectedFunc) {
      res |= 2;
   }

   std::swap(fModel, m); // finally replace model

   if (res & 1)
      SelectObject(model().fSelectedData);

   if (res != 0)
      SelectFunction(model().fSelectedFunc);

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

   int kind{0};
   TObject *obj = GetSelectedObject(m.fSelectedData, &kind);

   if (!obj) return false;

   if (m.fSelectedFunc.empty())
      m.fSelectedFunc = (m.fDim == 2) ? "dflt::xygaus" : "dflt::gaus";

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

   auto pad = GetDrawPad(obj);

   switch (kind) {
      case kObjectHisto: {

         TH1 *hist = dynamic_cast<TH1*>(obj);
         if (hist)
            ROOT::Fit::FitObject(hist, f1.get(), fitOpts, minOption, strDrawOpts, drange);

         break;
      }
      case kObjectGraph: {

         TGraph *gr = dynamic_cast<TGraph*>(obj);
         if (gr)
            ROOT::Fit::FitObject(gr, f1.get(), fitOpts, minOption, strDrawOpts, drange);
         break;
      }
      case kObjectMultiGraph: {

         TMultiGraph *mg = dynamic_cast<TMultiGraph*>(obj);
         if (mg)
            ROOT::Fit::FitObject(mg, f1.get(), fitOpts, minOption, strDrawOpts, drange);

         break;
      }
      case kObjectGraph2D: {

         TGraph2D *g2d = dynamic_cast<TGraph2D*>(obj);
         if (g2d)
            ROOT::Fit::FitObject(g2d, f1.get(), fitOpts, minOption, strDrawOpts, drange);

         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }

      default: {
         // N/A
         break;
      }
   }

   if (m.fSame && f1)
      f1->Draw("same");

   if (pad) {
      pad->Modified();
      pad->Update();
   }

   m.UpdateAdvanced(f1.get());

   std::string funcname = f1->GetName();
   if ((funcname.compare(0,4,"prev") == 0) && (funcname.find("-") > 4))
      funcname.erase(0, funcname.find("-") + 1);
   funcname = "prev"s + std::to_string(fPrevFuncs.size() + 1) + "-"s + funcname;
   f1->SetName(funcname.c_str());
   fPrevFuncs.emplace(m.fSelectedData, std::move(f1));

   UpdateFunctionsList();

   SelectFunction("previous::"s + funcname);

   if (save && (gPad != save))
      gPad = save;

   return true; // provide client with latest changes
}
