/// \file ROOT/RFitPanel.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \author Iliana Bessou <Iliana.Bessou@cern.ch>
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
      printf("FitPanel connection established %u\n", fConnId);
      fWindow->Send(fConnId, "INITDONE");
      ROOT::Experimental::RFitPanelModel6 model;
         
      //ComboBox for Data Set
      if (fHist) {
         model.fDataSet.emplace_back("0", Form("%s::%s", fHist->ClassName(), fHist->GetName()));
         model.fSelectDataId = "0";
      }

      // if (gDirectory) {
      //    TIter iter(gDirectory->GetList());
      //    TObject *item = nullptr;

      //    while ((item = iter()) != nullptr)
      //       if (item->InheritsFrom(TH1::Class()))
      //          model.fDataSet.emplace_back(item->GetName(), Form("%s::%s", item->ClassName(), item->GetName()));
      // }

       //ComboBox for Fit Function --- Type
       model.fTypeFunc.push_back(ROOT::Experimental::RComboBoxItem("0", "Predef-1D"));
       model.fTypeFunc.push_back(ROOT::Experimental::RComboBoxItem("1", "User Func"));
       model.fSelectTypeId = "0";

       //Sub ComboBox for Type Function
       model.fSelectXYId = "1";


       //corresponds when Type == Predef-1D (fSelectedTypeID == 0)
       model.fTypeXYAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vec0 = model.fTypeXYAll.back();
       vec0.push_back(ROOT::Experimental::RComboBoxItem("1", "gaus"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("2", "gausn"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("3", "expo"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("4", "landau"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("5", "landaun"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("6", "pol0"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("7", "pol1"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("8", "pol2"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("9", "pol3"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("10", "pol4"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("11", "pol5"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("12", "pol6"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("13", "pol7"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("14", "pol8"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("15", "pol9"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("16", "cheb0"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("17", "cheb1"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("18", "cheb2"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("19", "cheb3"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("20", "cheb4"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("21", "cheb5"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("22", "cheb6"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("23", "cheb7"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("24", "cheb8"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("25", "cheb9"));
       vec0.push_back(ROOT::Experimental::RComboBoxItem("26", "user"));


       //corresponds when Type == User Func (fSelectedTypeID == 1)
       model.fTypeXYAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vec1 = model.fTypeXYAll.back();
       vec1.push_back(ROOT::Experimental::RComboBoxItem("1", "chebyshev0"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("2", "chebyshev1"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("3", "chebyshev2"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("4", "chebyshev3"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("5", "chebyshev4"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("6", "chebyshev5"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("7", "chebyshev6"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("8", "chebyshev7"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("9", "chebyshev8"));
       vec1.push_back(ROOT::Experimental::RComboBoxItem("10", "chebyshev9"));

       //ComboBox for General Tab --- Method
       model.fMethod.push_back(ROOT::Experimental::RComboBoxItem("1", "Linear Chi-square"));
       model.fMethod.push_back(ROOT::Experimental::RComboBoxItem("2", "Non-Linear Chi-square"));
       model.fMethod.push_back(ROOT::Experimental::RComboBoxItem("3", "Linear Chi-square with Robust"));
       model.fMethod.push_back(ROOT::Experimental::RComboBoxItem("4", "Binned Likelihood"));
       model.fSelectMethodId = "1";

       //Sub ComboBox for Minimization Tab --- Method
       model.fSelectMethodMinId = "1";

       // corresponds to library == 0
       model.fMethodMinAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vect0 = model.fMethodMinAll.back();
       vect0.push_back(ROOT::Experimental::RComboBoxItem("1", "MIGRAD"));
       vect0.push_back(ROOT::Experimental::RComboBoxItem("2", "SIMPLEX"));
       vect0.push_back(ROOT::Experimental::RComboBoxItem("3", "SCAN"));
       vect0.push_back(ROOT::Experimental::RComboBoxItem("4", "Combination"));

       // corresponds to library == 1
       model.fMethodMinAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vect1 = model.fMethodMinAll.back();
       vect1.push_back(ROOT::Experimental::RComboBoxItem("1", "MIGRAD"));
       vect1.push_back(ROOT::Experimental::RComboBoxItem("2", "SIMPLEX"));
       vect1.push_back(ROOT::Experimental::RComboBoxItem("3", "SCAN"));
       vect1.push_back(ROOT::Experimental::RComboBoxItem("4", "Combination"));

       // corresponds to library == 2
       model.fMethodMinAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vect2 = model.fMethodMinAll.back();
       vect2.push_back(ROOT::Experimental::RComboBoxItem("1", "FUMILI"));

       // corresponds to library == 3
       model.fMethodMinAll.emplace_back();
       // std::vector<ROOT::Experimental::RComboBoxItem> &vect3 = model.fMethodMinAll.back();
       // vect3.push_back(ROOT::Experimental::RComboBoxItem("1", "Lib3_1"));
       // vect3.push_back(ROOT::Experimental::RComboBoxItem("2", "Lib3_2"));

       // corresponds to library == 4
       model.fMethodMinAll.emplace_back();
       std::vector<ROOT::Experimental::RComboBoxItem> &vect4 = model.fMethodMinAll.back();
       vect4.push_back(ROOT::Experimental::RComboBoxItem("1", "TMVA Genetic Algorithm"));

       // select items list for initial display
       model.fMethodMin = model.fMethodMinAll[model.fLibrary];
       model.fTypeXY = model.fTypeXYAll[model.fTypeId];

       if (fHist) {
          model.fMinRange = fHist->GetXaxis()->GetXmin();
          model.fMaxRange = fHist->GetXaxis()->GetXmax();

          model.fUpdateMinRange = fHist->GetXaxis()->GetXmin();
          model.fUpdateMaxRange = fHist->GetXaxis()->GetXmax();
       }

       //defined values
       model.fStep = (model.fMaxRange - model.fMinRange) / 100;
       model.fRange[0] = model.fMinRange;
       model.fRange[1] = model.fMaxRange;

       model.fUpdateRange[0] = model.fUpdateMinRange;
       model.fUpdateRange[1] = model.fUpdateMaxRange;
       //model.fOperation = 0;
       model.fFitOptions = 3;
       model.fRobust = false;
       model.fLibrary = 0;
       model.fPrint = 0;


       //Checkboxes Values
       model.fIntegral = false;
       model.fWeights = false;
       model.fBins = false;
       //model.fUseRange = false;
       model.fAddList = false;
       model.fUseGradient = false;
       model.fSame = false;
       model.fNoStore = false;
       model.fMinusErrors = false;
       //model.fImproveFit = false;

       if(model.fNoStore){
          model.fNoDrawing = true;
       }
       else{
          model.fNoDrawing = false;
       }

       if((model.fFuncChangeInt >= 6) && (model.fFuncChangeInt <= 15)){
          model.fLinear = true;

       }
       else {
          model.fLinear = false;

       }

       //Communication with the JSONModel in JS
       // TString json = TBufferJSON::ConvertToJSON(&model, gROOT->GetClass("FitPanelModel"));
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

      std::string arg1 = arg;
      arg1.erase(0,6);
      DoFit(arg1);
      return;
   }

    if (arg.find("SETCONTOUR:") == 0) {

      std::string argC = arg;
      argC.erase(0,11);
      DrawContour(argC);
      return;
   }

   if(arg.find("SETSCAN:") == 0) {

    std::string argS = arg;
    argS.erase(0,8);
    DrawScan(argS);
    return;
   }


   if (arg.find("GETPARS:") == 0) {

      RFitFunc info;
      //ROOT::Experimental::RFitPanelModel6 model;

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
            if ((par.min >= par.max) && ((par.min!=0) || (par.max != 0)))
               par.fixed = true;
         }
      } else {
         info.name = "<not exists>";
      }
      TString json = TBufferJSON::ToJSON(&info);

      fWindow->Send(fConnId, std::string("PARS:") + json.Data());
      return;
   }

   if (arg.find("SETPARS:") == 0) {
      auto info = TBufferJSON::FromJSON<RFitFunc>(arg.substr(8));

      if (info) {
         TF1 *func = dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(info->name.c_str()));

         if (func) {
            printf("Found func1 %s %p %d %d\n", info->name.c_str(), func, func->GetNpar(), (int) info->pars.size());
            // copy all parameters back to the function
            for (int n=0;n<func->GetNpar();++n) {
               func->SetParameter(n, info->pars[n].value);
               func->SetParError(n, info->pars[n].error);
               func->SetParLimits(n, info->pars[n].min, info->pars[n].max);
               if (info->pars[n].fixed)
                  func->FixParameter(n, info->pars[n].value);
             }
          }
      }
   }


   if (arg.find("GETADVANCED:") == 0) {
    RFitFunc info;
    ROOT::Experimental::RFitPanelModel6 modelAdv;

    info.name = arg.substr(12);
    TF1 *func = dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(info.name.c_str()));

    //printf("Found func1 %s %p\n", info.name.c_str(), func);

    if (func) {
     for (int n = 0; n < func->GetNpar(); ++n) {

      modelAdv.fContour1.emplace_back(Form("%d", n), Form("%s", func->GetParName(n)));
      modelAdv.fContourPar1Id = "0";
      modelAdv.fContour2.emplace_back(Form("%d", n), Form("%s", func->GetParName(n)));
      modelAdv.fContourPar2Id = "0";
      modelAdv.fScan.emplace_back(Form("%d", n), Form("%s", func->GetParName(n)));
      modelAdv.fScanId = "0";

     }
    } else {
     info.name = "<not exists>";
    }
    TString jsonModel = TBufferJSON::ToJSON(&modelAdv);

    fWindow->Send(fConnId, std::string("ADVANCED:") + jsonModel.Data());
    return;
  }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dummy function, called when "Fit" button pressed in UI
void ROOT::Experimental::RFitPanel6::DrawContour(const std::string &model)
{
  static TGraph * graph = 0;
  int colorC[3];
  std::string options;
  TBackCompFitter *fFitter = 0;
  auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanelModel6>(model);
  for(int i=0; i<3; i++){
    colorC[i] = std::stoi(obj->fColorContour[i]);
  }
  TColor *color = new TColor(1234, colorC[0], colorC[1], colorC[2]);
  

  if(!(obj->fContourImpose)) {
    if(graph){
      delete graph;
      options = "ALF";
    }
  } 
  else {
    options = "LF";
  }
  graph = new TGraph(static_cast<int>(obj->fContourPoints));

  if(obj->fContourPar1 == obj->fContourPar2) {
    Error("DrawContour", "Parameters cannot be the same");
    return;
  }


  //fFitter->Contour(obj->fContourPar1, obj->fContourPar2, graph, obj->fConfLevel);
  //graph->GetXaxis()->SetTitle( fFitter->GetParName(obj->fContourPar1) );
  //graph->GetYaxis()->SetTitle( fFitter->GetParName(obj->fContourPar2) );
  //graph->Draw( options.c_str() );
  gPad->Update();

 //printf("Points %d Contour1 %d Contour2 %d ConfLevel %f\n", obj->fContourPoints, obj->fContourPar1, obj->fContourPar2, obj->fConfLevel);
}

void ROOT::Experimental::RFitPanel6::DrawScan(const std::string &model)
{ 

  auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanelModel6>(model);
  static TGraph * graph = 0;
  TBackCompFitter *fFitter = 0;

  if(graph){
    delete graph;
  }
  graph = new TGraph(static_cast<int>(obj->fScanPoints));
  //fFitter->Scan(obj->fScanPar, graph, obj->fScanMin, obj->fScanMax);

  graph->SetLineColor(kBlue);
  graph->SetLineWidth(2);
 // graph->GetXaxis()->SetTitle(fFitter->GetParName(obj->fScanPar)); ///???????????
  graph->GetYaxis()->SetTitle("FCN");
  graph->Draw("APL");
  gPad->Update();


  //printf("SCAN Points %d, Par %d, Min %d, Max %d\n", obj->fScanPoints, obj->fScanPar, obj->fScanMin, obj->fScanMax);
}

void ROOT::Experimental::RFitPanel6::DoFit(const std::string &model)
{
   // printf("DoFit %s\n", model.c_str());
   auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanelModel6>(model);
   ROOT::Math::MinimizerOptions minOption;

   //Fitting Options
   if (obj) {

      printf("DOFIT: range %f %f select %s function %s\n ", obj->fUpdateRange[0], obj->fUpdateRange[1], obj->fSelectDataId.c_str(), obj->fSelectXYId.c_str());


      if (!obj->fRealFunc.empty()) {
         printf("GOT fRealFunc: %s\n", obj->fRealFunc.c_str());

        
      }
      else {
         obj->fRealFunc = "gaus";
         printf("%s\n", obj->fRealFunc.c_str());
      }

      if(!obj->fMinLibrary.empty()){
        printf("Min Tab: %s\n", obj->fMinLibrary.c_str());
        minOption.SetMinimizerAlgorithm(obj->fMinLibrary.c_str());
      }

      if(!obj->fErrorDef == 0) {
        minOption.SetErrorDef(obj->fErrorDef);
        //printf("Error Def %d\n", obj->fErrorDef);
      }
      else {
        minOption.SetErrorDef(1.00);
      }

      if(!obj->fMaxTol == 0) {
        minOption.SetTolerance(obj->fMaxTol);
        //printf("Tolerance %d\n", obj->fMaxTol );
      }
      else {
        minOption.SetTolerance(0.01);
      }

      if(!obj->fMaxInter == 0) {
        minOption.SetMaxIterations(obj->fMaxInter);
        //printf("Max Inte %d\n", obj->fMaxInter );
      }
      else {
        minOption.SetMaxIterations(0);
      }

      if(obj->fIntegral){
         obj->fOption = "I";
      }
      else if(obj->fMinusErrors){
         obj->fOption = "E";
      }
      else if(obj->fWeights){
         obj->fOption = "W";
      }
      else if(obj->fUseRange){
         obj->fOption = "R";
      }
      else if(obj->fNoDrawing){
         obj->fOption = "O";
      }
      else if((obj->fWeights) && (obj->fBins)){
         obj->fOption = "WW";
      }
      else if(obj->fAddList){
         obj->fOption = "+";
      }
      else if(obj->fSelectMethodId == "1"){
         obj->fOption = "P";
      }
      else if(obj->fSelectMethodId == "2"){
         obj->fOption = "L";
      }
      else {
         obj->fOption = "";
      }

      //Assign the options to Fitting function
      if (fHist && (obj->fSelectDataId == "0")) {

         fHist->Fit(obj->fRealFunc.c_str(), obj->fOption.c_str(), "*", obj->fUpdateRange[0], obj->fUpdateRange[1]);
         gPad->Update();
      }
   }

}
