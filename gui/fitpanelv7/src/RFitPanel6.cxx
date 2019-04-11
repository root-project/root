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
#include "TROOT.h"
#include "TPad.h"
#include "TDirectory.h"
#include "TBufferJSON.h"
#include <sstream>
#include <iostream>

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
       //model.fDataSet.push_back(ROOT::Experimental::RComboBoxItem("1", "No Selection"));
      if (fHist) {
         model.fDataSet.emplace_back("0", Form("%s::%s", fHist->ClassName(), fHist->GetName()));
         model.fSelectDataId = "0";
      }

      if (gDirectory) {
         TIter iter(gDirectory->GetList());
         TObject *item = nullptr;

         while ((item = iter()) != nullptr)
            if (item->InheritsFrom(TH1::Class()))
               model.fDataSet.emplace_back(item->GetName(), Form("%s::%s", item->ClassName(), item->GetName()));
      }

       model.fDataSet.emplace_back("1", "*TH1F::hpx");
       model.fDataSet.emplace_back("2", "*TH2F::hpxhpy");
       model.fDataSet.emplace_back("3", "*TProfile::hprof");
       model.fDataSet.emplace_back("4", "*TNtuple::ntuple");
       model.fSelectDataId = "2";

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

       //Contour ComboBoxes
       model.fContourPar1.push_back(ROOT::Experimental::RComboBoxItem("1","Coeff0"));
       model.fContourPar1.push_back(ROOT::Experimental::RComboBoxItem("2","Coeff1"));
       model.fContourPar1.push_back(ROOT::Experimental::RComboBoxItem("3","Coeff3"));
       model.fContourPar1Id = "1";

       model.fContourPar2.push_back(ROOT::Experimental::RComboBoxItem("1","Coeff0"));
       model.fContourPar2.push_back(ROOT::Experimental::RComboBoxItem("2","Coeff1"));
       model.fContourPar2.push_back(ROOT::Experimental::RComboBoxItem("3","Coeff3"));
       model.fContourPar2Id = "2";

       //Scan ComboBox
       model.fScanPar.push_back(ROOT::Experimental::RComboBoxItem("1","Coeff0"));
       model.fScanPar.push_back(ROOT::Experimental::RComboBoxItem("2","Coeff1"));
       model.fScanPar.push_back(ROOT::Experimental::RComboBoxItem("3","Coeff3"));
       model.fScanParId = "1";

       model.fUpdateMinRange = -4;
       model.fUpdateMaxRange = 4;
       model.fMinRange = -4;
       model.fMaxRange = 4;
       if (fHist) {
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dummy function, called when "Fit" button pressed in UI

void ROOT::Experimental::RFitPanel6::DoFit(const std::string &model)
{
   // printf("DoFit %s\n", model.c_str());
   auto obj = TBufferJSON::FromJSON<ROOT::Experimental::RFitPanelModel6>(model);
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
