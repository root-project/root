/// \file RFitPanel6Model.cxx
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

#include <ROOT/RFitPanel6Model.hxx>

#include "TH1.h"
#include "TDirectory.h"

#include "TF1.h"
#include "TF2.h"

using namespace std::string_literals;


void ROOT::Experimental::RFitFuncParsList::GetParameters(TF1 *func)
{
   pars.clear();

   for (int n = 0; n < func->GetNpar(); ++n) {
      pars.emplace_back(n, func->GetParName(n));
      auto &par = pars.back();

      par.value = func->GetParameter(n);
      par.error = func->GetParError(n);
      func->GetParLimits(n, par.min, par.max);
      if ((par.min >= par.max) && ((par.min != 0) || (par.max != 0)))
         par.fixed = true;
   }
}

void ROOT::Experimental::RFitFuncParsList::SetParameters(TF1 *func)
{
   if (func->GetNpar() != (int) pars.size()) {
      ::Error("RFitFuncParsList::SetParameters", "Mismatch in parameters numbers");
      return;
   }

   for (int n = 0; n < func->GetNpar(); ++n) {
      if (pars[n].name.compare(func->GetParName(n)) != 0) {
         ::Error("RFitFuncParsList::SetParameters", "Mismatch in parameter %d name %s %s", n, pars[n].name.c_str(), func->GetParName(n));
         return;
      }

      func->SetParameter(n, pars[n].value);
      func->SetParError(n, pars[n].error);
      if (pars[n].fixed) {
         func->FixParameter(n, pars[n].value);
      } else {
         func->ReleaseParameter(n);
         if (pars[n].min < pars[n].max)
            func->SetParLimits(n, pars[n].min, pars[n].max);
      }
   }
}

///////////////////////////////

TH1* ROOT::Experimental::RFitPanel6Model::FindHistogram(const std::string &id, TH1 *hist)
{
   if (id == "__hist__") return hist;
   if ((id.compare(0,6,"gdir::") != 0) || !gDirectory) return nullptr;

   std::string hname = id.substr(6);

   return dynamic_cast<TH1*> (gDirectory->GetList()->FindObject(hname.c_str()));
}

void ROOT::Experimental::RFitPanel6Model::Initialize(TH1 *hist)
{
   // build list of available histograms, as id use name from gdir
   std::string histid;

   if (gDirectory) {
      TIter iter(gDirectory->GetList());
      TObject *item = nullptr;

       while ((item = iter()) != nullptr)
         if (item->InheritsFrom(TH1::Class())) {
            std::string dataid = "gdir::"s + item->GetName();

            if (hist && (hist == item)) histid = dataid;
            fDataSet.emplace_back(dataid, Form("%s::%s", item->ClassName(), item->GetName()));
         }
   }

   if (hist && histid.empty()) {
      histid = "__hist__";
      fDataSet.emplace_back(histid, Form("%s::%s", hist->ClassName(), hist->GetName()));
   }
   fSelectDataId = histid;

   // build list of available functions

   // ComboBox for Fit Function --- Type
   fTypeFunc.emplace_back("0", "Predef-1D");
   fTypeFunc.emplace_back("1", "Predef-2D");
   fTypeFunc.emplace_back("2", "User");
   fSelectTypeFunc = "0";

   // Sub ComboBox for Type Function
   fSelectedFunc = "gaus";

   // corresponds when Type == Predef-1D (fSelectedTypeID == 0)
   fFuncListAll.emplace_back();
   fFuncListAll.emplace_back();

   auto &vec1d = fFuncListAll[0]; // for 1D histograms
   auto &vec2d = fFuncListAll[1]; // for 2D histograms

   TIter iter(gROOT->GetListOfFunctions());
   TObject *func = nullptr;
   while ((func = iter()) != nullptr) {
      if (dynamic_cast<TF2 *>(func))
         vec2d.emplace_back(func->GetName(), (dynamic_cast<TF2 *>(func))->IsLinear());
      else if (dynamic_cast<TF1 *>(func))
         vec1d.emplace_back(func->GetName(), (dynamic_cast<TF1 *>(func))->IsLinear());
   }
   if (vec1d.empty()) vec1d.emplace_back("none");
   if (vec2d.empty()) vec2d.emplace_back("none");

   //std::sort(vec1d.begin(), vec1d.end());
   //std::sort(vec2d.begin(), vec2d.end());

   // corresponds when Type == User Func (fSelectedTypeID == 1)
   fFuncListAll.emplace_back();
   fFuncListAll.back().emplace_back("user");

   // ComboBox for General Tab --- Method
   fMethod.emplace_back("1", "Linear Chi-square");
   fMethod.emplace_back("2", "Non-Linear Chi-square");
   fMethod.emplace_back("3", "Linear Chi-square with Robust");
   fMethod.emplace_back("4", "Binned Likelihood");
   fSelectMethodId = "1";

   // Sub ComboBox for Minimization Tab --- Method
   fSelectMethodMinId = "1";

   fLibrary = 0;

   // corresponds to library == 0
   fMethodMinAll.emplace_back();
   fMethodMinAll.back() = {{ "1", "MIGRAD" }, {"2", "SIMPLEX"}, {"3", "SCAN"}, {"4", "Combination"}};

   // corresponds to library == 1
   fMethodMinAll.emplace_back();
   fMethodMinAll.back() = {{ "1", "MIGRAD" }, {"2", "SIMPLEX"}, {"3", "SCAN"}, {"4", "Combination"}};

   // corresponds to library == 2
   fMethodMinAll.emplace_back();
   fMethodMinAll.back() = {{ "1", "FUMILI" }};

   // corresponds to library == 3
   fMethodMinAll.emplace_back();

   // corresponds to library == 4
   fMethodMinAll.emplace_back();
   fMethodMinAll.back() = {{ "1", "TMVA Genetic Algorithm" }};

   if (hist) {
      fUpdateMinRange = fMinRange = hist->GetXaxis()->GetXmin();
      fUpdateMaxRange = fMaxRange = hist->GetXaxis()->GetXmax();
   } else {
      fUpdateMinRange = fMinRange = 0.;
      fUpdateMaxRange = fMaxRange = 100.;
   }

   // defined values
   fStep = (fMaxRange - fMinRange) / 100;
   fRange[0] = fMinRange;
   fRange[1] = fMaxRange;

   fUpdateRange[0] = fUpdateMinRange;
   fUpdateRange[1] = fUpdateMaxRange;
   // fOperation = 0;
   fFitOptions = 3;
   fRobust = false;
   fPrint = 0;

   // Checkboxes Values
   fIntegral = false;
   fWeights = false;
   fBins = false;
   // fUseRange = false;
   fAddList = false;
   fUseGradient = false;
   fSame = false;
   fNoStore = false;
   fMinusErrors = false;
   // fImproveFit = false;

   if (fNoStore) {
      fNoDrawing = true;
   } else {
      fNoDrawing = false;
   }

   if ((fFuncChangeInt >= 6) && (fFuncChangeInt <= 15)) {
      fLinear = true;
   } else {
      fLinear = false;
   }
}

std::string ROOT::Experimental::RFitPanel6Model::GetFitOption()
{
   std::string opt;

   if (fIntegral) {
      opt = "I";
   } else if (fMinusErrors) {
      opt = "E";
   } else if (fWeights) {
      opt = "W";
   } else if (fUseRange) {
      opt = "R";
   } else if (fNoDrawing) {
      opt = "O";
   } else if (fWeights && fBins) {
      opt = "WW";
   } else if (fAddList) {
      opt = "+";
   } else if (fSelectMethodId == "1") {
      opt = "P";
   } else if (fSelectMethodId == "2") {
      opt = "L";
   }

   return opt;
}

