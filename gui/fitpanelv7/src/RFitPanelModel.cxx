/// \file RFitPanelModel.cxx
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

#include <ROOT/RFitPanelModel.hxx>

#include "TH1.h"
#include "TDirectory.h"

#include "TF1.h"
#include "TF2.h"

using namespace std::string_literals;


void ROOT::Experimental::RFitFuncParsList::Clear()
{
   pars.clear();
   name.clear();
   haspars = false;
}

void ROOT::Experimental::RFitFuncParsList::GetParameters(TF1 *func)
{
   pars.clear();
   haspars = true;

   for (int n = 0; n < func->GetNpar(); ++n) {
      pars.emplace_back(n, func->GetParName(n));
      auto &par = pars.back();

      par.value = std::to_string(func->GetParameter(n));
      par.error = std::to_string(func->GetParError(n));
      double min, max;
      func->GetParLimits(n, min, max);
      par.min = std::to_string(min);
      par.max = std::to_string(max);

      if ((min >= max) && ((min != 0) || (max != 0)))
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

      func->SetParameter(n, std::stod(pars[n].value));
      func->SetParError(n, std::stod(pars[n].error));
      if (pars[n].fixed) {
         func->FixParameter(n, std::stod(pars[n].value));
      } else {
         func->ReleaseParameter(n);
         double min = std::stod(pars[n].min);
         double max = std::stod(pars[n].max);
         if (min < max)
            func->SetParLimits(n, min, max);
      }
   }
}

///////////////////////////////

TH1* ROOT::Experimental::RFitPanelModel::GetSelectedHistogram(TH1 *hist)
{
   if (fSelectedData == "__hist__") return hist;
   if ((fSelectedData.compare(0,6,"gdir::") != 0) || !gDirectory) return nullptr;

   std::string hname = fSelectedData.substr(6);

   return dynamic_cast<TH1*> (gDirectory->GetList()->FindObject(hname.c_str()));
}


// Configure usage of histogram

bool ROOT::Experimental::RFitPanelModel::SelectHistogram(const std::string &hname, TH1 *hist)
{

   std::string histid;

   fDataSet.clear();
   TH1 *selected = nullptr;

   if (gDirectory) {
      TIter iter(gDirectory->GetList());
      TObject *item = nullptr;

       while ((item = iter()) != nullptr)
         if (item->InheritsFrom(TH1::Class())) {
            std::string dataid = "gdir::"s + item->GetName();

            if (hist && (hist == item)) {
               histid = dataid;
               selected = hist;
            } else if (!hname.empty() && hname.compare(item->GetName())) {
               histid = dataid;
               selected = dynamic_cast<TH1 *> (item);
            }
            fDataSet.emplace_back(dataid, Form("%s::%s", item->ClassName(), item->GetName()));
         }
   }

   if (hist && histid.empty()) {
      selected = hist;
      histid = "__hist__";
      fDataSet.emplace_back(histid, Form("%s::%s", hist->ClassName(), hist->GetName()));
   }

   fSelectedData = histid;

   UpdateRange(selected);

   auto *hfunc = UpdateFuncList(selected);

   UpdateAdvanced(hfunc);

   return selected != nullptr;
}

void ROOT::Experimental::RFitPanelModel::UpdateRange(TH1 *hist)
{
   fShowRangeX = false;
   fShowRangeY = false;
   fMinRangeX = 0.;
   fMaxRangeX = 100.;
   fMinRangeY = 0.;
   fMaxRangeY = 100.;

   if (hist) {
      fShowRangeX = true;
      fMinRangeX = hist->GetXaxis()->GetXmin();
      fMaxRangeX = hist->GetXaxis()->GetXmax();
      if (hist->GetDimension() > 1) {
         fShowRangeY = true;
         fMinRangeY = hist->GetYaxis()->GetXmin();
         fMaxRangeY = hist->GetYaxis()->GetXmax();
      }
   }

   // defined values
   fStepX = (fMaxRangeX - fMinRangeX) / 100;
   fRangeX[0] = fMinRangeX;
   fRangeX[1] = fMaxRangeX;

   fStepY = (fMaxRangeY - fMinRangeY) / 100;
   fRangeY[0] = fMinRangeY;
   fRangeY[1] = fMaxRangeY;
}

bool ROOT::Experimental::RFitPanelModel::SelectFunc(const std::string &name, TH1 *hist)
{
   fSelectedFunc = name;

   fFuncPars.Clear();

   TF1 *func = FindFunction(name, hist);

   if (func) {
      fFuncPars.name = name;
      fFuncPars.GetParameters(func);
   } else {
      fFuncPars.name = "<not exists>";
   }

   return func != nullptr;
}


TF1 *ROOT::Experimental::RFitPanelModel::UpdateFuncList(TH1 *hist, bool select_hist_func)
{
   int ndim = hist ? hist->GetDimension() : 1;

   fFuncList.clear();

   TIter iter(gROOT->GetListOfFunctions());
   TObject *func = nullptr;
   while ((func = iter()) != nullptr) {
      TF1 *f1 = dynamic_cast<TF1 *>(func);
      if (!f1) continue;
      TF2 *f2 = dynamic_cast<TF2 *>(f1);

      if (((ndim==2) && f2) || ((ndim==1) && !f2))
         fFuncList.emplace_back(f1->GetName(), f1->IsLinear());
   }

   TF1 *hfunc = nullptr;
   if (hist) {
      TObject *obj = nullptr;
      TIter hiter(hist->GetListOfFunctions());
      while ((obj = hiter()) != nullptr) {
         hfunc = dynamic_cast<TF1*> (obj);
         if (hfunc) break;
      }
   }

   if (hfunc) {
      fFuncList.emplace_back("hist::"s + hfunc->GetName(), hfunc->IsLinear());
      if (select_hist_func) fSelectedFunc = "hist::"s + hfunc->GetName();
   }

   fFuncList.emplace_back("user");

   return hfunc;
}


void ROOT::Experimental::RFitPanelModel::Initialize()
{
   // build list of available histograms, as id use name from gdir
   fSelectedData = "";

   // build list of available functions

   // Sub ComboBox for Type Function
   fSelectedFunc = "";
   UpdateFuncList();

   // corresponds when Type == User Func (fSelectedTypeID == 1)

   // ComboBox for General Tab --- Method
   fFitMethods = { {"P", "Chi-square"},
                   {"L", "Log Likelihood"},
                   {"WL", "Binned LogLikelihood"} };
   fFitMethod = "P";

   fLinearFit = false;
   fRobust = false;
   fRobustLevel = 0.95;

   fIntegral = false;
   fAllWeights1 = false;
   fAddToList = false;
   fEmptyBins1 = false;
   fUseGradient = false;

   fSame = false;
   fNoDrawing = false;
   fNoStoreDraw = false;



   // Minimization method
   fLibrary = 0;
   // corresponds to library == 0
   fMethodMinAll = {
         {"0", "MIGRAD"}, {"0", "SIMPLEX"}, {"0", "SCAN"}, {"0", "Combination"},
         {"1", "MIGRAD"}, {"1", "SIMPLEX"}, {"1", "SCAN"}, {"1", "Combination"},
         {"2", "FUMILI"},
         {"3", "none"},
         {"4", "TMVA Genetic Algorithm"}
   };
   fSelectMethodMin = "MIGRAD";

   // fOperation = 0;
   fFitOptions = 3;
   fPrint = 0;
}

TF1 *ROOT::Experimental::RFitPanelModel::FindFunction(const std::string &funcname, TH1 *hist)
{
   if (funcname.compare(0,6,"hist::")==0) {
      TH1 *h1 = GetSelectedHistogram(hist);
      if (!h1) return nullptr;
      return dynamic_cast<TF1 *> (h1->GetListOfFunctions()->FindObject(funcname.substr(6).c_str()));
   }

   return dynamic_cast<TF1 *>(gROOT->GetListOfFunctions()->FindObject(funcname.c_str()));
}


/// Update advanced parameters associated with fit function for histogram

void ROOT::Experimental::RFitPanelModel::UpdateAdvanced(TF1 *func)
{
   fContour1.clear();
   fContour2.clear();
   fScan.clear();
   fContourPar1Id = "0";
   fContourPar2Id = "0";
   fScanId = "0";

   fHasAdvanced = (func!=nullptr);

   if (func) {
      for (int n = 0; n < func->GetNpar(); ++n) {
         fContour1.emplace_back(std::to_string(n), func->GetParName(n));
         fContour2.emplace_back(std::to_string(n), func->GetParName(n));
         fScan.emplace_back(std::to_string(n), func->GetParName(n));
      }
      fFuncPars.GetParameters(func); // take func parameters
      fFuncPars.name = "hist::"s + func->GetName(); // clearly mark this as function from histogram
   } else {
      // fFuncPars.Clear();
   }
}


std::string ROOT::Experimental::RFitPanelModel::GetFitOption()
{
   std::string opt = fFitMethod;

   if (fIntegral) opt.append("I");
   if (fUseRange) opt.append("R");
   if (fBestErrors) opt.append("E");
   if (fImproveFitResults) opt.append("M");
   if (fAddToList) opt.append("+");
   if (fUseGradient) opt.append("G");

   if (fEmptyBins1)
      opt.append("WW");
   else if (fAllWeights1)
      opt.append("W");

   if (fNoStoreDraw)
      opt.append("N");
   else if (fNoDrawing)
      opt.append("O");

   return opt;
}

