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

void ROOT::Experimental::RFitPanel6Model::Initialize(TH1 *hist)
{
   if (hist) {
      fDataSet.emplace_back("0", Form("%s::%s", hist->ClassName(), hist->GetName()));
      fSelectDataId = "0";
   }

   // if (gDirectory) {
   //    TIter iter(gDirectory->GetList());
   //    TObject *item = nullptr;

   //    while ((item = iter()) != nullptr)
   //       if (item->InheritsFrom(TH1::Class()))
   //          fDataSet.emplace_back(item->GetName(), Form("%s::%s", item->ClassName(), item->GetName()));
   // }

   // ComboBox for Fit Function --- Type
   fTypeFunc.emplace_back("0", "Predef-1D");
   fTypeFunc.emplace_back("1", "User Func");
   fSelectTypeId = "0";

   // Sub ComboBox for Type Function
   fSelectXYId = "1";

   // corresponds when Type == Predef-1D (fSelectedTypeID == 0)
   fTypeXYAll.emplace_back();
   auto &vec0 = fTypeXYAll.back();
   vec0.emplace_back("1", "gaus");
   vec0.emplace_back("2", "gausn");
   vec0.emplace_back("3", "expo");
   vec0.emplace_back("4", "landau");
   vec0.emplace_back("5", "landaun");
   vec0.emplace_back("6", "pol0");
   vec0.emplace_back("7", "pol1");
   vec0.emplace_back("8", "pol2");
   vec0.emplace_back("9", "pol3");
   vec0.emplace_back("10", "pol4");
   vec0.emplace_back("11", "pol5");
   vec0.emplace_back("12", "pol6");
   vec0.emplace_back("13", "pol7");
   vec0.emplace_back("14", "pol8");
   vec0.emplace_back("15", "pol9");
   vec0.emplace_back("16", "cheb0");
   vec0.emplace_back("17", "cheb1");
   vec0.emplace_back("18", "cheb2");
   vec0.emplace_back("19", "cheb3");
   vec0.emplace_back("20", "cheb4");
   vec0.emplace_back("21", "cheb5");
   vec0.emplace_back("22", "cheb6");
   vec0.emplace_back("23", "cheb7");
   vec0.emplace_back("24", "cheb8");
   vec0.emplace_back("25", "cheb9");
   vec0.emplace_back("26", "user");

   // corresponds when Type == User Func (fSelectedTypeID == 1)
   fTypeXYAll.emplace_back();
   auto &vec1 = fTypeXYAll.back();
   vec1.emplace_back("1", "chebyshev0");
   vec1.emplace_back("2", "chebyshev1");
   vec1.emplace_back("3", "chebyshev2");
   vec1.emplace_back("4", "chebyshev3");
   vec1.emplace_back("5", "chebyshev4");
   vec1.emplace_back("6", "chebyshev5");
   vec1.emplace_back("7", "chebyshev6");
   vec1.emplace_back("8", "chebyshev7");
   vec1.emplace_back("9", "chebyshev8");
   vec1.emplace_back("10", "chebyshev9");

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
   auto &methods0 = fMethodMinAll.back();
   methods0.emplace_back("1", "MIGRAD");
   methods0.emplace_back("2", "SIMPLEX");
   methods0.emplace_back("3", "SCAN");
   methods0.emplace_back("4", "Combination");

   // corresponds to library == 1
   fMethodMinAll.emplace_back();
   auto &methods1 = fMethodMinAll.back();
   methods1.emplace_back("1", "MIGRAD");
   methods1.emplace_back("2", "SIMPLEX");
   methods1.emplace_back("3", "SCAN");
   methods1.emplace_back("4", "Combination");

   // corresponds to library == 2
   fMethodMinAll.emplace_back();
   auto &methods2 = fMethodMinAll.back();
   methods2.emplace_back("1", "FUMILI");

   // corresponds to library == 3
   fMethodMinAll.emplace_back();
   // auto &methods3 = fMethodMinAll.back();
   // methods3.emplace_back("1", "Lib3_1");
   // methods3.emplace_back("2", "Lib3_2");

   // corresponds to library == 4
   fMethodMinAll.emplace_back();
   auto &methods4 = fMethodMinAll.back();
   methods4.emplace_back("1", "TMVA Genetic Algorithm");

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
