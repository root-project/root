// \file ROOT/RFitPanel6Model.hxx
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

#ifndef ROOT_RFitPanel6Model
#define ROOT_RFitPanel6Model

#include <vector>
#include <string>

class TH1;

namespace ROOT {
namespace Experimental {

struct RComboBoxItem {
   std::string fId;
   std::string fSet;
   RComboBoxItem() = default;
   RComboBoxItem(const std::string &id, const std::string &set) : fId(id), fSet(set) {}
};

struct RFitFuncParameter {
   int ipar{0};
   std::string name;
   double value{0.};
   bool fixed{false};
   double error{0.};
   double min{0.}, max{0.};
   RFitFuncParameter() = default;
   RFitFuncParameter(int _ipar, const std::string &_name) : ipar(_ipar), name(_name) {}
};

struct RFitFunc {
   std::string name;
   std::vector<RFitFuncParameter> pars;
};

// Structure for the main fit panel model
struct RFitPanel6Model {
   std::vector<RComboBoxItem> fDataSet;
   std::string fSelectDataId;
   std::vector<RComboBoxItem> fTypeFunc;
   std::string fSelectXYId;
   std::vector<RComboBoxItem> fMethod;
   std::string fSelectMethodId;
   std::vector<RComboBoxItem> fContour1;
   std::string fContourPar1Id;
   std::vector<RComboBoxItem> fContour2;
   std::string fContourPar2Id;
   std::vector<RComboBoxItem> fScan;
   std::string fScanId;
   std::string fRealFunc;
   std::string fOption;
   std::string fFuncChange;
   std::string fMinLibrary;

   // all combo items for all methods

   // Minimization Tab
   std::vector<std::vector<RComboBoxItem>> fMethodMinAll;
   // Fit Function --- Type
   std::vector<std::vector<RComboBoxItem>> fTypeXYAll;

   std::string fSelectTypeId;
   std::string fSelectMethodMinId;

   float fUpdateMinRange{0};
   float fUpdateMaxRange{1};
   float fMinRange{0};
   float fMaxRange{1};
   float fStep{0.1};
   float fRange[2];
   float fUpdateRange[2];
   // float fOperation{0};
   float fFitOptions{0};
   bool fLinear{false};
   bool fRobust{false};
   int fLibrary{0};
   int fPrint{0};
   float fErrorDef{1.00};
   float fMaxTol{0.01};
   int fMaxInter{0};

   // convert fSelectTypeID from string to int
   int fTypeId{0};
   int fFuncChangeInt{0};

   // Checkboxes Options
   bool fIntegral{false};
   bool fMinusErrors{false};
   bool fWeights{false};
   bool fBins{false};
   bool fUseRange{false};
   // bool fImproveFit {false};
   bool fAddList{false};
   bool fUseGradient{false};
   bool fSame{false};
   bool fNoDrawing{false};
   bool fNoStore{false};

   /////////Advanced Options

   // Contour Tab
   int fContourPar1{0};
   int fContourPar2{0};
   int fContourPoints{0};
   float fConfLevel{0.};
   bool fContourImpose{false};
   std::string fColorContour[3];

   // Scan Tab
   int fScanPoints{0};
   int fScanPar{0};
   int fScanMin{0};
   int fScanMax{0};

   void Initialize(TH1 *hist);
};

} // namespace Experimental
} // namespace ROOT

#endif
