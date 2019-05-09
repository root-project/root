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
class TF1;

namespace ROOT {
namespace Experimental {

struct RComboBoxItem {
   std::string fId;
   std::string fSet;
   RComboBoxItem() = default;
   RComboBoxItem(const std::string &id, const std::string &set) : fId(id), fSet(set) {}
};

/// Basic function info, used in combo boxes
struct RFitFuncInfo {
   std::string name;
   bool linear{false};

   RFitFuncInfo() = default;
   RFitFuncInfo(const std::string &_name, bool _linear = false) : name(_name), linear(_linear) {}
};

/// Function parameter info, used in edit parameters dialog

struct RFitFuncParameter {
   int ipar{0};
   std::string name;
   std::string value;
   bool fixed{false};
   std::string error;
   std::string min;
   std::string max;
   RFitFuncParameter() = default;
   RFitFuncParameter(int _ipar, const std::string &_name) : ipar(_ipar), name(_name) {}
};

/// Class used to transfer functions parameters list from/to client
struct RFitFuncParsList {
   bool haspars{false};
   std::string name;
   std::vector<RFitFuncParameter> pars;
   void GetParameters(TF1 *f1);
   void SetParameters(TF1 *f1);
   void Clear();
};


// Structure for the main fit panel model
struct RFitPanel6Model {
   std::vector<RComboBoxItem> fDataSet;
   std::string fSelectedData;
   std::vector<RComboBoxItem> fMethod;
   std::string fSelectMethodId;
   std::string fRealFunc;                    ///< name of the fit function
   std::string fMinLibrary;

   // all combo items for all methods

   // Minimization Tab
   std::vector<std::vector<RComboBoxItem>> fMethodMinAll;

   std::vector<RFitFuncInfo>   fFuncList;   ///< all available fit functions
   std::string fSelectedFunc;                  ///< name of selected fit function

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


   /// Parameters

   RFitFuncParsList fFuncPars;

   /////////Advanced Options

   bool fHasAdvanced{false};
   std::vector<RComboBoxItem> fContour1;
   std::string fContourPar1Id;
   std::vector<RComboBoxItem> fContour2;
   std::string fContourPar2Id;
   std::vector<RComboBoxItem> fScan;
   std::string fScanId;

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

   RFitPanel6Model() { Initialize(); }

   void Initialize();

   bool SelectHistogram(const std::string &hname, TH1 *hist);

   bool SelectFunc(const std::string &name, TH1 *hist);

   void UpdateRange(TH1 *hist);

   void UpdateAdvanced(TH1 *hist);

   void UpdateFuncList(TH1 *hist = nullptr);

   bool IsSelectedHistogram() const { return !fSelectedData.empty(); }

   TH1* GetSelectedHistogram(TH1 *hist = nullptr);

   TF1 *FindFunction(const std::string &fname, TH1 *hist = nullptr);

   std::string GetFitOption();
};

} // namespace Experimental
} // namespace ROOT

#endif
