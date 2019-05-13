// \file ROOT/RFitPanelModel.hxx
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

#ifndef ROOT_RFitPanelModel
#define ROOT_RFitPanelModel

#include <vector>
#include <string>

class TH1;
class TF1;

namespace ROOT {
namespace Experimental {

/// Generic item for ui5 ComboBox
struct RComboBoxItem {
   std::string key;
   std::string value;
   RComboBoxItem() = default;
   RComboBoxItem(const std::string &_key, const std::string &_value) : key(_key), value(_value) {}
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


/** Data structure for the fit panel */

struct RFitPanelModel {

   std::string fTitle;                      ///< title of the fit panel

   std::vector<RComboBoxItem> fDataSet;     ///< list of available data sources
   std::string fSelectedData;               ///< selected data
   std::vector<RFitFuncInfo>   fFuncList;   ///< all available fit functions
   std::string fSelectedFunc;               ///< name of selected fit function


   // General tab

   // Method
   std::vector<RComboBoxItem> fFitMethods;   ///< all supported fit methods
   std::string fFitMethod;                   ///< selected fit method

   bool fLinearFit{false};
   bool fRobust{false};
   float fRobustLevel{0.95};

   // Fit Options
   bool fIntegral{false};
   bool fUseRange{false};
   bool fBestErrors{false};
   bool fImproveFitResults{false};
   bool fAllWeights1{false};
   bool fAddToList{false};
   bool fEmptyBins1{false};
   bool fUseGradient{false};

   // Draw Options
   bool fSame{false};
   bool fNoDrawing{false};
   bool fNoStoreDraw{false};


   // all combo items for all methods

   // Minimization Tab
   int fLibrary{0};   ///< selected minimization library
   std::vector<RComboBoxItem> fMethodMinAll;  // all items for all methods
   std::string fSelectMethodMin;

   // range selection
   bool fShowRangeX{true};
   float fMinRangeX{0};
   float fMaxRangeX{1};
   float fStepX{0.01};
   float fRangeX[2] = {0,1};

   bool fShowRangeY{false};
   float fMinRangeY{0};
   float fMaxRangeY{1};
   float fStepY{0.01};
   float fRangeY[2] = {0,1};

   // float fOperation{0};
   float fFitOptions{0};
   int fPrint{0};
   float fErrorDef{1.00};
   float fMaxTol{0.01};
   int fMaxInter{0};

   // convert fSelectTypeID from string to int
   int fTypeId{0};

   // bool fImproveFit {false};


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

   RFitPanelModel() { Initialize(); }

   void Initialize();

   bool SelectHistogram(const std::string &hname, TH1 *hist);

   bool SelectFunc(const std::string &name, TH1 *hist);

   void UpdateRange(TH1 *hist);

   void UpdateAdvanced(TF1 *func);

   TF1 *UpdateFuncList(TH1 *hist = nullptr, bool select_hist_func = false);

   bool IsSelectedHistogram() const { return !fSelectedData.empty(); }

   TH1* GetSelectedHistogram(TH1 *hist = nullptr);

   TF1 *FindFunction(const std::string &fname, TH1 *hist = nullptr);

   std::string GetFitOption();
};

} // namespace Experimental
} // namespace ROOT

#endif
