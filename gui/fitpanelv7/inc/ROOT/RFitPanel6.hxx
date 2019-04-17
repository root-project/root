// \file ROOT/RFitPanel6.hxx
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

#ifndef ROOT_RFitPanel6
#define ROOT_RFitPanel6

#include <ROOT/RWebWindow.hxx>

#include "TH1.h"

#include <vector>

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
   double valuePar{0.};
   double error{0.};
   double min{0.}, max{0.};
   RFitFuncParameter() = default;
   RFitFuncParameter(int _ipar, const std::string &_name, double _value, double _error, double _min, double _max) :
      ipar(_ipar), name(_name), valuePar(_value), error(_error), min(_min), max(_max) {}
};

struct RFitFunc {
   std::string name;
   double value{0.};
   double min{0.};
   double max{0.};
   double error{0.};
   std::vector<RFitFuncParameter> pars;
};

//Structure for the main fit panel model
struct RFitPanelModel6 {
   std::vector<RComboBoxItem> fDataSet;
   std::string fSelectDataId;
   std::vector<RComboBoxItem> fTypeFunc;
   std::string fSelectXYId;
   std::vector<RComboBoxItem> fMethod;
   std::string fSelectMethodId;
   std::vector<RComboBoxItem> fContourPar1;
   std::string fContourPar1Id;
   std::vector<RComboBoxItem> fContourPar2;
   std::string fContourPar2Id;
   std::vector<RComboBoxItem> fScanPar;
   std::string fScanParId;
   std::string fRealFunc;
   std::string fOption;
   std::string fFuncChange;

   // all combo items for all methods

   //Minimization Tab
   std::vector<std::vector<RComboBoxItem>> fMethodMinAll;
   //Fit Function --- Type
   std::vector<std::vector<RComboBoxItem>> fTypeXYAll;

   std::vector<RComboBoxItem> fMethodMin;
   std::vector<RComboBoxItem> fTypeXY;

   std::string fSelectTypeId;
   std::string fSelectMethodMinId;

   float fUpdateMinRange{0};
   float fUpdateMaxRange{1};
   float fMinRange{0};
   float fMaxRange{1};
   float fStep{0.1};
   float fRange[2];
   float fUpdateRange[2];
   //float fOperation{0};
   float fFitOptions{0};
   bool fLinear{false};
   bool fRobust{false};
   int fLibrary{0};
   int fPrint{0};

   //convert fSelectTypeID from string to int
   int fTypeId = atoi(fSelectTypeId.c_str());
   int fFuncChangeInt = atoi(fFuncChange.c_str());

   //Checkboxes Options
   bool fIntegral{false};
   bool fMinusErrors {false};
   bool fWeights{false};
   bool fBins{false};
   bool fUseRange {false};
   //bool fImproveFit {false};
   bool fAddList {false};
   bool fUseGradient {false};
   bool fSame {false};
   bool fNoDrawing {};
   bool fNoStore {false};
};

class RFitPanel6 {

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! connection id
   TH1 *fHist{nullptr};

   std::shared_ptr<RWebWindow> fWindow; ///!< configured display

   /// process data from UI
   void ProcessData(unsigned connid, const std::string &arg);

   void DoFit(const std::string &model);

public:
   /// normal constructor
   RFitPanel6(const std::string &title = "Fit panel") : fTitle(title) {}

   /// destructor
   virtual ~RFitPanel6() {}

   // method required when any panel want to be inserted into the RCanvas
   std::shared_ptr<RWebWindow> GetWindow();

   void AssignHistogram(TH1 *hist)
   {
      fHist = hist;
   }

   /// show FitPanel in specified place
   void Show(const std::string &where = "");

   /// hide FitPanel
   void Hide();

};

} // namespace Experimental
} // namespace ROOT

#endif
