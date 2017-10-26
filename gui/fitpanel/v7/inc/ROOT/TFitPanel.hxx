/// \file ROOT/TFitPanel.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-10-24
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TFitPanel
#define ROOT7_TFitPanel

#include <ROOT/TWebWindow.hxx>

#include <ROOT/TCanvas.hxx>

#include "ROOT/THist.hxx"

#include <vector>

namespace ROOT {
namespace Experimental {

struct ComboBoxItem {
   std::string fId{};
   std::string fName{};
   ComboBoxItem() = default;
   ComboBoxItem(const std::string &id, const std::string &name) : fId(id), fName(name) {}
};

struct TFitPanelModel {
   std::vector<ComboBoxItem> fDataNames{};
   std::string fSelectDataId{};
   std::vector<ComboBoxItem> fModelNames{};
   std::string fSelectModelId{};
   TFitPanelModel() = default;
};


class TFitPanel  {

   std::string     fTitle{};              ///<! title
   unsigned       fConnId{0};             ///<! connection id

   std::shared_ptr<TWebWindow> fWindow{}; ///!< configured display

   std::shared_ptr<TCanvas> fCanvas{};    ///!< canvas used to display results

   std::shared_ptr<TH1D>  fFitHist{};     ///!< histogram created when fit is performed

   /// Disable copy construction.
   TFitPanel(const TFitPanel &) = delete;

   /// Disable assignment.
   TFitPanel &operator=(const TFitPanel &) = delete;

   void ProcessData(unsigned connid, const std::string &arg);

public:

   TFitPanel(const std::string &title = "Fit panel") : fTitle(title) {}

   virtual ~TFitPanel() { printf("Fit panel destructor!!!\n"); }

   // method required when any panel want to be inserted into the TCanvas
   std::shared_ptr<TWebWindow> GetWindow();

   void Show(const std::string &where = "");

   void Hide();

   void UseCanvas(std::shared_ptr<TCanvas> &canv);

   void DoFit(const std::string &dname, const std::string &mname);

};

} // namespace Experimental
} // namespace ROOT

#endif
