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

/** struct ROOT::Experimental::ComboBoxItem
 * \ingroup webdisplay
 * Descriptor for the openui5 ComboBox, used in FitPanel
 */

struct ComboBoxItem {
   std::string fId;
   std::string fName;
   ComboBoxItem() = default;
   ComboBoxItem(const std::string &id, const std::string &name) : fId(id), fName(name) {}
};

/** struct ROOT::Experimental::TFitPanelModel
 * \ingroup webdisplay
 * Model, used to initialized openui5 FitPanel
 */

struct TFitPanelModel {
   std::vector<ComboBoxItem> fDataNames;
   std::string fSelectDataId;
   std::vector<ComboBoxItem> fModelNames;
   std::string fSelectModelId;
   TFitPanelModel() = default;
};

class TFitPanel {

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! connection id

   std::shared_ptr<TWebWindow> fWindow; ///!< configured display

   std::shared_ptr<TCanvas> fCanvas; ///!< canvas used to display results

   std::shared_ptr<TH1D> fFitHist; ///!< histogram created when fit is performed

   /// Disable copy construction.
   TFitPanel(const TFitPanel &) = delete;

   /// Disable assignment.
   TFitPanel &operator=(const TFitPanel &) = delete;

   /// process data from UI
   void ProcessData(unsigned connid, const std::string &arg);

public:
   /// normal constructor
   TFitPanel(const std::string &title = "Fit panel") : fTitle(title) {}

   /// destructor
   virtual ~TFitPanel() { printf("Fit panel destructor!!!\n"); }

   // method required when any panel want to be inserted into the TCanvas
   std::shared_ptr<TWebWindow> GetWindow();

   /// show FitPanel in specified place
   void Show(const std::string &where = "");

   /// hide FitPanel
   void Hide();

   /// let use canvas to display fit results
   void UseCanvas(std::shared_ptr<TCanvas> &canv);

   /// Dummy function, called when "Fit" button pressed in UI
   void DoFit(const std::string &dname, const std::string &mname);
};

} // namespace Experimental
} // namespace ROOT

#endif
