/// \file ROOT/RFitPanel.hxx
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

#ifndef ROOT7_RFitPanel
#define ROOT7_RFitPanel

#include <ROOT/RWebWindow.hxx>

#include <ROOT/RCanvas.hxx>

#include "ROOT/RHist.hxx"

#include <vector>

namespace ROOT {
namespace Experimental {

/** struct ROOT::Experimental::RFitPanelComboBoxItem
 * \ingroup webdisplay
 * Descriptor for the openui5 ComboBox, used in FitPanel
 */

struct RFitPanelComboBoxItem {
   std::string fId;
   std::string fName;
   RFitPanelComboBoxItem() = default;
   RFitPanelComboBoxItem(const std::string &id, const std::string &name) : fId(id), fName(name) {}
};

/** struct ROOT::Experimental::RFitPanelModel
 * \ingroup webdisplay
 * Model, used to initialized openui5 FitPanel
 */

struct RFitPanelModel {
   std::vector<RFitPanelComboBoxItem> fDataNames;
   std::string fSelectedData;
   std::vector<RFitPanelComboBoxItem> fModelNames;
   std::string fSelectModelId;
   RFitPanelModel() = default;
};

class RFitPanel {

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! connection id

   std::shared_ptr<RWebWindow> fWindow; ///!< configured display

   std::shared_ptr<RCanvas> fCanvas; ///!< canvas used to display results

   std::shared_ptr<RH1D> fFitHist; ///!< histogram created when fit is performed

   /// Disable copy construction.
   RFitPanel(const RFitPanel &) = delete;

   /// Disable assignment.
   RFitPanel &operator=(const RFitPanel &) = delete;

   /// process data from UI
   void ProcessData(unsigned connid, const std::string &arg);

public:
   /// normal constructor
   RFitPanel(const std::string &title = "Fit panel") : fTitle(title) {}

   /// destructor
   virtual ~RFitPanel() {}

   // method required when any panel want to be inserted into the RCanvas
   std::shared_ptr<RWebWindow> GetWindow();

   /// show FitPanel in specified place
   void Show(const std::string &where = "");

   /// hide FitPanel
   void Hide();

   /// let use canvas to display fit results
   void UseCanvas(std::shared_ptr<RCanvas> &canv);

   /// Dummy function, called when "Fit" button pressed in UI
   void DoFit(const std::string &dname, const std::string &mname);
};

} // namespace Experimental
} // namespace ROOT

#endif
