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

namespace ROOT {
namespace Experimental {

class TFitPanel  {

   std::string   fTitle{};                 ///<! title
   unsigned fConnId{0};                  ///<! connection id

   std::shared_ptr<TWebWindow>  fWindow{}; ///!< configured display

   /// Disable copy construction.
   TFitPanel(const TFitPanel &) = delete;

   /// Disable assignment.
   TFitPanel &operator=(const TFitPanel &) = delete;

   void ProcessData(unsigned connid, const std::string &arg);

public:

   TFitPanel(const std::string &title = "Fit panel") : fTitle(title) {}

   virtual ~TFitPanel() { printf("Fit panel destructor!!!\n"); }

   void Show(const std::string &where = "");

   void Hide();

};

} // namespace Experimental
} // namespace ROOT

#endif
