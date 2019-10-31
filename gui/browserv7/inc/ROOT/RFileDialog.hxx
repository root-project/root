/// \file ROOT/RFileDialog.hxx
/// \ingroup rbrowser
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-10-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFileDialog
#define ROOT7_RFileDialog

#include <ROOT/RWebWindow.hxx>
#include <ROOT/RBrowsable.hxx>

#include <vector>
#include <memory>
#include <stdint.h>

namespace ROOT {
namespace Experimental {

/** Web-based FileDialog */

class RFileDialog {

protected:

   std::string fTitle;  ///<! title
   unsigned fConnId{0}; ///<! default connection id

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window for file dialog

   std::string fWorkingDirectory;            ///<! top working directory used by Browsable
   RBrowsable  fBrowsable;                   ///<! central browsing element

   std::string ProcessBrowserRequest(const std::string &msg);
   std::string GetCurrentWorkingDirectory();

   void SendInitMsg(unsigned connid);
   void WebWindowCallback(unsigned connid, const std::string &arg);

   void Show(const RWebDisplayArgs &args = "");
   void Hide();

public:
   RFileDialog();
   virtual ~RFileDialog();

};

} // namespace Experimental
} // namespace ROOT

#endif
