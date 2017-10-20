/// \file ROOT/TWebWindowsManager.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TWebWindowsManager
#define ROOT7_TWebWindowsManager

#include <memory>
#include <string>
#include <list>

#include "THttpEngine.h"

#include <ROOT/TWebWindow.hxx>

class THttpServer;

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TWebWindowsManager
  Central handle to open web-based windows like Canvas or FitPanel.
  */


class TWebWindowsManager {
private:
   THttpServer    *fServer{0};      ///<!  central communication with the all used displays
   std::string     fAddr{};         ///<!   HTTP address of the server
   std::list<std::shared_ptr<TWebWindow>> fDisplays{}; ///<! list of existing displays
   unsigned                      fIdCnt{0};   ///<! counter for identifiers

   bool CreateHttpServer(bool with_http = false);

public:
   /// Create a temporary TCanvas
   TWebWindowsManager() = default;

   ~TWebWindowsManager();

   /// Returns central instance, which used by standard ROOT widgets like Canvas or FitPanel
   static std::shared_ptr<TWebWindowsManager> &Instance();

   std::shared_ptr<TWebWindow> CreateWindow(bool batch_mode = false);

   void CloseDisplay(TWebWindow *display);

   bool Show(TWebWindow *display, const std::string &where, bool first_time = false);

   bool WaitFor(WebWindowWaitFunc_t check, double tm);
};

} // namespace Experimental
} // namespace ROOT

#endif
