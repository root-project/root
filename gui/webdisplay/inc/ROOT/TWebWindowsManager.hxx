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
// #include <list>

#include "THttpEngine.h"

#include <ROOT/TWebWindow.hxx>

class THttpServer;
class THttpWSHandler;

namespace ROOT {
namespace Experimental {

class TWebWindowsManager {

   friend class TWebWindow;

private:
   std::unique_ptr<THttpServer> fServer; ///<!  central communication with the all used displays
   std::string fAddr;                    ///<!   HTTP address of the server
   // std::list<std::shared_ptr<TWebWindow>> fDisplays;   ///<! list of existing displays (not used at the moment)
   unsigned fIdCnt{0}; ///<! counter for identifiers

   /// Creates http server, if required - with real http engine (civetweb)
   bool CreateHttpServer(bool with_http = false);

   /// Release all references to specified window, called from TWebWindow destructor
   void Unregister(TWebWindow &win);

   /// Show window in specified location, invoked from TWebWindow::Show
   bool Show(TWebWindow &win, const std::string &where);

public:
   /// Default constructor
   TWebWindowsManager();

   /// Destructor
   ~TWebWindowsManager();

   /// Returns central instance, which used by standard ROOT widgets like Canvas or FitPanel
   static std::shared_ptr<TWebWindowsManager> &Instance();

   /// Creates new window
   std::shared_ptr<TWebWindow> CreateWindow(bool batch_mode = false);

   /// Wait until provided function returns non-zero value
   int WaitFor(WebWindowWaitFunc_t check, double tm);

   /// Terminate http server and ROOT application
   void Terminate();
};

} // namespace Experimental
} // namespace ROOT

#endif
