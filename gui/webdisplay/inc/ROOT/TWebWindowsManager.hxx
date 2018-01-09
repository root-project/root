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

   bool CreateHttpServer(bool with_http = false);

   void Unregister(TWebWindow &win);

   std::string GetUrl(TWebWindow &win, bool remote = false);

   bool Show(TWebWindow &win, const std::string &where);

public:
   TWebWindowsManager();

   ~TWebWindowsManager();

   /// Returns THttpServer instance
   THttpServer *GetServer() const { return fServer.get(); }

   static std::shared_ptr<TWebWindowsManager> &Instance();

   std::shared_ptr<TWebWindow> CreateWindow(bool batch_mode = false);

   int WaitFor(WebWindowWaitFunc_t check, double tm);

   void Terminate();
};

} // namespace Experimental
} // namespace ROOT

#endif
