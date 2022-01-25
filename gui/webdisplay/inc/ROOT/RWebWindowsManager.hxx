// Author: Sergey Linev <s.linev@gsi.de>
// Date: 2017-10-16
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebWindowsManager
#define ROOT7_RWebWindowsManager

#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <functional>

#include "THttpEngine.h"

#include <ROOT/RWebWindow.hxx>

class THttpServer;
class THttpWSHandler;
class TExec;

namespace ROOT {
namespace Experimental {

/// function signature for catching show calls of arbitrary RWebWindow
/// if returns true, normal show procedure will not be invoked
using WebWindowShowCallback_t = std::function<bool(RWebWindow &, const RWebDisplayArgs &)>;

class RWebWindowsManager {

   friend class RWebWindow;

private:
   std::unique_ptr<THttpServer> fServer;  ///<! central communication with the all used displays
   std::string fAddr;                     ///<! HTTP address of the server
   std::recursive_mutex fMutex;           ///<! main mutex, used for window creations
   unsigned fIdCnt{0};                    ///<! counter for identifiers
   bool fUseHttpThrd{false};              ///<! use special thread for THttpServer
   bool fUseSenderThreads{false};         ///<! use extra threads for sending data from RWebWindow to clients
   float fLaunchTmout{30.};               ///<! timeout in seconds to start browser process, default 30s
   bool fExternalProcessEvents{false};    ///<! indicate that there are external process events engine
   std::unique_ptr<TExec> fAssgnExec;     ///<! special exec to assign thread id via ProcessEvents
   WebWindowShowCallback_t fShowCallback; ///<! function called for each RWebWindow::Show call

   /// Returns true if http server use special thread for requests processing (default off)
   bool IsUseHttpThread() const { return fUseHttpThrd; }

   /// Returns true if extra threads to send data via websockets will be used (default off)
   bool IsUseSenderThreads() const { return fUseSenderThreads; }

   /// Returns timeout for launching new browser process
   float GetLaunchTmout() const { return fLaunchTmout; }

   void Unregister(RWebWindow &win);

   /// Show window in specified location, see Show() method for more details
   unsigned ShowWindow(RWebWindow &win, const RWebDisplayArgs &args);

   int WaitFor(RWebWindow &win, WebWindowWaitFunc_t check, bool timed = false, double tm = -1);

   std::string GetUrl(const RWebWindow &win, bool remote = false);

   bool CreateServer(bool with_http = false);

   void AssignWindowThreadId(RWebWindow &win);

public:
   RWebWindowsManager();

   ~RWebWindowsManager();

   /// Returns THttpServer instance
   THttpServer *GetServer() const { return fServer.get(); }

   /// Returns http address of the server, empty string when not available
   std::string GetServerAddr() const { return fAddr; }

   /// Assign show callback which can catch window showing, used by RBrowser
   void SetShowCallback(WebWindowShowCallback_t func) { fShowCallback = func; }

   static std::shared_ptr<RWebWindowsManager> &Instance();

   std::shared_ptr<RWebWindow> CreateWindow();

   void Terminate();

   static bool IsMainThrd();
   static void AssignMainThrd();
};

} // namespace Experimental
} // namespace ROOT

#endif
