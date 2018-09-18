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
#include <thread>
#include <mutex>

#include "THttpEngine.h"

#include <ROOT/TWebWindow.hxx>

class THttpServer;
class THttpWSHandler;

namespace ROOT {
namespace Experimental {

class TWebWindowManagerGuard;

class TWebWindowsManager {

   friend class TWebWindow;
   friend class TWebWindowManagerGuard;

private:
   std::unique_ptr<THttpServer> fServer; ///<! central communication with the all used displays
   std::string fAddr;                    ///<! HTTP address of the server
   std::recursive_mutex fMutex;          ///<! main mutex, used for window creations
   unsigned fIdCnt{0};                   ///<! counter for identifiers
   bool fUseHttpThrd{false};             ///<! use special thread for THttpServer
   bool fUseSenderThreads{false};        ///<! use extra threads for sending data from RWebWindow to clients
   float fLaunchTmout{30.};              ///<! timeout in seconds to start browser process, default 30s

   /// Returns true if http server use special thread for requests processing (default off)
   bool IsUseHttpThread() const { return fUseHttpThrd; }

   /// Returns true if extra threads to send data via websockets will be used (default off)
   bool IsUseSenderThreads() const { return fUseSenderThreads; }

   /// Returns timeout for launching new browser process
   float GetLaunchTmout() const { return fLaunchTmout; }

   bool CreateHttpServer(bool with_http = false);

   void Unregister(TWebWindow &win);

   std::string GetUrl(TWebWindow &win, bool batch_mode, bool remote = false);

   unsigned Show(TWebWindow &win, bool batch_mode, const std::string &where);

   void HaltClient(const std::string &procid);

   void TestProg(TString &prog, const std::string &nexttry);

   int WaitFor(TWebWindow &win, WebWindowWaitFunc_t check, bool timed = false, double tm = -1);

   static bool IsMainThrd();

public:
   TWebWindowsManager();

   ~TWebWindowsManager();

   /// Returns THttpServer instance
   THttpServer *GetServer() const { return fServer.get(); }

   static std::shared_ptr<TWebWindowsManager> &Instance();

   std::shared_ptr<TWebWindow> CreateWindow();

   void Terminate();
};

} // namespace Experimental
} // namespace ROOT

#endif
