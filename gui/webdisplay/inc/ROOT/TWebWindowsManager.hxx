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
   std::unique_ptr<THttpServer> fServer; ///<!  central communication with the all used displays
   std::string fAddr;                    ///<!  HTTP address of the server
   std::mutex fMutex;                    ///<!  main mutex to protect
   int fMutexBooked{0};                  ///<!  flag indicating that mutex is booked for some long operation
   std::thread::id fBookedThrd;          ///<!  thread where mutex is booked, can be reused
   unsigned fIdCnt{0};                   ///<! counter for identifiers
   // std::list<std::shared_ptr<TWebWindow>> fDisplays;   ///<! list of existing displays (not used at the moment)

   bool CreateHttpServer(bool with_http = false);

   void Unregister(TWebWindow &win);

   std::string GetUrl(TWebWindow &win, bool remote = false);

   bool Show(TWebWindow &win, const std::string &where);

   void HaltClient(const std::string &procid);

   void TestProg(TString &prog, const std::string &nexttry);

   int WaitFor(TWebWindow &win, WebWindowWaitFunc_t check, double tm = -1);

public:
   TWebWindowsManager();

   ~TWebWindowsManager();

   /// Returns THttpServer instance
   THttpServer *GetServer() const { return fServer.get(); }

   static void SetUseHttpThread(bool on = true);
   static void SetUseSenderThreads(bool on = true);
   static bool IsUseSenderThreads();

   static std::shared_ptr<TWebWindowsManager> &Instance();

   static bool IsMainThrd();

   std::shared_ptr<TWebWindow> CreateWindow(bool batch_mode = false);

   void Terminate();
};

} // namespace Experimental
} // namespace ROOT

#endif
