/// \file ROOT/TWebWindow.hxx
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

#ifndef ROOT7_TWebWindow
#define ROOT7_TWebWindow

#include <memory>
#include <list>
#include <string>
#include <functional>

class THttpCallArg;
class THttpWSEngine;

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TWebWindowsManager
  Central handle to open web-based windows like Canvas or FitPanel.
  */

using WebDisplayDataCallback_t = std::function<void(unsigned,const std::string&)>;

class TWebWindowsManager;
class TWebWindowWSHandler;

class TWebWindow {

friend class TWebWindowsManager;
friend class TWebWindowWSHandler;
private:

   struct WebConn {
      THttpWSEngine *fHandle{nullptr};       ///<! websocket handle
      unsigned       fConnId{0};             ///<! connection id (unique inside the window)
      bool           fReady{false};          ///<! when first ready is appears
      int            fRecvCount{0};          ///<! number of received packets, should return back with next sending
      int            fSendCredits{0};        ///<! how many send operation can be performed without confirmation from other side
      int            fClientCredits{0};      ///<! last received information about credits on client side, helps to resubmit credits back to client
      std::list<std::string> fQueue{};       ///<! small output queue for data which should be send via the connection (including channel)
      WebDisplayDataCallback_t fCallBack{};  ///<! additional data callback
      WebConn() = default;
   };

   std::shared_ptr<TWebWindowsManager>  fMgr{};     ///<!  display manager
   unsigned                             fId{0};     ///<!  unique identifier
   TWebWindowWSHandler               *fWSHandler{nullptr};  ///<!  specialize websocket handler for all incoming connections
   unsigned                           fConnCnt{0};  ///<!  counter of new connections to assign ids
   std::list<WebConn>                 fConn{};     ///<! list of all accepted connections
   static const unsigned       fMaxQueueLength{10}; ///<!  maximal number of queue entries
   WebDisplayDataCallback_t        fDataCallback{}; ///<!  main callback when data over channel 1 is arrived

   void SetId(unsigned id) { fId = id; }

   bool ProcessWS(THttpCallArg *arg);

   void SendDataViaConnection(WebConn &conn, int chid, const std::string &data);

   void CheckDataToSend(bool only_once = false);

public:
   TWebWindow() = default;

   ~TWebWindow();

   unsigned GetId() const { return fId; }

   unsigned NumConnections() const { return fConn.size(); }

   bool Show(const std::string &where);

   bool CanSend(unsigned connid, bool direct = true) const;

   void Send(const std::string &data, unsigned connid = 0);

   void SetDataCallBack(WebDisplayDataCallback_t &func) { fDataCallback = func; }


};

} // namespace Experimental
} // namespace ROOT

#endif
