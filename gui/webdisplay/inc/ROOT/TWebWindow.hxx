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
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <functional>
#include <mutex>
#include <thread>

class THttpCallArg;
class THttpServer;

namespace ROOT {
namespace Experimental {

/// function signature for call-backs from the window clients
/// first argument is connection id, second is received data
using WebWindowDataCallback_t = std::function<void(unsigned, const std::string &)>;

/// function signature for waiting call-backs
/// Such callback used when calling thread need to waits for some special data,
/// but wants to run application event loop
/// As argument, spent time in second will be provided
/// Waiting will be performed until function returns non-zero value
using WebWindowWaitFunc_t = std::function<int(double)>;

class TWebWindowsManager;
class TWebWindowWSHandler;

class TWebWindow {

   friend class TWebWindowsManager;
   friend class TWebWindowWSHandler;

private:
   struct QueueItem {
      int fChID{1};      ///<! channel
      bool fText{true};  ///<! is text data
      std::string fData; ///<! text or binary data
      QueueItem(int chid, bool txt, std::string &&data) : fChID(chid), fText(txt), fData(data) {}
   };

   struct WebConn {
      bool fActive{true};            ///<! flag indicates if connection is active
      unsigned fConnId{0};           ///<! connection id (unique inside the window)
      unsigned fWSId{0};             ///<! websocket id
      std::string fProcId;           ///<! client process identifier (when exists)
      int fReady{0};                 ///<! 0 - not ready, 1..9 - interim, 10 - done
      std::mutex fMutex;             ///<! mutex must be used to protect all following data
      int fRecvCount{0};             ///<! number of received packets, should return back with next sending
      int fSendCredits{0};           ///<! how many send operation can be performed without confirmation from other side
      int fClientCredits{0};         ///<! number of credits received from client
      bool fDoingSend{false};        ///<! true when performing send operation
      std::queue<QueueItem> fQueue;  ///<! output queue
      // WebWindowDataCallback_t fCallBack; ///<! additional data callback for extra channels
      WebConn() = default;
      WebConn(unsigned id, unsigned wsid) : fConnId(id), fWSId(wsid) {}
   };

   struct DataEntry {
      unsigned fConnId{0};         ///<! connection id
      std::string fData;           ///<! data for given connection
      DataEntry() = default;
      DataEntry(unsigned connid, std::string &&data) : fConnId(connid), fData(data) {}
   };

   std::shared_ptr<TWebWindowsManager> fMgr;        ///<! display manager
   bool fBatchMode{false};                          ///<! batch mode
   std::string fDefaultPage;                        ///<! HTML page (or file name) returned when window URL is opened
   std::string fPanelName;                          ///<! panel name which should be shown in the window
   unsigned fId{0};                                 ///<! unique identifier
   bool fProcessMT{false};                          ///<! if window event processing performed in dedicated thread
   bool fSendMT{false};                             ///<! true is special threads should be used for sending data
   std::shared_ptr<TWebWindowWSHandler> fWSHandler; ///<! specialize websocket handler for all incoming connections
   bool fShown{false};                              ///<! true when window was shown at least once
   unsigned fConnCnt{0};                            ///<! counter of new connections to assign ids
   std::vector<std::shared_ptr<WebConn>> fConn;     ///<! list of all accepted connections
   std::mutex fConnMutex;                           ///<! mutex used to protect connection list
   std::map<std::string, std::string> fKeys;        ///<! list of awaited keys
   unsigned fConnLimit{1};                          ///<! number of allowed active connections
   bool fNativeOnlyConn{false};                     ///<! only native connection are allowed, created by Show() method
   static const unsigned fMaxQueueLength{10};       ///<! maximal number of queue entries
   WebWindowDataCallback_t fDataCallback;           ///<! main callback when data over channel 1 is arrived
   std::thread::id fDataThrdId;                     ///<! thread id where data callback should be invoked
   std::queue<DataEntry> fDataQueue;                ///<! data queue for main callback
   std::mutex fDataMutex;                           ///<! mutex to protect data queue
   unsigned fWidth{0};                              ///<! initial window width when displayed
   unsigned fHeight{0};                             ///<! initial window height when displayed

   /// Set batch mode, used by TWebWindowsManager
   void SetBatchMode(bool mode) { fBatchMode = mode; }

   /// Set window id, used by TWebWindowsManager
   void SetId(unsigned id) { fId = id; }

   void CreateWSHandler();

   bool ProcessWS(THttpCallArg &arg);

   void CompleteWSSend(unsigned wsid);

   std::vector<std::shared_ptr<WebConn>> GetConnections(unsigned connid = 0);

   std::shared_ptr<WebConn> FindConnection(unsigned wsid, bool make_new = false);

   std::shared_ptr<WebConn> RemoveConnection(unsigned wsid);

   std::string _MakeSendHeader(std::shared_ptr<WebConn> &conn, bool txt, const std::string &data, int chid);

   void ProvideData(unsigned connid, std::string &&arg);

   void InvokeCallbacks(bool force = false);

   void SubmitData(unsigned connid, bool txt, std::string &&data, int chid = 1);

   bool CheckDataToSend(std::shared_ptr<WebConn> &conn);

   void CheckDataToSend(bool only_once = false);

   bool HasKey(const std::string &key) const { return fKeys.count(key) > 0 ; }

   void AddKey(const std::string &key, const std::string &procid) { fKeys[key] = procid; }

public:
   TWebWindow();

   ~TWebWindow();

   /// Method returns true if window should run in batch mode - without creating GUI elements
   /// Can be individually set for different windows - not necessary all windows should be batch
   bool IsBatchMode() const { return fBatchMode; }

   /// Returns true if window was shown at least once
   /// It can happen that shown window not yet have connections
   bool IsShown() const { return fShown; }

   /// Returns ID for the window - unique inside window manager
   unsigned GetId() const { return fId; }

   /// Set content of default window HTML page
   /// This page returns when URL address of the window will be requested
   /// Either HTML code or file name in the form "file:/home/user/data/file.htm"
   /// One also can use configure JSROOT location like "file:$jsrootsys/files/canvas.htm"
   void SetDefaultPage(const std::string &page) { fDefaultPage = page; }

   void SetPanelName(const std::string &name);

   /// Set window geometry. Will be applied if supported by used web display (like CEF or Chromium)
   void SetGeometry(unsigned width, unsigned height)
   {
      fWidth = width;
      fHeight = height;
   }

   /////////////////////////////////////////////////////////////////////////
   /// returns configured window width (0 - default)
   /// actual window width can be different
   unsigned GetWidth() const { return fWidth; }

   /////////////////////////////////////////////////////////////////////////
   /// returns configured window height (0 - default)
   unsigned GetHeight() const { return fHeight; }

   /////////////////////////////////////////////////////////////////////////
   /// Configure maximal number of allowed connections - 0 is unlimited
   /// Will not affect already existing connections
   /// Default is 1 - the only client is allowed
   void SetConnLimit(unsigned lmt = 0) { fConnLimit = lmt; }

   /////////////////////////////////////////////////////////////////////////
   /// returns configured connections limit (0 - default)
   unsigned GetConnLimit() const { return fConnLimit; }

   /////////////////////////////////////////////////////////////////////////
   /// configures that only native (own-created) connections are allowed
   void SetNativeOnlyConn(bool on = true) { fNativeOnlyConn = on; }

   /////////////////////////////////////////////////////////////////////////
   /// returns true if only native (own-created) connections are allowed
   bool IsNativeOnlyConn() const { return fNativeOnlyConn; }

   /// Returns current number of active clients connections
   int NumConnections();

   unsigned GetConnectionId(int num = 0);

   void CloseConnections();

   void CloseConnection(unsigned connid);

   std::string GetUrl(bool remote = true);

   THttpServer *GetServer();

   void Sync();

   void Run(double tm = 0.);

   bool Show(const std::string &where = "");

   bool CanSend(unsigned connid, bool direct = true);

   int GetSendQueueLength(unsigned connid);

   void Send(unsigned connid, const std::string &data);

   void SendBinary(unsigned connid, const void *data, std::size_t len);

   void SendBinary(unsigned connid, std::string &&data);

   std::string RelativeAddr(std::shared_ptr<TWebWindow> &win);

   void SetDataCallBack(WebWindowDataCallback_t func);

   int WaitFor(WebWindowWaitFunc_t check);

   int WaitForTimed(WebWindowWaitFunc_t check, double tm = 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
