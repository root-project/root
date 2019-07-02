/// \file ROOT/RWebWindow.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebWindow
#define ROOT7_RWebWindow

#include <ROOT/RWebDisplayHandle.hxx>

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <functional>
#include <mutex>
#include <thread>
#include <chrono>

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

class RWebWindowsManager;
class RWebWindowWSHandler;

class RWebWindow {

   friend class RWebWindowsManager;
   friend class RWebWindowWSHandler;
   friend class RWebDisplayHandle;

private:
   using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

   struct QueueItem {
      int fChID{1};      ///<! channel
      bool fText{true};  ///<! is text data
      std::string fData; ///<! text or binary data
      QueueItem(int chid, bool txt, std::string &&data) : fChID(chid), fText(txt), fData(data) {}
   };

   struct WebConn {
      unsigned fConnId{0};                 ///<! connection id (unique inside the window)
      bool fBatchMode{false};              ///<! indicate if connection represent batch job
      std::string fKey;                    ///<! key value supplied to the window (when exists)
      std::unique_ptr<RWebDisplayHandle> fDisplayHandle;  ///<! handle assigned with started web display (when exists)
      std::shared_ptr<THttpCallArg> fHold; ///<! request used to hold headless browser
      timestamp_t fSendStamp;              ///<! last server operation, always used from window thread
      bool fActive{false};                 ///<! flag indicates if connection is active
      unsigned fWSId{0};                   ///<! websocket id
      int fReady{0};                       ///<! 0 - not ready, 1..9 - interim, 10 - done
      std::mutex fMutex;                   ///<! mutex must be used to protect all following data
      timestamp_t fRecvStamp;              ///<! last receive operation, protected with connection mutex
      int fRecvCount{0};                   ///<! number of received packets, should return back with next sending
      int fSendCredits{0};                 ///<! how many send operation can be performed without confirmation from other side
      int fClientCredits{0};               ///<! number of credits received from client
      bool fDoingSend{false};              ///<! true when performing send operation
      std::queue<QueueItem> fQueue;        ///<! output queue
      WebConn() = default;
      WebConn(unsigned connid) : fConnId(connid) {}
      WebConn(unsigned connid, unsigned wsid) : fConnId(connid), fActive(true), fWSId(wsid) {}
      WebConn(unsigned connid, bool batch_mode, const std::string &key)
         : fConnId(connid), fBatchMode(batch_mode), fKey(key)
      {
         ResetStamps();
      }
      ~WebConn();

      void ResetStamps() { fSendStamp = fRecvStamp = std::chrono::system_clock::now(); }
   };

   struct DataEntry {
      unsigned fConnId{0};         ///<! connection id
      std::string fData;           ///<! data for given connection
      DataEntry() = default;
      DataEntry(unsigned connid, std::string &&data) : fConnId(connid), fData(data) {}
   };

   typedef std::vector<std::shared_ptr<WebConn>> ConnectionsList;

   std::shared_ptr<RWebWindowsManager> fMgr;        ///<! display manager
   std::string fDefaultPage;                        ///<! HTML page (or file name) returned when window URL is opened
   std::string fPanelName;                          ///<! panel name which should be shown in the window
   unsigned fId{0};                                 ///<! unique identifier
   bool fProcessMT{false};                          ///<! if window event processing performed in dedicated thread
   bool fSendMT{false};                             ///<! true is special threads should be used for sending data
   std::shared_ptr<RWebWindowWSHandler> fWSHandler; ///<! specialize websocket handler for all incoming connections
   unsigned fConnCnt{0};                            ///<! counter of new connections to assign ids
   ConnectionsList fPendingConn;                    ///<! list of pending connection with pre-assigned keys
   ConnectionsList fConn;                           ///<! list of all accepted connections
   std::mutex fConnMutex;                           ///<! mutex used to protect connection list
   unsigned fConnLimit{1};                          ///<! number of allowed active connections
   bool fNativeOnlyConn{false};                     ///<! only native connection are allowed, created by Show() method
   unsigned fMaxQueueLength{10};                    ///<! maximal number of queue entries
   WebWindowDataCallback_t fDataCallback;           ///<! main callback when data over channel 1 is arrived
   std::thread::id fDataThrdId;                     ///<! thread id where data callback should be invoked
   std::queue<DataEntry> fDataQueue;                ///<! data queue for main callback
   std::mutex fDataMutex;                           ///<! mutex to protect data queue
   unsigned fWidth{0};                              ///<! initial window width when displayed
   unsigned fHeight{0};                             ///<! initial window height when displayed
   float fOperationTmout{50.};                      ///<! timeout in seconds to perform synchronous operation, default 50s
   std::string fProtocolFileName;                   ///<! local file where communication protocol will be written
   int fProtocolCnt{-1};                            ///<! counter for protocol recording
   unsigned fProtocolConnId{0};                     ///<! connection id, which is used for writing protocol
   std::string fProtocolPrefix;                     ///<! prefix for created files names
   std::string fProtocol;                           ///<! protocol

   std::shared_ptr<RWebWindowWSHandler> CreateWSHandler(std::shared_ptr<RWebWindowsManager> mgr, unsigned id, double tmout);

   bool ProcessWS(THttpCallArg &arg);

   void CompleteWSSend(unsigned wsid);

   ConnectionsList GetConnections(unsigned connid = 0);

   std::shared_ptr<WebConn> FindOrCreateConnection(unsigned wsid, bool make_new, const char *query);

   std::shared_ptr<WebConn> FindConnection(unsigned wsid) { return FindOrCreateConnection(wsid, false, nullptr); }

   std::shared_ptr<WebConn> RemoveConnection(unsigned wsid);

   std::string _MakeSendHeader(std::shared_ptr<WebConn> &conn, bool txt, const std::string &data, int chid);

   void ProvideData(unsigned connid, std::string &&arg);

   void InvokeCallbacks(bool force = false);

   void SubmitData(unsigned connid, bool txt, std::string &&data, int chid = 1);

   bool CheckDataToSend(std::shared_ptr<WebConn> &conn);

   void CheckDataToSend(bool only_once = false);

   bool HasKey(const std::string &key);

   void CheckPendingConnections();

   void CheckInactiveConnections();

   unsigned AddDisplayHandle(bool batch_mode, const std::string &key, std::unique_ptr<RWebDisplayHandle> &handle);

   bool ProcessBatchHolder(std::shared_ptr<THttpCallArg> &arg);

public:

   RWebWindow();

   ~RWebWindow();

   /// Returns ID for the window - unique inside window manager
   unsigned GetId() const { return fId; }

   /// Set content of default window HTML page
   /// This page returns when URL address of the window will be requested
   /// Either HTML code or file name in the form "file:/home/user/data/file.htm"
   /// One also can using default locations like "file:rootui5sys/canv/canvas.html"
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
   /// configures maximal queue length of data which can be held by window
   void SetMaxQueueLength(unsigned len = 10) { fMaxQueueLength = len; }

   /////////////////////////////////////////////////////////////////////////
   /// Return maximal queue length of data which can be held by window
   unsigned GetMaxQueueLength() const { return fMaxQueueLength; }

   /////////////////////////////////////////////////////////////////////////
   /// configures that only native (own-created) connections are allowed
   void SetNativeOnlyConn(bool on = true) { fNativeOnlyConn = on; }

   /////////////////////////////////////////////////////////////////////////
   /// returns true if only native (own-created) connections are allowed
   bool IsNativeOnlyConn() const { return fNativeOnlyConn; }

   int NumConnections();

   unsigned GetConnectionId(int num = 0);

   bool HasConnection(unsigned connid = 0, bool only_active = true);

   void CloseConnections();

   void CloseConnection(unsigned connid);

   /// Returns timeout for synchronous WebWindow operations
   float GetOperationTmout() const { return fOperationTmout; }

   /// Set timeout for synchronous WebWindow operations
   void SetOperationTmout(float tm = 50.) { fOperationTmout = tm; }

   std::string GetUrl(bool remote = true);

   THttpServer *GetServer();

   void Sync();

   void Run(double tm = 0.);

   unsigned Show(const RWebDisplayArgs &args = "");

   unsigned GetDisplayConnection();

   /// Returns true when window was shown at least once
   bool IsShown() { return GetDisplayConnection() != 0; }

   unsigned MakeBatch(bool create_new = false, const RWebDisplayArgs &args = "");

   unsigned FindBatch();

   bool CanSend(unsigned connid, bool direct = true);

   int GetSendQueueLength(unsigned connid);

   void Send(unsigned connid, const std::string &data);

   void SendBinary(unsigned connid, const void *data, std::size_t len);

   void SendBinary(unsigned connid, std::string &&data);

   void RecordData(const std::string &fname = "protocol.json", const std::string &fprefix = "");

   std::string RelativeAddr(std::shared_ptr<RWebWindow> &win);

   void SetDataCallBack(WebWindowDataCallback_t func);

   int WaitFor(WebWindowWaitFunc_t check);

   int WaitForTimed(WebWindowWaitFunc_t check);

   int WaitForTimed(WebWindowWaitFunc_t check, double duration);

   static std::shared_ptr<RWebWindow> Create();
};

} // namespace Experimental
} // namespace ROOT

#endif
