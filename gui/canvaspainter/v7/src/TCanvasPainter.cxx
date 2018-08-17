/// \file TCanvasPainter.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RVirtualCanvasPainter.hxx"
#include "ROOT/RCanvas.hxx"
#include <ROOT/TLogger.hxx>
#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RPadDisplayItem.hxx>
#include <ROOT/RMenuItem.hxx>

#include <ROOT/TWebWindow.hxx>
#include <ROOT/TWebWindowsManager.hxx>

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <thread>
#include <chrono>
#include <fstream>

#include "TList.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"
#include "TBase64.h"

// ==========================================================================================================

// new implementation of canvas painter, using TWebWindow

namespace ROOT {
namespace Experimental {

class TCanvasPainter : public Internal::RVirtualCanvasPainter {
private:
   struct WebConn {
      unsigned fConnId{0};    ///<! connection id
      bool fDrawReady{false}; ///!< when first drawing is performed
      std::string fGetMenu;   ///<! object id for menu request
      uint64_t fSend{0};      ///<! indicates version send to connection
      uint64_t fDelivered{0}; ///<! indicates version confirmed from canvas
      WebConn() = default;
   };

   struct WebCommand {
      std::string fId;                     ///<! command identifier
      std::string fName;                   ///<! command name
      std::string fArg;                    ///<! command arg
      bool fRunning{false};                ///<! true when command submitted
      CanvasCallback_t fCallback{nullptr}; ///<! callback function associated with command
      unsigned fConnId{0};                 ///<! connection id was used to send command
      WebCommand() = default;
      void CallBack(bool res)
      {
         if (fCallback) fCallback(res);
         fCallback = nullptr;
      }
   };

   struct WebUpdate {
      uint64_t fVersion{0};                ///<! canvas version
      CanvasCallback_t fCallback{nullptr}; ///<! callback function associated with the update
      WebUpdate() = default;
      void CallBack(bool res)
      {
         if (fCallback) fCallback(res);
         fCallback = nullptr;
      }
   };

   typedef std::list<WebConn> WebConnList;

   typedef std::list<WebCommand> WebCommandsList;

   typedef std::list<WebUpdate> WebUpdatesList;

   typedef std::vector<ROOT::Experimental::Detail::RMenuItem> MenuItemsVector;

   /// The canvas we are painting. It might go out of existence while painting.
   const RCanvas &fCanvas; ///<!  Canvas

   std::shared_ptr<TWebWindow> fWindow; ///!< configured display

   WebConnList fWebConn;    ///<! connections list
   bool fHadWebConn{false}; ///<! true if any connection were existing
   // RPadDisplayItem fDisplayList;   ///<! full list of items to display
   // std::string fCurrentDrawableId; ///<! id of drawable, which paint method is called
   WebCommandsList fCmds;     ///<! list of submitted commands
   uint64_t fCmdsCnt{0};      ///<! commands counter
   std::string fWaitingCmdId; ///<! command id waited for completion

   uint64_t fSnapshotVersion{0};   ///<! version of snapshot
   std::string fSnapshot;          ///<! last produced snapshot
   uint64_t fSnapshotDelivered{0}; ///<! minimal version delivered to all connections
   WebUpdatesList fUpdatesLst;     ///<! list of callbacks for canvas update

   std::string fNextDumpName;     ///<! next filename for dumping JSON

   /// Disable copy construction.
   TCanvasPainter(const TCanvasPainter &) = delete;

   /// Disable assignment.
   TCanvasPainter &operator=(const TCanvasPainter &) = delete;

   void CancelUpdates();

   void CancelCommands(unsigned connid = 0);

   void CheckDataToSend();

   void ProcessData(unsigned connid, const std::string &arg);

   std::string CreateSnapshot(const ROOT::Experimental::RCanvas &can);

   std::shared_ptr<RDrawable> FindDrawable(const ROOT::Experimental::RCanvas &can, const std::string &id);

   void SaveCreatedFile(std::string &reply);

   bool FrontCommandReplied(const std::string &reply);

   void PopFrontCommand(bool result);

   int CheckDeliveredVersion(uint64_t ver, double);

   int CheckWaitingCmd(const std::string &cmdname, double);

public:
   TCanvasPainter(const RCanvas &canv) : fCanvas(canv) {}

   virtual ~TCanvasPainter();

   //   virtual void AddDisplayItem(std::unique_ptr<RDisplayItem> &&item) override
   //   {
   //      item->SetObjectID(fCurrentDrawableId);
   //      fDisplayList.Add(std::move(item));
   //   }

   virtual void CanvasUpdated(uint64_t ver, bool async, ROOT::Experimental::CanvasCallback_t callback) override;

   /// return true if canvas modified since last painting
   virtual bool IsCanvasModified(uint64_t id) const override { return fSnapshotDelivered != id; }

   /// perform special action when drawing is ready
   virtual void
   DoWhenReady(const std::string &name, const std::string &arg, bool async, CanvasCallback_t callback) override;

   virtual void NewDisplay(const std::string &where) override;

   virtual int NumDisplays() const override;

   virtual void Run(double tm = 0.) override;

   virtual bool AddPanel(std::shared_ptr<TWebWindow>) override;

   /** \class CanvasPainterGenerator
          Creates TCanvasPainter objects.
        */

   class GeneratorImpl : public Generator {
   public:
      /// Create a new TCanvasPainter to paint the given RCanvas.
      std::unique_ptr<RVirtualCanvasPainter> Create(const ROOT::Experimental::RCanvas &canv) const override
      {
         return std::make_unique<TCanvasPainter>(canv);
      }
      ~GeneratorImpl() = default;

      /// Set RVirtualCanvasPainter::fgGenerator to a new GeneratorImpl object.
      static void SetGlobalPainter()
      {
         if (GetGenerator()) {
            R__ERROR_HERE("CanvasPainter") << "Generator is already set! Skipping second initialization.";
            return;
         }
         GetGenerator().reset(new GeneratorImpl());
      }

      /// Release the GeneratorImpl object.
      static void ResetGlobalPainter() { GetGenerator().reset(); }
   };
};

struct TNewCanvasPainterReg {
   TNewCanvasPainterReg() { TCanvasPainter::GeneratorImpl::SetGlobalPainter(); }
   ~TNewCanvasPainterReg() { TCanvasPainter::GeneratorImpl::ResetGlobalPainter(); }
} newCanvasPainterReg;

} // namespace Experimental
} // namespace ROOT

/////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::TCanvasPainter::~TCanvasPainter()
{
   CancelCommands();
   CancelUpdates();
   if (fWindow)
      fWindow->CloseConnections();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Checks if specified version was delivered to all clients
/// Used to wait for such condition

int ROOT::Experimental::TCanvasPainter::CheckDeliveredVersion(uint64_t ver, double)
{
   if (fWebConn.empty() && fHadWebConn)
      return -1;
   if (fSnapshotDelivered >= ver)
      return 1;
   return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Cancel all pending Canvas::Update()

void ROOT::Experimental::TCanvasPainter::CancelUpdates()
{
   fSnapshotDelivered = 0;
   auto iter = fUpdatesLst.begin();
   while (iter != fUpdatesLst.end()) {
      auto curr = iter++;
      curr->fCallback(false);
      fUpdatesLst.erase(curr);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Cancel command execution on provided connection
/// All commands are cancelled, when connid === 0

void ROOT::Experimental::TCanvasPainter::CancelCommands(unsigned connid)
{
   auto iter = fCmds.begin();
   while (iter != fCmds.end()) {
      auto curr = iter++;
      if (!connid || (curr->fConnId == connid)) {
         if (fWaitingCmdId == curr->fId)
            fWaitingCmdId.clear();
         curr->CallBack(false);
         fCmds.erase(curr);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if canvas need to sand data to the clients

void ROOT::Experimental::TCanvasPainter::CheckDataToSend()
{
   uint64_t min_delivered = 0;

   for (auto &&conn : fWebConn) {

      if (conn.fDelivered && (!min_delivered || (min_delivered < conn.fDelivered)))
         min_delivered = conn.fDelivered;

      // check if direct data sending is possible
      if (!fWindow->CanSend(conn.fConnId, true))
         continue;

      TString buf;

      if (conn.fDrawReady && !fCmds.empty() && !fCmds.front().fRunning) {
         WebCommand &cmd = fCmds.front();
         cmd.fRunning = true;
         buf = "CMD:";
         buf.Append(cmd.fId);
         buf.Append(":");
         buf.Append(cmd.fName);
         cmd.fConnId = conn.fConnId;
      } else if (!conn.fGetMenu.empty()) {
         auto drawable = FindDrawable(fCanvas, conn.fGetMenu);

         R__DEBUG_HERE("CanvasPainter") << "Request menu for object " << conn.fGetMenu;

         if (drawable) {

            ROOT::Experimental::RMenuItems items;

            drawable->PopulateMenu(items);

            // FIXME: got problem with std::list<RMenuItem>, can be generic TBufferJSON
            buf = "MENU:";
            buf.Append(conn.fGetMenu);
            buf.Append(":");
            buf.Append(items.ProduceJSON());
         }

         conn.fGetMenu = "";
      } else if (conn.fSend != fSnapshotVersion) {
         // buf = "JSON";
         // buf  += TBufferJSON::ConvertToJSON(Canvas(), 3);

         conn.fSend = fSnapshotVersion;
         buf = "SNAP:";
         buf += TString::ULLtoa(fSnapshotVersion, 10);
         buf += ":";
         buf += fSnapshot;
      }

      if (buf.Length() > 0) {
         // sending of data can be moved into separate thread - not to block user code
         fWindow->Send(conn.fConnId, buf.Data());
      }
   }

   // if there are updates submitted, but all connections disappeared - cancel all updates
   if (fWebConn.empty() && fSnapshotDelivered)
      return CancelUpdates();

   if (fSnapshotDelivered != min_delivered) {
      fSnapshotDelivered = min_delivered;

      auto iter = fUpdatesLst.begin();
      while (iter != fUpdatesLst.end()) {
         auto curr = iter;
         iter++;
         if (curr->fVersion <= fSnapshotDelivered) {
            curr->CallBack(true);
            fUpdatesLst.erase(curr);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Method invoked when canvas should be updated on the client side
/// Depending from delivered status, each client will received new data

void ROOT::Experimental::TCanvasPainter::CanvasUpdated(uint64_t ver, bool async,
                                                       ROOT::Experimental::CanvasCallback_t callback)
{
   if (fWindow)
      fWindow->Sync();

   if (ver && fSnapshotDelivered && (ver <= fSnapshotDelivered)) {
      // if given canvas version was already delivered to clients, can return immediately
      if (callback)
         callback(true);
      return;
   }

   fSnapshotVersion = ver;
   fSnapshot = CreateSnapshot(fCanvas);

   if (!fWindow || !fWindow->IsShown()) {
      if (callback)
         callback(false);
      return;
   }

   CheckDataToSend();

   if (callback) {
      WebUpdate item;
      item.fVersion = ver;
      item.fCallback = callback;
      fUpdatesLst.push_back(item);
   }

   // wait that canvas is painted
   if (!async)
      fWindow->WaitFor([this, ver](double tm) { return CheckDeliveredVersion(ver, tm); });
}

///////////////////////////////////////////////////
/// Used to wait until submitted command executed

int ROOT::Experimental::TCanvasPainter::CheckWaitingCmd(const std::string &cmdname, double)
{
   if (fWebConn.empty() && fHadWebConn)
      return -1;
   if (fWaitingCmdId.empty()) {
      R__DEBUG_HERE("CanvasPainter") << "Waiting for command finished " << cmdname.c_str();
      return 1;
   }
   return 0;
}

//////////////////////////////////////////////////////////////////////////
/// perform special action when drawing is ready

void ROOT::Experimental::TCanvasPainter::DoWhenReady(const std::string &name, const std::string &arg, bool async,
                                                     CanvasCallback_t callback)
{
   if (name == "JSON") {
      fNextDumpName = arg;
      return;
   }

   if (!async && !fWaitingCmdId.empty()) {
      R__ERROR_HERE("CanvasPainter") << "Fail to submit sync command when previous is still awaited - use async";
      async = true;
   }

   if (!fWindow || !fWindow->IsShown()) {
      if (callback)
         callback(false);
      return;
   }

   WebCommand cmd;
   cmd.fId = TString::ULLtoa(++fCmdsCnt, 10);
   cmd.fName = name;
   cmd.fArg = arg;
   cmd.fRunning = false;
   cmd.fCallback = callback;
   fCmds.push_back(cmd);

   if (!async)
      fWaitingCmdId = cmd.fId;

   CheckDataToSend();

   if (async) return;

   int res = fWindow->WaitFor([this, name](double tm) { return CheckWaitingCmd(name, tm); });
   if (!res)
      R__ERROR_HERE("CanvasPainter") << name << " fail with " << arg;
}

//////////////////////////////////////////////////////////////////////////
/// Process data from the client

void ROOT::Experimental::TCanvasPainter::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      // special argument from TWebWindow itself
      // indication that new connection appeared

      WebConn newconn;
      newconn.fConnId = connid;

      fWebConn.push_back(newconn);
      fHadWebConn = true;

      CheckDataToSend();
      return;
   }

   WebConn *conn(0);
   auto iter = fWebConn.begin();
   while (iter != fWebConn.end()) {
      if (iter->fConnId == connid) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (!conn)
      return; // no connection found

   // R__DEBUG_HERE("CanvasPainter") << "from client " << connid << " got data len:" << arg.length() << " val:" <<
   // arg.substr(0,30);

   if (arg == "CONN_CLOSED") {
      // special argument from TWebWindow itself
      // connection is closed

      fWebConn.erase(iter);

      // if there are no other connections - cancel all submitted commands
      CancelCommands(connid);

   } else if (arg.find("READY") == 0) {

   } else if (arg.find("SNAPDONE:") == 0) {
      std::string cdata = arg;
      cdata.erase(0, 9);
      conn->fDrawReady = kTRUE;                       // at least first drawing is performed
      conn->fDelivered = (uint64_t)std::stoll(cdata); // delivered version of the snapshot
   } else if (arg.find("RREADY:") == 0) {
      conn->fDrawReady = kTRUE; // at least first drawing is performed
   } else if (arg.find("GETMENU:") == 0) {
      std::string cdata = arg;
      cdata.erase(0, 8);
      conn->fGetMenu = cdata;
   } else if (arg == "QUIT") {
      // use window manager to correctly terminate http server
      TWebWindowsManager::Instance()->Terminate();
      return;
   } else if (arg == "RELOAD") {
      conn->fSend = 0; // reset send version, causes new data sending
   } else if (arg == "INTERRUPT") {
      gROOT->SetInterrupt();
   } else if (arg.find("REPLY:") == 0) {
      std::string cdata = arg;
      cdata.erase(0, 6);
      const char *sid = cdata.c_str();
      const char *separ = strchr(sid, ':');
      std::string id;
      if (separ)
         id.append(sid, separ - sid);
      if (fCmds.empty()) {
         R__ERROR_HERE("CanvasPainter") << "Get REPLY without command";
      } else if (!fCmds.front().fRunning) {
         R__ERROR_HERE("CanvasPainter") << "Front command is not running when get reply";
      } else if (fCmds.front().fId != id) {
         R__ERROR_HERE("CanvasPainter") << "Mismatch with front command and ID in REPLY";
      } else {
         bool res = FrontCommandReplied(separ + 1);
         PopFrontCommand(res);
      }
   } else if (arg.find("SAVE:") == 0) {
      std::string cdata = arg;
      cdata.erase(0, 5);
      SaveCreatedFile(cdata);
   } else if (arg.find("OBJEXEC:") == 0) {
      std::string cdata = arg;
      cdata.erase(0, 8);
      size_t pos = cdata.find(':');

      if ((pos != std::string::npos) && (pos > 0)) {
         std::string id(cdata, 0, pos);
         cdata.erase(0, pos + 1);
         auto drawable = FindDrawable(fCanvas, id);
         if (drawable && (cdata.length() > 0)) {
            R__DEBUG_HERE("CanvasPainter") << "execute " << cdata << " for drawable " << id;
            drawable->Execute(cdata);
         } else if (id == "canvas") {
            R__DEBUG_HERE("CanvasPainter") << "execute " << cdata << " for canvas itself (ignored)";
         }
      }
   } else {
      R__ERROR_HERE("CanvasPainter") << "Got not recognized reply" << arg;
   }

   CheckDataToSend();
}

//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// See ROOT::Experimental::TWebWindowsManager::Show() docu for more info

void ROOT::Experimental::TCanvasPainter::NewDisplay(const std::string &where)
{
   std::string showarg = where;
   bool batch_mode = false;
   if (showarg == "batch_canvas") {
      batch_mode = true;
      showarg.clear();
   }

   if (!fWindow) {
      fWindow = TWebWindowsManager::Instance()->CreateWindow(batch_mode);
      fWindow->SetConnLimit(0); // allow any number of connections
      fWindow->SetDefaultPage("file:$jsrootsys/files/canvas.htm");
      fWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); });
      // fWindow->SetGeometry(500,300);
   }

   fWindow->Show(showarg);
}

//////////////////////////////////////////////////////////////////////////
/// Returns number of connected displays

int ROOT::Experimental::TCanvasPainter::NumDisplays() const
{
   if (!fWindow) return 0;

   return fWindow->NumConnections();
}


//////////////////////////////////////////////////////////////////////////
/// Add window as panel inside canvas window

bool ROOT::Experimental::TCanvasPainter::AddPanel(std::shared_ptr<TWebWindow> win)
{
   if (!fWindow) {
      R__ERROR_HERE("CanvasPainter") << "Canvas not yet shown in AddPanel";
      return false;
   }

   if (fWindow->IsBatchMode()) {
      R__ERROR_HERE("CanvasPainter") << "Canvas shown in batch mode when calling AddPanel";
      return false;
   }

   std::string addr = fWindow->RelativeAddr(win);

   if (addr.length() == 0) {
      R__ERROR_HERE("CanvasPainter") << "Cannot attach panel to canvas";
      return false;
   }

   // connection is assigned, but can be refused by the client later
   // therefore handle may be removed later

   std::string cmd("ADDPANEL:");
   cmd.append(addr);

   /// one could use async mode
   DoWhenReady(cmd, "AddPanel", true, nullptr);

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Create JSON representation of data, which should be send to the clients
/// Here server-side painting is performed - each drawable adds own elements in
/// so-called display list, which transferred to the clients

std::string ROOT::Experimental::TCanvasPainter::CreateSnapshot(const ROOT::Experimental::RCanvas &can)
{

   PaintDrawables(can);

   fPadDisplayItem->SetObjectID("canvas"); // for canvas itself use special id
   fPadDisplayItem->SetTitle(can.GetTitle());

   TString res = TBufferJSON::ToJSON(fPadDisplayItem.get(), 23);

   if (!fNextDumpName.empty()) {
      TBufferJSON::ExportToFile(fNextDumpName.c_str(), fPadDisplayItem.get(),
         gROOT->GetClass("ROOT::Experimental::RPadDisplayItem"));
      fNextDumpName.clear();
   }

   fPadDisplayItem.reset(); // no need to keep memory any longer

   return std::string(res.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Find drawable in the canvas with specified id
/// Used to communicate with the clients, which does not have any pointer

std::shared_ptr<ROOT::Experimental::RDrawable>
ROOT::Experimental::TCanvasPainter::FindDrawable(const ROOT::Experimental::RCanvas &can, const std::string &id)
{
   std::string search = id;
   size_t pos = search.find("#");
   // exclude extra specifier, later can be used for menu and commands execution
   if (pos != std::string::npos)
      search.resize(pos);

   return can.FindDrawable(search);
}

////////////////////////////////////////////////////////////////////////////////
/// Method called when GUI sends file to save on local disk
/// File coded with base64 coding

void ROOT::Experimental::TCanvasPainter::SaveCreatedFile(std::string &reply)
{
   size_t pos = reply.find(":");
   if ((pos == std::string::npos) || (pos == 0)) {
      R__ERROR_HERE("CanvasPainter") << "SaveCreatedFile does not found ':' separator";
      return;
   }

   std::string fname(reply, 0, pos);
   reply.erase(0, pos + 1);

   TString binary = TBase64::Decode(reply.c_str());

   std::ofstream ofs(fname, std::ios::binary);
   ofs.write(binary.Data(), binary.Length());
   ofs.close();

   R__INFO_HERE("CanvasPainter") << " Save file from GUI " << fname << " len " << binary.Length();
}

////////////////////////////////////////////////////////////////////////////////
/// Process reply on the currently active command

bool ROOT::Experimental::TCanvasPainter::FrontCommandReplied(const std::string &reply)
{
   WebCommand &cmd = fCmds.front();

   cmd.fRunning = false;

   bool result = false;

   if ((cmd.fName == "SVG") || (cmd.fName == "PNG") || (cmd.fName == "JPEG")) {
      if (reply.length() == 0) {
         R__ERROR_HERE("CanvasPainter") << "Fail to produce image" << cmd.fArg;
      } else {
         TString content = TBase64::Decode(reply.c_str());
         std::ofstream ofs(cmd.fArg, std::ios::binary);
         ofs.write(content.Data(), content.Length());
         ofs.close();
         R__INFO_HERE("CanvasPainter") << cmd.fName << " create file " << cmd.fArg << " length " << content.Length();
         result = true;
      }
   } else if (cmd.fName.find("ADDPANEL:") == 0) {
      R__DEBUG_HERE("CanvasPainter") << "get reply for ADDPANEL " << reply;
      result = (reply == "true");
   } else {
      R__ERROR_HERE("CanvasPainter") << "Unknown command " << cmd.fName;
   }

   return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Remove front command from the command queue
/// If necessary, configured call-back will be invoked

void ROOT::Experimental::TCanvasPainter::PopFrontCommand(bool result)
{
   if (fCmds.empty())
      return;

   // simple condition, which will be checked in waiting loop
   if (!fWaitingCmdId.empty() && (fWaitingCmdId == fCmds.front().fId))
      fWaitingCmdId.clear();

   fCmds.front().CallBack(result);

   fCmds.pop_front();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Run canvas functionality for specified period of time
/// Required when canvas used not from the main thread

void ROOT::Experimental::TCanvasPainter::Run(double tm)
{
   if (fWindow) {
      fWindow->Run(tm);
   } else if (tm>0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(int(tm*1000)));
   }

}
