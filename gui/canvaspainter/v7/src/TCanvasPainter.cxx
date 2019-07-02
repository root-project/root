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
#include <ROOT/RLogger.hxx>
#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RPadDisplayItem.hxx>
#include <ROOT/RMenuItem.hxx>

#include <ROOT/RWebWindow.hxx>

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>

#include "TList.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"
#include "TBase64.h"

// ==========================================================================================================

// new implementation of canvas painter, using RWebWindow

namespace ROOT {
namespace Experimental {

class TCanvasPainter : public Internal::RVirtualCanvasPainter {
private:
   struct WebConn {
      unsigned fConnId{0};    ///<! connection id
      std::string fGetMenu;   ///<! object id for menu request
      uint64_t fSend{0};      ///<! indicates version send to connection
      uint64_t fDelivered{0}; ///<! indicates version confirmed from canvas
      WebConn() = default;
      WebConn(unsigned connid) : fConnId(connid) {}
   };

   struct WebCommand {
      std::string fId;                                ///<! command identifier
      std::string fName;                              ///<! command name
      std::string fArg;                               ///<! command arguments
      enum { sInit, sRunning, sReady } fState{sInit}; ///<! true when command submitted
      bool fResult{false};                            ///<! result of command execution
      CanvasCallback_t fCallback{nullptr};            ///<! callback function associated with command
      unsigned fConnId{0};                            ///<! connection id for the command, when 0 specified command will be sumbited to any available connection
      WebCommand() = default;
      WebCommand(const std::string &id, const std::string &name, const std::string &arg, CanvasCallback_t callback,
                 unsigned connid)
         : fId(id), fName(name), fArg(arg), fCallback(callback), fConnId(connid)
      {
      }
      void CallBack(bool res)
      {
         if (fCallback)
            fCallback(res);
         fCallback = nullptr;
      }
   };

   struct WebUpdate {
      uint64_t fVersion{0};                ///<! canvas version
      CanvasCallback_t fCallback{nullptr}; ///<! callback function associated with the update
      WebUpdate() = default;
      WebUpdate(uint64_t ver, CanvasCallback_t callback) : fVersion(ver), fCallback(callback) {}
      void CallBack(bool res)
      {
         if (fCallback)
            fCallback(res);
         fCallback = nullptr;
      }
   };

   typedef std::vector<ROOT::Experimental::Detail::RMenuItem> MenuItemsVector;

   const RCanvas &fCanvas; ///<!  Canvas we are painting, *this will be owned by canvas

   std::shared_ptr<RWebWindow> fWindow; ///!< configured display

   std::list<WebConn> fWebConn;                  ///<! connections list
   std::list<std::shared_ptr<WebCommand>> fCmds; ///<! list of submitted commands
   uint64_t fCmdsCnt{0};                         ///<! commands counter

   uint64_t fSnapshotVersion{0};     ///<! version of snapshot
   std::string fSnapshot;            ///<! last produced snapshot
   uint64_t fSnapshotDelivered{0};   ///<! minimal version delivered to all connections
   std::list<WebUpdate> fUpdatesLst; ///<! list of callbacks for canvas update

   std::string fNextDumpName;     ///<! next filename for dumping JSON
   int fJsonComp{23};             ///<! json compression for data send to client

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

   void CreateWindow();

   void SaveCreatedFile(std::string &reply);

   void FrontCommandReplied(const std::string &reply);

public:
   TCanvasPainter(const RCanvas &canv);

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

   virtual bool AddPanel(std::shared_ptr<RWebWindow>) override;

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
/// constructor

ROOT::Experimental::TCanvasPainter::TCanvasPainter(const RCanvas &canv) : fCanvas(canv)
{
   auto comp = gEnv->GetValue("WebGui.JsonComp", -1);
   if (comp >= 0) fJsonComp = comp;
}

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
/// Cancel all pending Canvas::Update()

void ROOT::Experimental::TCanvasPainter::CancelUpdates()
{
   fSnapshotDelivered = 0;
   for (auto &item: fUpdatesLst)
      item.fCallback(false);
   fUpdatesLst.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Cancel command execution on provided connection
/// All commands are cancelled, when connid === 0

void ROOT::Experimental::TCanvasPainter::CancelCommands(unsigned connid)
{
   std::list<std::shared_ptr<WebCommand>> remainingCmds;

   for (auto &&cmd : fCmds) {
      if (!connid || (cmd->fConnId == connid)) {
         cmd->CallBack(false);
         cmd->fState = WebCommand::sReady;
      } else {
         remainingCmds.emplace_back(std::move(cmd));
      }
   }
   swap(fCmds, remainingCmds);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if canvas need to sand data to the clients

void ROOT::Experimental::TCanvasPainter::CheckDataToSend()
{
   uint64_t min_delivered = 0;

   for (auto &conn : fWebConn) {

      if (conn.fDelivered && (!min_delivered || (min_delivered < conn.fDelivered)))
         min_delivered = conn.fDelivered;

      // check if direct data sending is possible
      if (!fWindow->CanSend(conn.fConnId, true))
         continue;

      TString buf;

      if (conn.fDelivered && !fCmds.empty() && (fCmds.front()->fState == WebCommand::sInit) &&
          ((fCmds.front()->fConnId == 0) || (fCmds.front()->fConnId == conn.fConnId))) {
         auto &cmd = fCmds.front();
         cmd->fState = WebCommand::sRunning;
         cmd->fConnId = conn.fConnId; // assign command to the connection
         buf = "CMD:";
         buf.Append(cmd->fId);
         buf.Append(":");
         buf.Append(cmd->fName);
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
      } else if ((conn.fSend != fSnapshotVersion) && (conn.fDelivered == conn.fSend)) {
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

      if (fUpdatesLst.size() > 0)
         fUpdatesLst.erase(std::remove_if(fUpdatesLst.begin(), fUpdatesLst.end(), [this](WebUpdate &item) {
            if (item.fVersion > fSnapshotDelivered)
               return false;
            item.CallBack(true);
            return true;
         }));
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

   if (!fWindow || !fWindow->HasConnection(0, false)) {
      if (callback)
         callback(false);
      return;
   }

   CheckDataToSend();

   if (callback)
      fUpdatesLst.emplace_back(ver, callback);

   // wait that canvas is painted
   if (!async) {
      fWindow->WaitForTimed([this, ver](double) {

         if (fSnapshotDelivered >= ver)
            return 1;

         // all connections are gone
         if (fWebConn.empty() && !fWindow->HasConnection(0, false))
            return -2;

         // time is not important - timeout handle before
         // if (tm > 100) return -3;

         // continue waiting
         return 0;
      });
   }
}

//////////////////////////////////////////////////////////////////////////
/// perform special action when drawing is ready

void ROOT::Experimental::TCanvasPainter::DoWhenReady(const std::string &name, const std::string &arg, bool async,
                                                     CanvasCallback_t callback)
{
   if (name == "JSON") {
      // it is only for debugging, JSON does not invoke callback
      fNextDumpName = arg;
      return;
   }

   // ensure that window exists
   CreateWindow();

   unsigned connid = 0;

   if (arg == "AddPanel") {
      // take first connection to add panel
      connid = fWindow->GetConnectionId();
   } else {
      // create batch job to execute action
      connid = fWindow->MakeBatch();
   }

   if (!connid) {
      if (callback)
         callback(false);
      return;
   }

   auto cmd = std::make_shared<WebCommand>(std::to_string(++fCmdsCnt), name, arg, callback, connid);
   fCmds.emplace_back(cmd);

   CheckDataToSend();

   if (async) return;

   int res = fWindow->WaitForTimed([this, cmd](double) {
      if (cmd->fState == WebCommand::sReady) {
         R__DEBUG_HERE("CanvasPainter") << "Command " << cmd->fName << " done";
         return cmd->fResult ? 1 : -1;
      }

      // connection is gone
      if (!fWindow->HasConnection(cmd->fConnId, false))
         return -2;

      // time is not important - timeout handle before
      // if (tm > 100.) return -3;

      return 0;
   });

   if (res <= 0)
      R__ERROR_HERE("CanvasPainter") << name << " fail with " << arg << " result = " << res;
}


//////////////////////////////////////////////////////////////////////////
/// Process data from the client

void ROOT::Experimental::TCanvasPainter::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      // special argument from RWebWindow itself
      // indication that new connection appeared

      fWebConn.emplace_back(connid);

      CheckDataToSend();
      return;
   }

   auto check_header = [arg](const std::string &header) {
      return arg.compare(0, header.length(), header) == 0;
   };

   auto conn =
      std::find_if(fWebConn.begin(), fWebConn.end(), [connid](WebConn &item) { return item.fConnId == connid; });

   if (conn == fWebConn.end())
      return; // no connection found

   // R__DEBUG_HERE("CanvasPainter") << "from client " << connid << " got data len:" << arg.length() << " val:" <<
   // arg.substr(0,30);

   if (arg == "CONN_CLOSED") {
      // special argument from RWebWindow itself
      // connection is closed

      fWebConn.erase(conn);

      // if there are no other connections - cancel all submitted commands
      CancelCommands(connid);

   } else if (check_header("READY")) {

   } else if (check_header("SNAPDONE:")) {
      std::string cdata = arg;
      cdata.erase(0, 9);
      conn->fDelivered = (uint64_t)std::stoll(cdata); // delivered version of the snapshot
   } else if (check_header("GETMENU:")) {
      std::string cdata = arg;
      cdata.erase(0, 8);
      conn->fGetMenu = cdata;
   } else if (arg == "QUIT") {
      // use window manager to correctly terminate http server and ROOT session
      fWindow->TerminateROOT();
      return;
   } else if (arg == "RELOAD") {
      conn->fSend = 0; // reset send version, causes new data sending
   } else if (arg == "INTERRUPT") {
      gROOT->SetInterrupt();
   } else if (check_header("REPLY:")) {
      std::string cdata = arg;
      cdata.erase(0, 6);
      const char *sid = cdata.c_str();
      const char *separ = strchr(sid, ':');
      std::string id;
      if (separ)
         id.append(sid, separ - sid);
      if (fCmds.empty()) {
         R__ERROR_HERE("CanvasPainter") << "Get REPLY without command";
      } else if (fCmds.front()->fState != WebCommand::sRunning) {
         R__ERROR_HERE("CanvasPainter") << "Front command is not running when get reply";
      } else if (fCmds.front()->fId != id) {
         R__ERROR_HERE("CanvasPainter") << "Mismatch with front command and ID in REPLY";
      } else {
         FrontCommandReplied(separ + 1);
      }
   } else if (check_header("SAVE:")) {
      std::string cdata = arg;
      cdata.erase(0, 5);
      SaveCreatedFile(cdata);
   } else if (check_header("OBJEXEC:")) {
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

void ROOT::Experimental::TCanvasPainter::CreateWindow()
{
   if (fWindow) return;

   fWindow = RWebWindow::Create();
   fWindow->SetConnLimit(0); // allow any number of connections
   fWindow->SetDefaultPage("file:rootui5sys/canv/canvas.html");
   fWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); });
   // fWindow->SetGeometry(500,300);
}


//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// See ROOT::Experimental::RWebWindowsManager::Show() docu for more info

void ROOT::Experimental::TCanvasPainter::NewDisplay(const std::string &where)
{
   CreateWindow();

   fWindow->Show(where);
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

bool ROOT::Experimental::TCanvasPainter::AddPanel(std::shared_ptr<RWebWindow> win)
{
   if (gROOT->IsWebDisplayBatch())
      return false;

   if (!fWindow) {
      R__ERROR_HERE("CanvasPainter") << "Canvas not yet shown in AddPanel";
      return false;
   }

   if (!fWindow->IsShown()) {
      R__ERROR_HERE("CanvasPainter") << "Canvas window was not shown to call AddPanel";
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
   fPadDisplayItem->SetWindowSize(can.GetSize());

   TString res = TBufferJSON::ToJSON(fPadDisplayItem.get(), fJsonComp);

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

void ROOT::Experimental::TCanvasPainter::FrontCommandReplied(const std::string &reply)
{
   auto cmd = fCmds.front();
   fCmds.pop_front();

   cmd->fState = WebCommand::sReady;

   bool result = false;

   if ((cmd->fName == "SVG") || (cmd->fName == "PNG") || (cmd->fName == "JPEG")) {
      if (reply.length() == 0) {
         R__ERROR_HERE("CanvasPainter") << "Fail to produce image" << cmd->fArg;
      } else {
         TString content = TBase64::Decode(reply.c_str());
         std::ofstream ofs(cmd->fArg, std::ios::binary);
         ofs.write(content.Data(), content.Length());
         ofs.close();
         R__INFO_HERE("CanvasPainter") << cmd->fName << " create file " << cmd->fArg << " length " << content.Length();
         result = true;
      }
   } else if (cmd->fName.find("ADDPANEL:") == 0) {
      R__DEBUG_HERE("CanvasPainter") << "get reply for ADDPANEL " << reply;
      result = (reply == "true");
   } else {
      R__ERROR_HERE("CanvasPainter") << "Unknown command " << cmd->fName;
   }

   cmd->fResult = result;
   cmd->CallBack(result);
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
