/// \file RCanvasPainter.cxx
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
#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RWebWindow.hxx>

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <queue>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <regex>

#include "TList.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"
#include "TBase64.h"
#include "TSystem.h"

using namespace std::string_literals;

// ==========================================================================================================

// new implementation of canvas painter, using RWebWindow

namespace ROOT {
namespace Experimental {

class RCanvasPainter : public Internal::RVirtualCanvasPainter {
private:
   struct WebConn {
      unsigned fConnId{0};                 ///<! connection id
      std::string fGetMenu;                ///<! object id for menu request
      std::queue<std::string> fSendQueue;  ///<! send queue for the connection
      RDrawable::Version_t fSend{0};       ///<! indicates version send to connection
      RDrawable::Version_t fDelivered{0};  ///<! indicates version confirmed from canvas
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

   RCanvas &fCanvas; ///<!  Canvas we are painting, *this will be owned by canvas

   std::shared_ptr<RWebWindow> fWindow; ///!< configured display

   std::list<WebConn> fWebConn;                  ///<! connections list
   std::list<std::shared_ptr<WebCommand>> fCmds; ///<! list of submitted commands
   uint64_t fCmdsCnt{0};                         ///<! commands counter

   uint64_t fSnapshotDelivered{0};   ///<! minimal version delivered to all connections
   std::list<WebUpdate> fUpdatesLst; ///<! list of callbacks for canvas update

   int fJsonComp{23};             ///<! json compression for data send to client

   /// Disable copy construction.
   RCanvasPainter(const RCanvasPainter &) = delete;

   /// Disable assignment.
   RCanvasPainter &operator=(const RCanvasPainter &) = delete;

   void CancelUpdates();

   void CancelCommands(unsigned connid = 0);

   void CheckDataToSend();

   void ProcessData(unsigned connid, const std::string &arg);

   std::string CreateSnapshot(RDrawable::Version_t vers);

   std::shared_ptr<RDrawable> FindPrimitive(const RCanvas &can, const std::string &id, const RPadBase **subpad = nullptr);

   void CreateWindow();

   void SaveCreatedFile(std::string &reply);

   void FrontCommandReplied(const std::string &reply);

public:
   RCanvasPainter(RCanvas &canv);

   virtual ~RCanvasPainter();

   void CanvasUpdated(uint64_t ver, bool async, ROOT::Experimental::CanvasCallback_t callback) final;

   /// return true if canvas modified since last painting
   bool IsCanvasModified(uint64_t id) const final { return fSnapshotDelivered != id; }

   /// perform special action when drawing is ready
   void DoWhenReady(const std::string &name, const std::string &arg, bool async, CanvasCallback_t callback) final;

   bool ProduceBatchOutput(const std::string &fname, int width, int height) final;

   void NewDisplay(const std::string &where) final;

   int NumDisplays() const final;

   std::string GetWindowAddr() const final;

   void Run(double tm = 0.) final;

   bool AddPanel(std::shared_ptr<RWebWindow>) final;

   /** \class CanvasPainterGenerator
          Creates RCanvasPainter objects.
        */

   class GeneratorImpl : public Generator {
   public:
      /// Create a new RCanvasPainter to paint the given RCanvas.
      std::unique_ptr<RVirtualCanvasPainter> Create(ROOT::Experimental::RCanvas &canv) const override
      {
         return std::make_unique<RCanvasPainter>(canv);
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
   TNewCanvasPainterReg() { RCanvasPainter::GeneratorImpl::SetGlobalPainter(); }
   ~TNewCanvasPainterReg() { RCanvasPainter::GeneratorImpl::ResetGlobalPainter(); }
} newCanvasPainterReg;

} // namespace Experimental
} // namespace ROOT


/////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

ROOT::Experimental::RCanvasPainter::RCanvasPainter(RCanvas &canv) : fCanvas(canv)
{
   auto comp = gEnv->GetValue("WebGui.JsonComp", -1);
   if (comp >= 0) fJsonComp = comp;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

ROOT::Experimental::RCanvasPainter::~RCanvasPainter()
{
   CancelCommands();
   CancelUpdates();
   if (fWindow)
      fWindow->CloseConnections();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Cancel all pending Canvas::Update()

void ROOT::Experimental::RCanvasPainter::CancelUpdates()
{
   fSnapshotDelivered = 0;
   for (auto &item: fUpdatesLst)
      item.fCallback(false);
   fUpdatesLst.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Cancel command execution on provided connection
/// All commands are cancelled, when connid === 0

void ROOT::Experimental::RCanvasPainter::CancelCommands(unsigned connid)
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

void ROOT::Experimental::RCanvasPainter::CheckDataToSend()
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

         ROOT::Experimental::RMenuItems items;
         items.SetFullId(conn.fGetMenu);
         conn.fGetMenu.clear();

         auto drawable = FindPrimitive(fCanvas, items.GetDrawableId());

         if (drawable) {
            R__DEBUG_HERE("CanvasPainter") << "Request menu for drawable " << items.GetDrawableId();
            drawable->PopulateMenu(items);
            buf = "MENU:";
            buf.Append(TBufferJSON::ToJSON(&items, fJsonComp).Data());
         } else {
            R__ERROR_HERE("CanvasPainter") << "Drawable not found " << items.GetDrawableId();
         }

      } else if (!conn.fSendQueue.empty()) {
         buf = conn.fSendQueue.front().c_str();
         conn.fSendQueue.pop();

      } else if ((conn.fSend != fCanvas.GetModified()) && (conn.fDelivered == conn.fSend)) {

         buf = "SNAP:";
         buf += TString::ULLtoa(fCanvas.GetModified(), 10);
         buf += ":";
         buf += CreateSnapshot(conn.fSend);

         conn.fSend = fCanvas.GetModified();
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

void ROOT::Experimental::RCanvasPainter::CanvasUpdated(uint64_t ver, bool async,
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

void ROOT::Experimental::RCanvasPainter::DoWhenReady(const std::string &name, const std::string &arg, bool async,
                                                     CanvasCallback_t callback)
{
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
/// Produce batch output, using chrome headless mode with DOM dump

bool ROOT::Experimental::RCanvasPainter::ProduceBatchOutput(const std::string &fname, int width, int height)
{
   auto snapshot = CreateSnapshot(0);

   return ROOT::Experimental::RWebDisplayHandle::ProduceImage(fname, snapshot, width, height);
}

//////////////////////////////////////////////////////////////////////////
/// Process data from the client

void ROOT::Experimental::RCanvasPainter::ProcessData(unsigned connid, const std::string &arg)
{
   auto conn =
      std::find_if(fWebConn.begin(), fWebConn.end(), [connid](WebConn &item) { return item.fConnId == connid; });

   if (conn == fWebConn.end())
      return; // no connection found

   std::string cdata;

   auto check_header = [&arg, &cdata](const std::string &header) {
      if (arg.compare(0, header.length(), header) != 0)
         return false;
      cdata = arg.substr(header.length());
      return true;
   };

   // R__DEBUG_HERE("CanvasPainter") << "from client " << connid << " got data len:" << arg.length() << " val:" <<
   // arg.substr(0,30);

   if (check_header("READY")) {

   } else if (check_header("SNAPDONE:")) {
      conn->fDelivered = (uint64_t)std::stoll(cdata); // delivered version of the snapshot
   } else if (check_header("GETMENU:")) {
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
      SaveCreatedFile(cdata);
   } else if (check_header("OBJEXEC:")) {
      size_t pos = cdata.find(':');

      if ((pos != std::string::npos) && (pos > 0)) {
         std::string id(cdata, 0, pos);
         cdata.erase(0, pos + 1);
         auto drawable = FindPrimitive(fCanvas, id);
         if (drawable && (cdata.length() > 0)) {
            R__DEBUG_HERE("CanvasPainter") << "execute " << cdata << " for drawable " << id;
            drawable->Execute(cdata);
            fCanvas.Modified();
            fCanvas.Update(true);
         } else if (id == "canvas") {
            R__DEBUG_HERE("CanvasPainter") << "execute " << cdata << " for canvas itself (ignored)";
         }
      }
   } else if (check_header("ATTRCHANGE:")) {
      auto vect = TBufferJSON::FromJSON<std::vector<RChangeAttr>>(cdata);
      if (vect) {
         if (fCanvas.ApplyAttrChanges(*vect))
            fCanvas.Update(true);
      } else {
         R__ERROR_HERE("CanvasPainter") << "Fail to parse vector<RChangeAttr>";
      }
   } else if (check_header("REQ:")) {
      auto req = TBufferJSON::FromJSON<RDrawableRequest>(cdata);
      if (req) {
         std::shared_ptr<RDrawable> drawable;

         req->SetCanvas(&fCanvas);

         if (!req->GetId().empty()) {
            const RPadBase *subpad = nullptr;
            drawable = FindPrimitive(fCanvas, req->GetId(), &subpad);
            req->SetPad(subpad);
            req->SetDrawable(drawable.get());
         }

         auto reply = req->Process();

         if (!reply)
            reply = std::make_unique<RDrawableReply>();

         reply->SetRequestId(req->GetRequestId());

         auto json = TBufferJSON::ToJSON(reply.get(), TBufferJSON::kNoSpaces);
         conn->fSendQueue.emplace("REPL_REQ:"s + json.Data());
      } else {
         R__ERROR_HERE("CanvasPainter") << "Fail to parse RDrawableRequest";
      }
   } else {
      R__ERROR_HERE("CanvasPainter") << "Got not recognized message" << arg;
   }

   CheckDataToSend();
}

//////////////////////////////////////////////////////////////////////////
/// Create web window for canvas

void ROOT::Experimental::RCanvasPainter::CreateWindow()
{
   if (fWindow) return;

   fWindow = RWebWindow::Create();
   fWindow->SetConnLimit(0); // allow any number of connections
   fWindow->SetDefaultPage("file:rootui5sys/canv/canvas.html");
   fWindow->SetCallBacks(
      // connect
      [this](unsigned connid) {
         fWebConn.emplace_back(connid);
         CheckDataToSend();
      },
      // data
      [this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); },
      // disconnect
      [this](unsigned connid) {
         auto conn =
            std::find_if(fWebConn.begin(), fWebConn.end(), [connid](WebConn &item) { return item.fConnId == connid; });

         if (conn != fWebConn.end()) {
            fWebConn.erase(conn);
            CancelCommands(connid);
         }
      });
   // fWindow->SetGeometry(500,300);
}


//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// See ROOT::Experimental::RWebWindowsManager::Show() docu for more info

void ROOT::Experimental::RCanvasPainter::NewDisplay(const std::string &where)
{
   CreateWindow();

   auto sz = fCanvas.GetSize();

   RWebDisplayArgs args(where);

   if ((sz[0].fVal > 10) && (sz[1].fVal > 10)) {
      // extra size of browser window header + ui5 menu
      args.SetWidth((int) sz[0].fVal + 1);
      args.SetHeight((int) sz[1].fVal + 40);
   }

   fWindow->Show(args);
}

//////////////////////////////////////////////////////////////////////////
/// Returns number of connected displays

int ROOT::Experimental::RCanvasPainter::NumDisplays() const
{
   if (!fWindow) return 0;

   return fWindow->NumConnections();
}

//////////////////////////////////////////////////////////////////////////
/// Returns web window name

std::string ROOT::Experimental::RCanvasPainter::GetWindowAddr() const
{
   if (!fWindow) return "";

   return fWindow->GetAddr();
}

//////////////////////////////////////////////////////////////////////////
/// Add window as panel inside canvas window

bool ROOT::Experimental::RCanvasPainter::AddPanel(std::shared_ptr<RWebWindow> win)
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

   std::string addr = fWindow->GetRelativeAddr(win);

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

std::string ROOT::Experimental::RCanvasPainter::CreateSnapshot(RDrawable::Version_t vers)
{
   auto canvitem = std::make_unique<RCanvasDisplayItem>();

   fCanvas.DisplayPrimitives(*canvitem.get(), vers);

   canvitem->SetTitle(fCanvas.GetTitle());
   canvitem->SetWindowSize(fCanvas.GetSize());

   canvitem->BuildFullId(""); // create object id which unique identify it via pointer and position in subpads
   canvitem->SetObjectID("canvas"); // for canvas itself use special id

   TBufferJSON json;
   json.SetCompact(fJsonComp);

   static std::vector<const TClass *> exclude_classes = {
      TClass::GetClass<ROOT::Experimental::RAttrMap::NoValue_t>(),
      TClass::GetClass<ROOT::Experimental::RAttrMap::BoolValue_t>(),
      TClass::GetClass<ROOT::Experimental::RAttrMap::IntValue_t>(),
      TClass::GetClass<ROOT::Experimental::RAttrMap::DoubleValue_t>(),
      TClass::GetClass<ROOT::Experimental::RAttrMap::StringValue_t>(),
      TClass::GetClass<ROOT::Experimental::RAttrMap>(),
      TClass::GetClass<ROOT::Experimental::RStyle::Block_t>(),
      TClass::GetClass<ROOT::Experimental::RPadPos>(),
      TClass::GetClass<ROOT::Experimental::RPadLength>(),
      TClass::GetClass<ROOT::Experimental::RPadExtent>(),
      TClass::GetClass<std::unordered_map<std::string,ROOT::Experimental::RAttrMap::Value_t*>>()
   };

   for (auto cl : exclude_classes)
      json.SetSkipClassInfo(cl);

   auto res = json.StoreObject(canvitem.get(), TClass::GetClass<RCanvasDisplayItem>());

   return std::string(res.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Find drawable in the canvas with specified id
/// Used to communicate with the clients, which does not have any pointer

std::shared_ptr<ROOT::Experimental::RDrawable>
ROOT::Experimental::RCanvasPainter::FindPrimitive(const ROOT::Experimental::RCanvas &can, const std::string &id, const RPadBase **subpad)
{
   std::string search = id;
   size_t pos = search.find("#");
   // exclude extra specifier, later can be used for menu and commands execution
   if (pos != std::string::npos)
      search.resize(pos);

   if (subpad) *subpad = can.FindPadForPrimitiveWithDisplayId(search);

   return can.FindPrimitiveByDisplayId(search);
}

////////////////////////////////////////////////////////////////////////////////
/// Method called when GUI sends file to save on local disk
/// File coded with base64 coding

void ROOT::Experimental::RCanvasPainter::SaveCreatedFile(std::string &reply)
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

void ROOT::Experimental::RCanvasPainter::FrontCommandReplied(const std::string &reply)
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

void ROOT::Experimental::RCanvasPainter::Run(double tm)
{
   if (fWindow) {
      fWindow->Run(tm);
   } else if (tm>0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(int(tm*1000)));
   }
}
