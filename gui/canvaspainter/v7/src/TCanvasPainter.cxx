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

#include "ROOT/TVirtualCanvasPainter.hxx"
#include "ROOT/TCanvas.hxx"
#include <ROOT/TLogger.hxx>
#include <ROOT/TDisplayItem.hxx>
#include <ROOT/TPadDisplayItem.hxx>
#include <ROOT/TMenuItem.hxx>

#include <ROOT/TWebWindow.hxx>
#include <ROOT/TWebWindowsManager.hxx>

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <fstream>

#include "TList.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"

// =================================================================

// found on https://github.com/ReneNyffenegger/cpp-base64

#include <ctype.h>

/*
   base64.cpp and base64.h
   base64 encoding and decoding with C++.
   Version: 1.01.00
   Copyright (C) 2004-2017 René Nyffenegger
   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.
   3. This notice may not be removed or altered from any source distribution.
   René Nyffenegger rene.nyffenegger@adp-gmbh.ch
*/

#include <iostream>

namespace {

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static inline bool is_base64(unsigned char c)
{
   return (isalnum(c) || (c == '+') || (c == '/'));
}

/*
std::string base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len)
{
   std::string ret;
   int i = 0;
   int j = 0;
   unsigned char char_array_3[3];
   unsigned char char_array_4[4];

   while (in_len--) {
      char_array_3[i++] = *(bytes_to_encode++);
      if (i == 3) {
         char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
         char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
         char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
         char_array_4[3] = char_array_3[2] & 0x3f;

         for (i = 0; (i < 4); i++) ret += base64_chars[char_array_4[i]];
         i = 0;
      }
   }

   if (i) {
      for (j = i; j < 3; j++) char_array_3[j] = '\0';

      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

      for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];

      while ((i++ < 3)) ret += '=';
   }

   return ret;
}
*/

std::string base64_decode(std::string const &encoded_string)
{
   int in_len = encoded_string.size();
   int i = 0;
   int j = 0;
   int in_ = 0;
   unsigned char char_array_4[4], char_array_3[3];
   std::string ret;

   while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
      char_array_4[i++] = encoded_string[in_];
      in_++;
      if (i == 4) {
         for (i = 0; i < 4; i++)
            char_array_4[i] = base64_chars.find(char_array_4[i]);

         char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
         char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
         char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

         for (i = 0; (i < 3); i++)
            ret += char_array_3[i];
         i = 0;
      }
   }

   if (i) {
      for (j = 0; j < i; j++)
         char_array_4[j] = base64_chars.find(char_array_4[j]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

      for (j = 0; (j < i - 1); j++)
         ret += char_array_3[j];
   }

   return ret;
}

} // namespace

// ==========================================================================================================

// new implementation of canvas painter, using TWebWindow

namespace ROOT {
namespace Experimental {

class TCanvasPainter : public Internal::TVirtualCanvasPainter {
private:
   struct WebConn {
      unsigned fConnId{0};    ///<! connection id
      bool fDrawReady{false}; ///!< when first drawing is performed
      std::string fGetMenu{}; ///<! object id for menu request
      uint64_t fSend{0};      ///<! indicates version send to connection
      uint64_t fDelivered{0}; ///<! indicates version confirmed from canvas
      WebConn() = default;
   };

   struct WebCommand {
      std::string fId{};                   ///<! command identifier
      std::string fName{};                 ///<! command name
      std::string fArg{};                  ///<! command arg
      bool fRunning{false};                ///<! true when command submitted
      CanvasCallback_t fCallback{nullptr}; ///<! callback function associated with command
      unsigned fConnId{0};                 ///<! connection id was used to send command
      WebCommand() = default;
   };

   struct WebUpdate {
      uint64_t fVersion{0};                ///<! canvas version
      CanvasCallback_t fCallback{nullptr}; ///<! callback function associated with command
      WebUpdate() = default;
   };

   typedef std::list<WebConn> WebConnList;

   typedef std::list<WebCommand> WebCommandsList;

   typedef std::list<WebUpdate> WebUpdatesList;

   typedef std::vector<ROOT::Experimental::Detail::TMenuItem> MenuItemsVector;

   /// The canvas we are painting. It might go out of existence while painting.
   const TCanvas &fCanvas; ///<!  Canvas

   std::shared_ptr<TWebWindow> fWindow; ///!< configured display

   WebConnList fWebConn;           ///<! connections list
   bool fHadWebConn{false};        ///<! true if any connection were existing
   //TPadDisplayItem fDisplayList;   ///<! full list of items to display
   //std::string fCurrentDrawableId; ///<! id of drawable, which paint method is called
   WebCommandsList fCmds;          ///<! list of submitted commands
   uint64_t fCmdsCnt{0};           ///<! commands counter
   std::string fWaitingCmdId;      ///<! command id waited for completion

   uint64_t fSnapshotVersion{0};   ///<! version of snapshot
   std::string fSnapshot;          ///<! last produced snapshot
   uint64_t fSnapshotDelivered{0}; ///<! minimal version delivered to all connections
   WebUpdatesList fUpdatesLst;     ///<! list of callbacks for canvas update

   /// Disable copy construction.
   TCanvasPainter(const TCanvasPainter &) = delete;

   /// Disable assignment.
   TCanvasPainter &operator=(const TCanvasPainter &) = delete;

   void CancelUpdates();

   void CancelCommands(unsigned connid = 0);

   void CheckDataToSend();

   void ProcessData(unsigned connid, const std::string &arg);

   std::string CreateSnapshot(const ROOT::Experimental::TCanvas &can);

   std::shared_ptr<TDrawable> FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id);

   void SaveCreatedFile(std::string &reply);

   bool FrontCommandReplied(const std::string &reply);

   void PopFrontCommand(bool result);

   int CheckDeliveredVersion(uint64_t ver, double);

   int CheckWaitingCmd(const std::string &cmdname, double);

public:
   TCanvasPainter(const TCanvas &canv) : fCanvas(canv) {}

   virtual ~TCanvasPainter();

//   virtual void AddDisplayItem(std::unique_ptr<TDisplayItem> &&item) override
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

   virtual bool AddPanel(std::shared_ptr<TWebWindow>) override;

   /** \class CanvasPainterGenerator
          Creates TCanvasPainter objects.
        */

   class GeneratorImpl : public Generator {
   public:
      /// Create a new TCanvasPainter to paint the given TCanvas.
      std::unique_ptr<TVirtualCanvasPainter> Create(const ROOT::Experimental::TCanvas &canv) const override
      {
         return std::make_unique<TCanvasPainter>(canv);
      }
      ~GeneratorImpl() = default;

      /// Set TVirtualCanvasPainter::fgGenerator to a new GeneratorImpl object.
      static void SetGlobalPainter()
      {
         if (GetGenerator()) {
            R__ERROR_HERE("NewPainter") << "Generator is already set! Skipping second initialization.";
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
      auto curr = iter;
      iter++;
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
      auto next = iter;
      next++;
      if (!connid || (iter->fConnId == connid)) {
         if (fWaitingCmdId == iter->fId)
            fWaitingCmdId.clear();
         iter->fCallback(false);
         fCmds.erase(iter);
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

         printf("Request menu for object %s found drawable %p\n", conn.fGetMenu.c_str(), drawable.get());

         if (drawable) {

            ROOT::Experimental::TMenuItems items;

            drawable->PopulateMenu(items);

            // FIXME: got problem with std::list<TMenuItem>, can be generic TBufferJSON
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
         fWindow->Send(buf.Data(), conn.fConnId);
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
            curr->fCallback(true);
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

   // wait 100 seconds that canvas is painted
   if (!async)
      fWindow->WaitFor([this, ver](double tm) { return CheckDeliveredVersion(ver, tm); }, 100);
}

///////////////////////////////////////////////////
/// Used to wait until submitted command executed

int ROOT::Experimental::TCanvasPainter::CheckWaitingCmd(const std::string &cmdname, double)
{
   if (fWebConn.empty() && fHadWebConn)
      return -1;
   if (fWaitingCmdId.empty()) {
      printf("Command %s waiting READY!!!\n", cmdname.c_str());
      return 1;
   }
   return 0;
}

//////////////////////////////////////////////////////////////////////////
/// perform special action when drawing is ready

void ROOT::Experimental::TCanvasPainter::DoWhenReady(const std::string &name, const std::string &arg, bool async,
                                                     CanvasCallback_t callback)
{
   if (!async && !fWaitingCmdId.empty()) {
      R__ERROR_HERE("DoWhenReady") << "Fail to submit sync command when previous is still awaited - use async";
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

   if (!async)
      fWindow->WaitFor([this, name](double tm) { return CheckWaitingCmd(name, tm); }, 100);
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
      printf("websocket is ready %u\n", connid);

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

   printf("Get data %u %.30s\n", connid, arg.c_str());

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
         printf("Get REPLY without command\n");
      } else if (!fCmds.front().fRunning) {
         printf("Front command is not running when get reply\n");
      } else if (fCmds.front().fId != id) {
         printf("Mismatch with front command and ID in REPLY\n");
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
            printf("Execute %s for drawable %p\n", cdata.c_str(), drawable.get());
            drawable->Execute(cdata);
         } else if (id == "canvas") {
            printf("Execute %s for canvas itself (ignored for the moment)\n", cdata.c_str());
         }
      }
   } else {
      printf("Got not recognized reply %s\n", arg.c_str());
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
/// Add window as panel inside canvas window

bool ROOT::Experimental::TCanvasPainter::AddPanel(std::shared_ptr<TWebWindow> win)
{
   printf("TCanvasPainter::AddPanel %p\n", win.get());

   if (!fWindow) {
      R__ERROR_HERE("AddPanel") << "Canvas not yet shown";
      return false;
   }

   if (fWindow->IsBatchMode()) {
      R__ERROR_HERE("AddPanel") << "Canvas shown in batch mode";
      return false;
   }

   std::string addr = fWindow->RelativeAddr(win);

   if (addr.length() == 0) {
      R__ERROR_HERE("AddPanel") << "Cannot attach panel to canvas";
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

std::string ROOT::Experimental::TCanvasPainter::CreateSnapshot(const ROOT::Experimental::TCanvas &can)
{

/*   fDisplayList.Clear();

   fDisplayList.SetObjectIDAsPtr((void *)&can);

   fDisplayList.SetFrame(can.GetFrame());

   for (auto &&drawable : can.GetPrimitives()) {

      fCurrentDrawableId = TDisplayItem::MakeIDFromPtr(drawable.get());

      drawable->Paint(*this);
   }
*/

   PaintDrawables(can);

   // fPadDisplayItem.SetObjectID("canvas"); // for canvas itself use special id

   TString res = TBufferJSON::ToJSON(fPadDisplayItem.get() /*, 23 */);

   TBufferJSON::ExportToFile("canv.json", fPadDisplayItem.get(), gROOT->GetClass("ROOT::Experimental::TPadDisplayItem"));

   fPadDisplayItem.reset(); // no need to keep memory any longer

   // fDisplayList.Clear();



   //   std::ofstream ofs("snap.json");
   //   ofs << res.Data() << std::endl;

   return std::string(res.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Find drawable in the canvas with specified id
/// Used to communicate with the clients, which does not have any pointer

std::shared_ptr<ROOT::Experimental::TDrawable>
ROOT::Experimental::TCanvasPainter::FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id)
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
      R__ERROR_HERE("SaveCreatedFile") << "Not found : separator";
      return;
   }

   std::string fname(reply, 0, pos);
   reply.erase(0, pos + 1);

   std::string binary = base64_decode(reply);
   std::ofstream ofs(fname);
   ofs.write(binary.c_str(), binary.length());
   ofs.close();

   printf("Create file %s len %d\n", fname.c_str(), (int)binary.length());
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
         R__ERROR_HERE("FrontCommandReplied") << "Fail to produce image" << cmd.fArg;
      } else {
         std::string content = base64_decode(reply);
         std::ofstream ofs(cmd.fArg);
         ofs.write(content.c_str(), content.length());
         ofs.close();
         printf("Create %s file %s len %d\n", cmd.fName.c_str(), cmd.fArg.c_str(), (int)content.length());
         result = true;
      }
   } else if (cmd.fName.find("ADDPANEL:") == 0) {
      printf("Get reply for ADDPANEL %s\n", reply.c_str());
      result = (reply == "true");
   } else {
      R__ERROR_HERE("FrontCommandReplied") << "Unknown command " << cmd.fName;
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

   if (fCmds.front().fCallback)
      fCmds.front().fCallback(result);

   fCmds.pop_front();
}
