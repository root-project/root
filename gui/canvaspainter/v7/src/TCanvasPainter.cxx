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
#include <ROOT/TMenuItem.hxx>

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <fstream>

#include "THttpEngine.h"
#include "THttpServer.h"
#include "TSystem.h"
#include "TList.h"
#include "TRandom.h"
#include "TPad.h"
#include "TROOT.h"
#include "TApplication.h"
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
         for (i = 0; i < 4; i++) char_array_4[i] = base64_chars.find(char_array_4[i]);

         char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
         char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
         char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

         for (i = 0; (i < 3); i++) ret += char_array_3[i];
         i = 0;
      }
   }

   if (i) {
      for (j = 0; j < i; j++) char_array_4[j] = base64_chars.find(char_array_4[j]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

      for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
   }

   return ret;
}

} // namespace

// ==================================================================

namespace {

/** \class TCanvasPainter
  Handles TCanvas communication with THttpServer.
  */

class TCanvasPainter : public THttpWSHandler,
                       public ROOT::Experimental::Internal::TVirtualCanvasPainter /*, THttpSocketListener*/ {
private:
   struct WebConn {
      THttpWSEngine *fHandle; ///<! websocket handle
      bool fReady;            ///!< when connection ready to send new data
      bool fDrawReady;        ///!< when first drawing is performed
      std::string fGetMenu;   ///<! object id for menu request
      uint64_t fSend;         ///<! indicates version send to connection
      uint64_t fDelivered;    ///<! indicates version confirmed from canvas
      WebConn() : fHandle(0), fReady(false), fDrawReady(false), fGetMenu(), fSend(0), fDelivered(0) {}
   };

   struct WebCommand {
      std::string fId;                                ///<! command identifier
      std::string fName;                              ///<! command name
      std::string fArg;                               ///<! command arg
      bool fRunning;                                  ///<! true when command submitted
      ROOT::Experimental::CanvasCallback_t fCallback; ///<! callback function associated with command
      UInt_t fConnId;                                 ///<! connection id was used to send command
      WebCommand() : fId(), fName(), fArg(), fRunning(false), fCallback(), fConnId(0) {}
   };

   struct WebUpdate {
      uint64_t fVersion;                              ///<! canvas version
      ROOT::Experimental::CanvasCallback_t fCallback; ///<! callback function associated with command
      WebUpdate() : fVersion(0), fCallback() {}
   };

   typedef std::list<WebConn> WebConnList;

   typedef std::list<WebCommand> WebCommandsList;

   typedef std::list<WebUpdate> WebUpdatesList;

   typedef std::vector<ROOT::Experimental::Detail::TMenuItem> MenuItemsVector;

   /// The canvas we are painting. It might go out of existence while painting.
   const ROOT::Experimental::TCanvas &fCanvas;

   Bool_t fBatchMode; ///<! indicate if canvas works in batch mode (can be independent from gROOT->isBatch())

   WebConnList fWebConn;                             ///<! connections list
   ROOT::Experimental::TPadDisplayItem fDisplayList; ///!< full list of items to display
   WebCommandsList fCmds;                            ///!< list of submitted commands
   uint64_t fCmdsCnt;                                ///!< commands counter
   std::string fWaitingCmdId;                        ///!< command id waited for complition

   uint64_t fSnapshotVersion;   ///!< version of snapshot
   std::string fSnapshot;       ///!< last produced snapshot
   uint64_t fSnapshotDelivered; ///!< minimal version delivered to all connections
   WebUpdatesList fUpdatesLst;  ///!< list of callbacks for canvas update

   static TString fAddr;        ///<! real http address (when assigned)
   static THttpServer *gServer; ///<! server

   /// Disable copy construction.
   TCanvasPainter(const TCanvasPainter &) = delete;

   /// Disable assignment.
   TCanvasPainter &operator=(const TCanvasPainter &) = delete;

   ROOT::Experimental::TDrawable *FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id);

   bool CreateHttpServer(bool with_http = false);
   void CheckDataToSend();

   bool WaitWhenCanvasPainted(uint64_t ver);

   std::string CreateSnapshot(const ROOT::Experimental::TCanvas &can);

   /// Send the canvas primitives to the THttpServer.
   // void SendCanvas();

   bool FrontCommandReplied(const std::string &reply);

   void PopFrontCommand(bool res = false);

   void SaveCreatedFile(std::string &reply);

   virtual Bool_t ProcessWS(THttpCallArg *arg) override;

   void CancelCommands(bool cancel_all, UInt_t connid = 0);

   void CancelUpdates();

   void CloseConnections();

public:
   /// Create a TVirtualCanvasPainter for the given canvas.
   /// The painter observes it; it needs to know should the TCanvas be deleted.
   TCanvasPainter(const std::string &name, const ROOT::Experimental::TCanvas &canv, bool batch_mode)
      : THttpWSHandler(name.c_str(), "title"), fCanvas(canv), fBatchMode(batch_mode), fWebConn(), fDisplayList(),
        fCmds(), fCmdsCnt(0), fWaitingCmdId(), fSnapshotVersion(0), fSnapshot(), fSnapshotDelivered(0), fUpdatesLst()
   {
      CreateHttpServer();
      gServer->Register("/web7gui", this);
   }

   virtual ~TCanvasPainter()
   {
      CancelCommands(true);
      CancelUpdates();
      CloseConnections();

      if (gServer)
         gServer->Unregister(this);
      // TODO: should we close server when all canvases are closed?
   }

   virtual bool IsBatchMode() const override { return fBatchMode; }

   virtual void AddDisplayItem(ROOT::Experimental::TDisplayItem *item) final;

   virtual void CanvasUpdated(uint64_t, bool, ROOT::Experimental::CanvasCallback_t) override;

   virtual bool IsCanvasModified(uint64_t) const override;

   /// perform special action when drawing is ready
   virtual void DoWhenReady(const std::string &cmd, const std::string &arg, bool async,
                            ROOT::Experimental::CanvasCallback_t callback) final;

   // open new display for the canvas
   virtual void NewDisplay(const std::string &where) override;

   // void ReactToSocketNews(...) override { SendCanvas(); }

   /** \class CanvasPainterGenerator
       Creates TCanvasPainter objects.
     */

   class GeneratorImpl : public Generator {
   public:
      /// Create a new TCanvasPainter to paint the given TCanvas.
      std::unique_ptr<TVirtualCanvasPainter> Create(const ROOT::Experimental::TCanvas &canv,
                                                    bool batch_mode) const override
      {
         return std::make_unique<TCanvasPainter>("name", canv, batch_mode);
      }
      ~GeneratorImpl() = default;

      /// Set TVirtualCanvasPainter::fgGenerator to a new GeneratorImpl object.
      static void SetGlobalPainter()
      {
         if (TVirtualCanvasPainter::fgGenerator) {
            R__ERROR_HERE("CanvasPainter") << "Generator is already set! Skipping second initialization.";
            return;
         }
         TVirtualCanvasPainter::fgGenerator.reset(new GeneratorImpl());
      }

      /// Release the GeneratorImpl object.
      static void ResetGlobalPainter() { TVirtualCanvasPainter::fgGenerator.reset(); }
   };
};

TString TCanvasPainter::fAddr = "";
THttpServer *TCanvasPainter::gServer = 0;

/** \class TCanvasPainterReg
  Registers TCanvasPainterGenerator as generator with ROOT::Experimental::Internal::TVirtualCanvasPainter.
  */
struct TCanvasPainterReg {
   TCanvasPainterReg() { TCanvasPainter::GeneratorImpl::SetGlobalPainter(); }
   ~TCanvasPainterReg() { TCanvasPainter::GeneratorImpl::ResetGlobalPainter(); }
} canvasPainterReg;

/// \}

} // unnamed namespace

bool TCanvasPainter::CreateHttpServer(bool with_http)
{
   if (!gServer)
      gServer = new THttpServer("dummy");

   if (!with_http || (fAddr.Length() > 0))
      return true;

   // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");

   int http_port = 0;
   const char *ports = gSystem->Getenv("WEBGUI_PORT");
   if (ports)
      http_port = TString(ports).Atoi();
   if (!http_port)
      gRandom->SetSeed(0);

   for (int ntry = 0; ntry < 100; ++ntry) {
      if (!http_port)
         http_port = (int)(8800 + 1000 * gRandom->Rndm(1));

      // TODO: ensure that port can be used
      if (gServer->CreateEngine(TString::Format("http:%d?websocket_timeout=10000", http_port))) {
         fAddr.Form("http://localhost:%d", http_port);
         return true;
      }

      http_port = 0;
   }

   return false;
}

//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// Parameter \par where specified  which program could be used for display creation
/// Possible values:
///
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine (when running via rootqt5), local display, local communication
///  browser - default system web-browser, communication via random http port from range 8800 - 9800
///  <prog> - any program name which will be started instead of default browser, like firefox or /usr/bin/opera
///           one could also specify $url in program name, which will be replaced with canvas URL
///  native - either any available local display or default browser
///
///  Canvas can be displayed in several different places

void TCanvasPainter::NewDisplay(const std::string &where)
{
   TString addr;

   bool is_native = where.empty() || (where == "native"), is_qt5 = (where == "qt5"), ic_cef = (where == "cef");

   Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");

   if (symbol_qt5 && (is_native || is_qt5)) {
      typedef void (*FunctionQt5)(const char *, void *, bool);

      addr.Form("://dummy:8080/web7gui/%s/draw.htm?longpollcanvas&no_root_json%s&qt5", GetName(),
                (IsBatchMode() ? "&batch_mode" : ""));
      // addr.Form("example://localhost:8080/Canvases/%s/draw.htm", Canvas()->GetName());

      Info("NewDisplay", "Show canvas in Qt5 window:  %s", addr.Data());

      FunctionQt5 func = (FunctionQt5)symbol_qt5;
      func(addr.Data(), gServer, IsBatchMode());
      return;
   }

   // TODO: one should try to load CEF libraries only when really needed
   // probably, one should create separate DLL with CEF-related code
   Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (symbol_cef && cef_path && !gSystem->AccessPathName(cef_path) && rootsys && (is_native || ic_cef)) {
      typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *);

      // addr.Form("/web7gui/%s/draw.htm?cef_canvas%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));
      addr.Form("/web7gui/%s/draw.htm?cef_canvas&no_root_json%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));

      Info("NewDisplay", "Show canvas in CEF window:  %s", addr.Data());

      FunctionCef3 func = (FunctionCef3)symbol_cef;
      func(addr.Data(), gServer, IsBatchMode(), rootsys, cef_path);

      return;
   }

   if (!CreateHttpServer(true)) {
      Error("NewDisplay", "Fail to start HTTP server");
      return;
   }

   addr.Form("%s/web7gui/%s/draw.htm?webcanvas", fAddr.Data(), GetName());

   TString exec;

   if (!is_native && !ic_cef && !is_qt5 && (where != "browser")) {
      if (where.find("$url") != std::string::npos) {
         exec = where.c_str();
         exec.ReplaceAll("$url", addr);
      } else {
         exec.Form("%s %s", where.c_str(), addr.Data());
      }
   } else if (gSystem->InheritsFrom("TMacOSXSystem"))
      exec.Form("open %s", addr.Data());
   else
      exec.Form("xdg-open %s &", addr.Data());

   Info("NewDisplay", "Show canvas in browser with cmd:  %s", exec.Data());

   gSystem->Exec(exec);
}

void TCanvasPainter::CanvasUpdated(uint64_t ver, bool async, ROOT::Experimental::CanvasCallback_t callback)
{
   if (ver && fSnapshotDelivered && (ver <= fSnapshotDelivered)) {
      // if given canvas version was already delivered to clients, can return immediately
      if (callback)
         callback(true);
      return;
   }

   fSnapshotVersion = ver;
   fSnapshot = CreateSnapshot(fCanvas);

   CheckDataToSend();

   if (callback) {
      WebUpdate item;
      item.fVersion = ver;
      item.fCallback = callback;
      fUpdatesLst.push_back(item);
   }

   if (!async)
      WaitWhenCanvasPainted(ver);
}

bool TCanvasPainter::WaitWhenCanvasPainted(uint64_t ver)
{
   // simple polling loop until specified version delivered to the clients

   uint64_t cnt = 0;
   bool had_connection = false;

   while (true) {
      if (fWebConn.size() > 0)
         had_connection = true;
      if ((fWebConn.size() == 0) && (had_connection || (cnt > 1000)))
         return false; // wait ~1 min if no new connection established
      if (fSnapshotDelivered >= ver) {
         printf("PAINT READY!!!\n");
         return true;
      }
      gSystem->ProcessEvents();
      gSystem->Sleep((++cnt < 500) ? 1 : 100); // increase sleep interval when do very often
   }

   return false;
}

void TCanvasPainter::DoWhenReady(const std::string &name, const std::string &arg, bool async,
                                 ROOT::Experimental::CanvasCallback_t callback)
{
   if (!async && !fWaitingCmdId.empty()) {
      Error("DoWhenReady", "Fail to submit sync command when previous is still awaited - use async");
      async = true;
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

   if (async)
      return;

   uint64_t cnt = 0;
   bool had_connection = false;

   while (true) {
      if (fWebConn.size() > 0)
         had_connection = true;
      if ((fWebConn.size() == 0) && (had_connection || (cnt > 1000)))
         return; // wait ~1 min if no new connection established
      if (fWaitingCmdId.empty()) {
         printf("Command %s waiting READY!!!\n", name.c_str());
         return;
      }
      gSystem->ProcessEvents();
      gSystem->Sleep((++cnt < 500) ? 1 : 100); // increase sleep interval when do very often
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Remove front command from the command queue
/// If necessary, configured call-back will be invoked

void TCanvasPainter::PopFrontCommand(bool result)
{
   if (fCmds.size() == 0)
      return;

   // simple condition, which will be checked in waiting loop
   if (!fWaitingCmdId.empty() && (fWaitingCmdId == fCmds.front().fId))
      fWaitingCmdId.clear();

   if (fCmds.front().fCallback)
      fCmds.front().fCallback(result);

   fCmds.pop_front();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Cancel commands for given connection ID
/// Invoke all callbacks

void TCanvasPainter::CancelCommands(bool cancel_all, UInt_t connid)
{
   auto iter = fCmds.begin();
   while (iter != fCmds.end()) {
      auto next = iter;
      next++;
      if (cancel_all || (iter->fConnId == connid)) {
         if (fWaitingCmdId == iter->fId)
            fWaitingCmdId.clear();
         iter->fCallback(false);
         fCmds.erase(iter);
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Cancel all pending Canvas::Update()

void TCanvasPainter::CancelUpdates()
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

/////////////////////////////////////////////////////////////////////////////////////////////
/// Close all web connections - will lead to close

void TCanvasPainter::CloseConnections()
{
   for (auto &&conn : fWebConn) {

      if (!conn.fHandle)
         continue;

      conn.fReady = kFALSE;
      conn.fHandle->SendCharStar("CLOSE");
      conn.fHandle->ClearHandle();
      delete conn.fHandle;
      conn.fHandle = nullptr;
   }

   fWebConn.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Process reply of first command in the queue
/// For the moment commands use to create image files

bool TCanvasPainter::FrontCommandReplied(const std::string &reply)
{
   WebCommand &cmd = fCmds.front();

   cmd.fRunning = false;

   bool result = false;

   if ((cmd.fName == "SVG") || (cmd.fName == "PNG") || (cmd.fName == "JPEG")) {
      if (reply.length() == 0) {
         Error("FrontCommandReplied", "Fail to produce %s image %s", cmd.fName.c_str(), cmd.fArg.c_str());
      } else {
         std::string content = base64_decode(reply);
         std::ofstream ofs(cmd.fArg);
         ofs.write(content.c_str(), content.length());
         ofs.close();
         Info("FrontCommandReplied", "Create %s file %s len %d", cmd.fName.c_str(), cmd.fArg.c_str(), (int)content.length());
         result = true;
      }
   } else {
      Error("FrontCommandReplied", "Unknown command %s", cmd.fName.c_str());
   }

   return result;
}

/// Method called when GUI sends file to save on local disk
/// File coded with base64 coding
void TCanvasPainter::SaveCreatedFile(std::string &reply)
{
   size_t pos = reply.find(":");
   if ((pos == std::string::npos) || (pos==0)) {
      Error("SaveCreatedFile", "Not found : separator");
      return;
   }

   std::string fname(reply, 0, pos);
   reply.erase(0, pos+1);

   std::string binary = base64_decode(reply);
   std::ofstream ofs(fname);
   ofs.write(binary.c_str(), binary.length());
   ofs.close();

   Info("SaveCreatedFile", "Create file %s len %d", fname.c_str(), (int)binary.length());
}


Bool_t TCanvasPainter::ProcessWS(THttpCallArg *arg)
{
   if (!arg)
      return kTRUE;

   // try to identify connection for given WS request
   WebConn *conn = 0;
   WebConnList::iterator iter = fWebConn.begin();
   while (iter != fWebConn.end()) {
      if (iter->fHandle && (iter->fHandle->GetId() == arg->GetWSId()) && arg->GetWSId()) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (strcmp(arg->GetMethod(), "WS_CONNECT") == 0) {

      // accept all requests, in future one could limit number of connections
      // arg->Set404(); // refuse connection
      return kTRUE;
   }

   if (strcmp(arg->GetMethod(), "WS_READY") == 0) {
      THttpWSEngine *wshandle = dynamic_cast<THttpWSEngine *>(arg->TakeWSHandle());

      if (conn != 0)
         Error("ProcessWSRequest", "WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fHandle = wshandle;

      fWebConn.push_back(newconn);
      // printf("socket is ready %d\n", fWebConn.back().fReady);

      CheckDataToSend();

      return kTRUE;
   }

   if (strcmp(arg->GetMethod(), "WS_CLOSE") == 0) {
      // connection is closed, one can remove handle

      printf("Connection closed\n");

      UInt_t connid = 0;

      if (conn && conn->fHandle) {
         connid = conn->fHandle->GetId();
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn)
         fWebConn.erase(iter);

      // if there are no other connections - cancel all submitted commands
      CancelCommands((fWebConn.size() == 0), connid);

      CheckDataToSend(); // check if data should be send via other connections

      return kTRUE;
   }

   if (strcmp(arg->GetMethod(), "WS_DATA") != 0) {
      Error("ProcessWSRequest", "WSHandle DATA request expected!");
      return kFALSE;
   }

   if (!conn) {
      Error("ProcessWSRequest", "Get websocket data without valid connection - ignore!!!");
      return kFALSE;
   }

   if (conn->fHandle->PreviewData(arg))
      return kTRUE;

   if (arg->GetPostDataLength() <= 0)
      return kTRUE;

   std::string cdata((const char *)arg->GetPostData(), arg->GetPostDataLength());

   if (cdata.find("READY") == 0) {
      conn->fReady = kTRUE;
      CheckDataToSend();
   } else if (cdata.find("SNAPDONE:") == 0) {
      cdata.erase(0, 9);
      conn->fReady = kTRUE;
      conn->fDrawReady = kTRUE;                       // at least first drawing is performed
      conn->fDelivered = (uint64_t)std::stoll(cdata); // delivered version of the snapshot
      CheckDataToSend();
   } else if (cdata.find("RREADY:") == 0) {
      conn->fReady = kTRUE;
      conn->fDrawReady = kTRUE; // at least first drawing is performed
      CheckDataToSend();
   } else if (cdata.find("GETMENU:") == 0) {
      conn->fReady = kTRUE;
      cdata.erase(0, 8);
      conn->fGetMenu = cdata;
      CheckDataToSend();
   } else if (cdata == "QUIT") {
      if (gApplication)
         gApplication->Terminate(0);
   } else if (cdata == "RELOAD") {
      conn->fSend = 0; // reset send version, causes new data sending
      CheckDataToSend();
   } else if (cdata == "INTERRUPT") {
      gROOT->SetInterrupt();
   } else if (cdata.find("REPLY:") == 0) {
      cdata.erase(0, 6);
      const char *sid = cdata.c_str();
      const char *separ = strchr(sid, ':');
      std::string id;
      if (separ)
         id.append(sid, separ - sid);
      if (fCmds.size() == 0) {
         Error("ProcessWS", "Get REPLY without command");
      } else if (!fCmds.front().fRunning) {
         Error("ProcessWS", "Front command is not running when get reply");
      } else if (fCmds.front().fId != id) {
         Error("ProcessWS", "Mismatch with front command and ID in REPLY");
      } else {
         bool res = FrontCommandReplied(separ + 1);
         PopFrontCommand(res);
      }
      conn->fReady = kTRUE;
      CheckDataToSend();
   } else if (cdata.find("SAVE:") == 0) {
      cdata.erase(0,5);
      SaveCreatedFile(cdata);
   } else if (cdata.find("OBJEXEC:") == 0) {
      cdata.erase(0, 8);
      size_t pos = cdata.find(':');

      if ((pos != std::string::npos) && (pos > 0)) {
         std::string id(cdata, 0, pos);
         cdata.erase(0, pos + 1);
         ROOT::Experimental::TDrawable *drawable = FindDrawable(fCanvas, id);
         if (drawable && (cdata.length() > 0)) {
            printf("Execute %s for drawable %p\n", cdata.c_str(), drawable);
            drawable->Execute(cdata);
         } else if (id == ROOT::Experimental::TDisplayItem::MakeIDFromPtr((void *)&fCanvas)) {
            printf("Execute %s for canvas itself (ignore for the moment)\n", cdata.c_str());
         }
      }
   } else if (cdata == "KEEPALIVE") {
      // do nothing, it is just keep alive message for websocket
   }

   return kTRUE;
}

void TCanvasPainter::CheckDataToSend()
{

   uint64_t min_delivered = 0;

   for (auto &&conn : fWebConn) {

      if (conn.fDelivered && (!min_delivered || (min_delivered < conn.fDelivered)))
         min_delivered = conn.fDelivered;

      if (!conn.fReady || !conn.fHandle)
         continue;

      TString buf;

      if (conn.fDrawReady && (fCmds.size() > 0) && !fCmds.front().fRunning) {
         WebCommand &cmd = fCmds.front();
         cmd.fRunning = true;
         buf = "CMD:";
         buf.Append(cmd.fId);
         buf.Append(":");
         buf.Append(cmd.fName);
         cmd.fConnId = conn.fHandle->GetId();
      } else if (!conn.fGetMenu.empty()) {
         ROOT::Experimental::TDrawable *drawable = FindDrawable(fCanvas, conn.fGetMenu);

         printf("Request menu for object %s found drawable %p\n", conn.fGetMenu.c_str(), drawable);

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
         conn.fReady = kFALSE;
         conn.fHandle->SendCharStar(buf.Data());
      }
   }

   // if there are updates submitted, but all connections disappeared - cancel all updates
   if ((fWebConn.size() == 0) && fSnapshotDelivered)
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

bool TCanvasPainter::IsCanvasModified(uint64_t id) const
{
   return fSnapshotDelivered != id;
}

void TCanvasPainter::AddDisplayItem(ROOT::Experimental::TDisplayItem *item)
{
   fDisplayList.Add(item);
}

ROOT::Experimental::TDrawable *TCanvasPainter::FindDrawable(const ROOT::Experimental::TCanvas &can,
                                                                      const std::string &id)
{

   std::string search = id;
   size_t pos = search.find("#");
   // exclude extra specifier, later can be used for menu and commands execution
   if (pos != std::string::npos) search.resize(pos);

   for (auto &&drawable : can.GetPrimitives()) {

      if (search == ROOT::Experimental::TDisplayItem::MakeIDFromPtr(&(*drawable)))
         return &(*drawable);
   }

   return nullptr;
}

std::string TCanvasPainter::CreateSnapshot(const ROOT::Experimental::TCanvas &can)
{

   fDisplayList.Clear();

   fDisplayList.SetObjectIDAsPtr((void *)&can);

   TPad *dummy = new TPad(); // just provide old class where all kind of info (size, ranges) already provided

   auto *snap = new ROOT::Experimental::TUniqueDisplayItem<TPad>(dummy);
   snap->SetObjectIDAsPtr((void *)&can);
   fDisplayList.Add(snap);

   for (auto &&drawable : can.GetPrimitives()) {

      drawable->Paint(*this);

      fDisplayList.Last()->SetObjectIDAsPtr(&(*drawable));

      // ROOT::Experimental::TDisplayItem *sub = drawable->CreateSnapshot(can);
      // if (!sub) continue;
      // sub->SetObjectIDAsPtr(&(*drawable));
      // lst.Add(sub);
   }

   TString res = TBufferJSON::ConvertToJSON(&fDisplayList, gROOT->GetClass("ROOT::Experimental::TPadDisplayItem"));

   fDisplayList.Clear();

   // printf("JSON %s\n", res.Data());

   return std::string(res.Data());
}

// void TCanvasPainter::SendCanvas() {
//  for (auto &&drawable: fCanvas.GetPrimitives()) {
//    drawable->Paint(*this);
//  }
//}
