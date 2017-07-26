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
      Bool_t fReady;          ///!< when connection ready to send new data
      Bool_t fDrawReady;      ///!< when first drawing is performed
      std::string fGetMenu;   ///<! object id for menu request
      Bool_t fModified;       ///<! indicates if canvas was modified for that connection
      WebConn() : fHandle(0), fReady(kFALSE), fDrawReady(kFALSE), fGetMenu(), fModified(kFALSE) {}
   };

   typedef std::list<WebConn> WebConnList;

   typedef std::vector<ROOT::Experimental::Detail::TMenuItem> MenuItemsVector;

   /// The canvas we are painting. It might go out of existence while painting.
   const ROOT::Experimental::TCanvas &fCanvas;

   Bool_t fBatchMode; ///<! indicate if canvas works in batch mode (can be independent from gROOT->isBatch())

   WebConnList fWebConn;                             ///<! connections list
   ROOT::Experimental::TPadDisplayItem fDisplayList; ///!< full list of items to display
   std::string fNextCmd;                             ///!< command which will be executed next

   static std::string fAddr;    ///<! real http address (when assigned)
   static THttpServer *gServer; ///<! server

   /// Disable copy construction.
   TCanvasPainter(const TCanvasPainter &) = delete;

   /// Disable assignment.
   TCanvasPainter &operator=(const TCanvasPainter &) = delete;

   ROOT::Experimental::Internal::TDrawable *FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id);

   void CreateHttpServer(Bool_t with_http = kFALSE);
   void PopupBrowser();
   void CheckModifiedFlag();
   std::string CreateSnapshot(const ROOT::Experimental::TCanvas &can);

   /// Send the canvas primitives to the THttpServer.
   // void SendCanvas();

   virtual Bool_t ProcessWS(THttpCallArg *arg);

public:
   /// Create a TVirtualCanvasPainter for the given canvas.
   /// The painter observes it; it needs to know should the TCanvas be deleted.
   TCanvasPainter(const std::string &name, const ROOT::Experimental::TCanvas &canv, bool batch_mode)
      : THttpWSHandler(name.c_str(), "title"), fCanvas(canv), fBatchMode(batch_mode), fWebConn()
   {
      CreateHttpServer();
      gServer->Register("/web7gui", this);
      PopupBrowser();
   }

   virtual bool IsBatchMode() const { return fBatchMode; }

   virtual void AddDisplayItem(ROOT::Experimental::TDisplayItem *item) final;

   /// perform special action when drawing is ready
   virtual void DoWhenReady(const std::string &cmd, const std::string &arg) final;

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

std::string TCanvasPainter::fAddr = "";
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

void TCanvasPainter::CreateHttpServer(Bool_t with_http)
{
   if (!gServer)
      gServer = new THttpServer("dummy");

   if (!with_http || !fAddr.empty())
      return;

   // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");
   const char *port = gSystem->Getenv("WEBGUI_PORT");
   TString buf;
   if (!port) {
      gRandom->SetSeed(0);
      buf.Form("%d", (int)(8800 + 1000 * gRandom->Rndm(1)));
      port = buf.Data(); // "8181";
   }
   fAddr = TString::Format("http://localhost:%s", port).Data();
   gServer->CreateEngine(TString::Format("http:%s?websocket_timeout=10000", port).Data());
}

void TCanvasPainter::PopupBrowser()
{
   TString addr;

   Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");
   if (symbol_qt5) {
      typedef void (*FunctionQt5)(const char *, void *, bool);

      addr.Form("://dummy:8080/web7gui/%s/draw.htm?longpollcanvas%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));
      // addr.Form("example://localhost:8080/Canvases/%s/draw.htm", Canvas()->GetName());

      Info("PopupBrowser", "Show canvas in Qt5 window:  %s", addr.Data());

      FunctionQt5 func = (FunctionQt5)symbol_qt5;
      func(addr.Data(), gServer, IsBatchMode());
      return;
   }

   Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (symbol_cef && cef_path && !gSystem->AccessPathName(cef_path) && rootsys) {
      typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *);

      // addr.Form("/web7gui/%s/draw.htm?cef_canvas%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));
      addr.Form("/web7gui/%s/draw.htm?cef_canvas%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));

      Info("PopupBrowser", "Show canvas in CEF window:  %s", addr.Data());

      FunctionCef3 func = (FunctionCef3)symbol_cef;
      func(addr.Data(), gServer, IsBatchMode(), rootsys, cef_path);

      return;
   }

   CreateHttpServer(kTRUE); // ensure that http port is available

   addr.Form("%s/web7gui/%s/draw.htm?webcanvas", fAddr.c_str(), GetName());

   TString exec;

   if (gSystem->InheritsFrom("TMacOSXSystem"))
      exec.Form("open %s", addr.Data());
   else
      exec.Form("xdg-open %s &", addr.Data());

   Info("PopupBrowser", "Show canvas in browser with cmd:  %s", exec.Data());

   gSystem->Exec(exec);
}

void TCanvasPainter::DoWhenReady(const std::string &cmd, const std::string &arg)
{
   fNextCmd = cmd + ":" + arg;
   CheckModifiedFlag();
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
      newconn.fModified = kTRUE;

      fWebConn.push_back(newconn);
      printf("socket is ready\n");

      return kTRUE;
   }

   if (strcmp(arg->GetMethod(), "WS_CLOSE") == 0) {
      // connection is closed, one can remove handle

      if (conn && conn->fHandle) {
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn)
         fWebConn.erase(iter);
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

   const char *cdata = (arg->GetPostDataLength() <= 0) ? "" : (const char *)arg->GetPostData();

   if (strncmp(cdata, "READY", 5) == 0) {
      conn->fReady = kTRUE;
      CheckModifiedFlag();
   } else if (strncmp(cdata, "RREADY:", 7) == 0) {
      conn->fReady = kTRUE;
      conn->fDrawReady = kTRUE; // at least first drawing is performed
      CheckModifiedFlag();
   } else if (strncmp(cdata, "GETMENU:", 8) == 0) {
      conn->fReady = kTRUE;
      conn->fGetMenu = cdata + 8;
      CheckModifiedFlag();
   } else if (strncmp(cdata, "GEXE:", 5) == 0) {
      // TODO: temporary solution, should be removed later
      // used now to terminate ROOT session
      gROOT->ProcessLine(cdata + 5);
   } else if (strncmp(cdata, "DONESVG:", 8) == 0) {
      TString buf(cdata + 8);
      Int_t pos = buf.First(':');
      if (pos > 0) {
         TString fname = buf(0, pos);
         buf.Remove(0, pos + 1);
         std::ofstream ofs(fname.Data());
         ofs.write(buf.Data(), buf.Length());
         ofs.close();
         printf("Create SVG file %s len %d\n", fname.Data(), buf.Length());
      }
      conn->fReady = kTRUE;
      CheckModifiedFlag();
   } else if (strncmp(cdata, "DONEPNG:", 8) == 0) {
      TString buf(cdata + 8);
      Int_t pos = buf.First(':');
      if (pos > 0) {
         TString fname = buf(0, pos);
         buf.Remove(0, pos + 1);

         std::string png = base64_decode(buf.Data());

         std::ofstream ofs(fname.Data());
         ofs.write(png.c_str(), png.length());
         ofs.close();
         printf("Create PNG file %s len %d\n", fname.Data(), (int)png.length());
      } else {
         printf("Error when producing PNG image %s\n", buf.Data());
      }
      conn->fReady = kTRUE;
      CheckModifiedFlag();
   } else if (strncmp(cdata, "OBJEXEC:", 8) == 0) {
      TString buf(cdata + 8);
      Int_t pos = buf.First(':');

      if (pos > 0) {
         TString id = buf(0, pos);
         buf.Remove(0, pos + 1);
         ROOT::Experimental::Internal::TDrawable *drawable = FindDrawable(fCanvas, id.Data());
         if (drawable && (buf.Length() > 0)) {
            printf("Execute %s for drawable %p\n", buf.Data(), drawable);
            drawable->Execute(buf.Data());
         }
      }
   }

   return kTRUE;
}

void TCanvasPainter::CheckModifiedFlag()
{

   for (WebConnList::iterator citer = fWebConn.begin(); citer != fWebConn.end(); ++citer) {
      WebConn &conn = *citer;

      if (!conn.fReady || !conn.fHandle)
         continue;

      TString buf;

      if (conn.fDrawReady && !fNextCmd.empty()) {
         buf = fNextCmd;
         fNextCmd.clear();
      } else if (!conn.fGetMenu.empty()) {
         ROOT::Experimental::Internal::TDrawable *drawable = FindDrawable(fCanvas, conn.fGetMenu);

         printf("Request menu for object %s found drawable %p\n", conn.fGetMenu.c_str(), drawable);

         if (drawable) {

            ROOT::Experimental::TMenuItems items;

            drawable->PopulateMenu(items);

            // FIXME: got problem with std::list<TMenuItem>, can be generic TBufferJSON
            buf = "MENU";
            buf.Append(items.ProduceJSON());
         }

         conn.fGetMenu = "";
      } else if (conn.fModified) {
         // buf = "JSON";
         // buf  += TBufferJSON::ConvertToJSON(Canvas(), 3);

         buf = "SNAP";
         buf += CreateSnapshot(fCanvas);
         conn.fModified = kFALSE;
      }

      if (buf.Length() > 0) {
         // sending of data can be moved into separate thread - not to block user code
         conn.fReady = kFALSE;
         conn.fHandle->SendCharStar(buf.Data());
      }
   }
}

void TCanvasPainter::AddDisplayItem(ROOT::Experimental::TDisplayItem *item)
{
   fDisplayList.Add(item);
}

ROOT::Experimental::Internal::TDrawable *TCanvasPainter::FindDrawable(const ROOT::Experimental::TCanvas &can,
                                                                      const std::string &id)
{

   for (auto &&drawable : can.GetPrimitives()) {

      if (id == ROOT::Experimental::TDisplayItem::MakeIDFromPtr(&(*drawable)))
         return &(*drawable);
   }

   return nullptr;
}

std::string TCanvasPainter::CreateSnapshot(const ROOT::Experimental::TCanvas &can)
{

   fDisplayList.Clear();

   fDisplayList.SetObjectIDAsPtr((void *)&can);

   TPad *dummy = new TPad(); // just to keep information we need for different ranges now

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
