/// \file TCanvasPainter.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


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

#include "THttpEngine.h"
#include "THttpServer.h"
#include "TSystem.h"
#include "TList.h"
#include "TRandom.h"
#include "TPad.h"
#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"

namespace {

/** \class TCanvasPainter
  Handles TCanvas communication with THttpServer.
  */

class TCanvasPainter: public THttpWSHandler,
                      public ROOT::Experimental::Internal::TVirtualCanvasPainter /*, THttpSocketListener*/ {
private:

   struct WebConn {
      THttpWSEngine  *fHandle;     ///<! websocket handle
      Bool_t          fReady;
      std::string     fGetMenu;    ///<! object id for menu request
      Bool_t          fModified;
      WebConn() : fHandle(0), fReady(kFALSE), fGetMenu(), fModified(kFALSE) {}
   };

  typedef std::list<WebConn> WebConnList;

  typedef std::vector<ROOT::Experimental::Detail::TMenuItem> MenuItemsVector;

  /// The canvas we are painting. It might go out of existence while painting.
  const ROOT::Experimental::TCanvas& fCanvas;

   WebConnList     fWebConn;      ///<! connections list
   ROOT::Experimental::TPadDisplayItem  fDisplayList; ///!< full list of items to display

   MenuItemsVector  fMenuItems;    ///<! list of menu items

   static std::string fAddr;
   static THttpServer *gServer;



  /// Disable copy construction.
  TCanvasPainter(const TCanvasPainter&) = delete;


  /// Disable assignment.
  TCanvasPainter& operator=(const TCanvasPainter&) = delete;

  ROOT::Experimental::Internal::TDrawable *FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id);

  void CreateHttpServer();
  void PopupBrowser();
  void CheckModifiedFlag();
  std::string CreateSnapshot(const ROOT::Experimental::TCanvas& can);


  /// Send the canvas primitives to the THttpServer.
  // void SendCanvas();

   virtual Bool_t ProcessWS(THttpCallArg *arg);


public:
  /// Create a TVirtualCanvasPainter for the given canvas.
  /// The painter observes it; it needs to know should the TCanvas be deleted.
  TCanvasPainter(const std::string &name, const ROOT::Experimental::TCanvas& canv):
     THttpWSHandler(name.c_str(), "title"),
     fCanvas(canv),
     fWebConn()
   {
      CreateHttpServer();
      gServer->Register("/web7gui", this);
      PopupBrowser();
   }


   virtual void AddDisplayItem(ROOT::Experimental::TDisplayItem *item) final;

   virtual void AddMenuItem(const std::string &name, const std::string &title, const std::string &exec) final;

   virtual void AddChkMenuItem(const std::string &name, const std::string &title, bool checked, const std::string &toggle) final;

  // void ReactToSocketNews(...) override { SendCanvas(); }

/** \class CanvasPainterGenerator
    Creates TCanvasPainter objects.
  */

  class GeneratorImpl: public Generator {
  public:

    /// Create a new TCanvasPainter to paint the given TCanvas.
    std::unique_ptr<TVirtualCanvasPainter> Create(const ROOT::Experimental::TCanvas& canv) const override {
      return std::make_unique<TCanvasPainter>("name", canv);
    }
    ~GeneratorImpl() = default;

    /// Set TVirtualCanvasPainter::fgGenerator to a new GeneratorImpl object.
    static void SetGlobalPainter() {
      if (TVirtualCanvasPainter::fgGenerator) {
        R__ERROR_HERE("CanvasPainter") << "Generator is already set! Skipping second initialization.";
        return;
      }
      TVirtualCanvasPainter::fgGenerator.reset(new GeneratorImpl());
    }

    /// Release the GeneratorImpl object.
    static void ResetGlobalPainter() {
      TVirtualCanvasPainter::fgGenerator.reset();
    }
  };
};

std::string TCanvasPainter::fAddr = "";
THttpServer *TCanvasPainter::gServer = 0;


/** \class TCanvasPainterReg
  Registers TCanvasPainterGenerator as generator with ROOT::Experimental::Internal::TVirtualCanvasPainter.
  */
struct TCanvasPainterReg {
  TCanvasPainterReg() {
    TCanvasPainter::GeneratorImpl::SetGlobalPainter();
  }
  ~TCanvasPainterReg() {
    TCanvasPainter::GeneratorImpl::ResetGlobalPainter();
  }
} canvasPainterReg;

/// \}

} // unnamed namespace


void TCanvasPainter::CreateHttpServer()
{
   if (gServer) return;

   // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");
   const char *port = gSystem->Getenv("WEBGUI_PORT");
   TString buf;
   if (!port) {
      gRandom->SetSeed(0);
      buf.Form("%d", (int) (8800 + 1000* gRandom->Rndm(1)));
      port = buf.Data(); // "8181";
   }
   fAddr = TString::Format("http://localhost:%s", port).Data();
   gServer = new THttpServer(TString::Format("http:%s?websocket_timeout=10000", port).Data());
}


void TCanvasPainter::PopupBrowser()
{
   TString addr, exec;

   addr.Form("%s/web7gui/%s/draw.htm?webcanvas", fAddr.c_str(), GetName());

   if (gSystem->InheritsFrom("TMacOSXSystem"))
      exec.Form("open %s", addr.Data());
   else
      exec.Form("xdg-open %s &", addr.Data());
   printf("Exec %s\n", exec.Data());

   gSystem->Exec(exec);
}

Bool_t TCanvasPainter::ProcessWS(THttpCallArg *arg)
{
   if (!arg) return kTRUE;

   // try to identify connection for given WS request
   WebConn* conn = 0;
   WebConnList::iterator iter = fWebConn.begin();
   while (iter != fWebConn.end()) {
      if (iter->fHandle && (iter->fHandle->GetId() == arg->GetWSId()) && arg->GetWSId()) {
         conn = &(*iter); break;
      }
      ++iter;
   }

   if (strcmp(arg->GetMethod(),"WS_CONNECT")==0) {

      // accept all requests, in future one could limit number of connections
      // arg->Set404(); // refuse connection
      return kTRUE;
   }

   if (strcmp(arg->GetMethod(),"WS_READY")==0) {
      THttpWSEngine* wshandle = dynamic_cast<THttpWSEngine*> (arg->TakeWSHandle());

      if (conn != 0) Error("ProcessWSRequest","WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fHandle = wshandle;
      newconn.fModified = kTRUE;

      fWebConn.push_back(newconn);
      printf("socket is ready\n");

      return kTRUE;
   }

   if (strcmp(arg->GetMethod(),"WS_CLOSE")==0) {
      // connection is closed, one can remove handle

      if (conn && conn->fHandle) {
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn) fWebConn.erase(iter);
      return kTRUE;
   }

   if (strcmp(arg->GetMethod(),"WS_DATA")!=0) {
      Error("ProcessWSRequest","WSHandle DATA request expected!");
      return kFALSE;
   }


   if (!conn) {
      Error("ProcessWSRequest","Get websocket data without valid connection - ignore!!!");
      return kFALSE;
   }

   if (conn->fHandle->PreviewData(arg)) return kTRUE;

   const char* cdata = (arg->GetPostDataLength()<=0) ? "" : (const char*) arg->GetPostData();

   if (strncmp(cdata,"READY",5)==0) {
      conn->fReady = kTRUE;
      CheckModifiedFlag();
   } else if (strncmp(cdata, "RREADY:", 7)==0) {
      conn->fReady = kTRUE;
      CheckModifiedFlag();
   } else if (strncmp(cdata,"GETMENU:",8)==0) {
      conn->fReady = kTRUE;
      conn->fGetMenu = cdata+8;
      CheckModifiedFlag();
   } else if (strncmp(cdata,"GEXE:",5)==0) {
      // TODO: temporary solution, should be removed later
      // used now to terminate ROOT session
      gROOT->ProcessLine(cdata+5);
   } else if (strncmp(cdata,"OBJEXEC:",8)==0) {
      TString buf(cdata+8);
      Int_t pos = buf.First(':');

      if (pos>0) {
         TString id = buf(0,pos);
         buf.Remove(0, pos+1);
         ROOT::Experimental::Internal::TDrawable *drawable = FindDrawable(fCanvas, id.Data());
         if (drawable && (buf.Length()>0)) {
            printf("Execute %s for drawable %p\n", buf.Data(), drawable);
            drawable->Execute(buf.Data());
         }
      }
   }

   return kTRUE;
}

void TCanvasPainter::AddMenuItem(const std::string &name, const std::string &title, const std::string &exec)
{
    ROOT::Experimental::Detail::TMenuItem item(name, title);
    item.SetExec(exec);
    fMenuItems.push_back(item);
}

void TCanvasPainter::AddChkMenuItem(const std::string &name, const std::string &title, bool checked, const std::string &toggle)
{
   ROOT::Experimental::Detail::TMenuItem item(name, title);
   item.SetChecked(checked);
   item.SetExec(toggle);
   fMenuItems.push_back(item);
}

void TCanvasPainter::CheckModifiedFlag()
{

   for (WebConnList::iterator citer = fWebConn.begin(); citer != fWebConn.end(); ++citer) {
      WebConn& conn = *citer;

      if (!conn.fReady || !conn.fHandle) continue;

      TString buf;

      if (!conn.fGetMenu.empty()) {
         ROOT::Experimental::Internal::TDrawable *drawable = FindDrawable(fCanvas, conn.fGetMenu);

         printf("Request menu for object %s found drawable %p\n", conn.fGetMenu.c_str(), drawable);

         if (drawable) {
            fMenuItems.clear();
            drawable->PopulateMenu(*this);

            TClass *cl = gROOT->GetClass("std::vector<ROOT::Experimental::Detail::TMenuItem>");

            // printf("Got items %d class %p %s\n", (int) fMenuItems.size(), cl, cl->GetName());

            // FIXME: got problem with std::list<TMenuItem>, can be generic TBufferJSON
            buf = "MENU";
            buf += TBufferJSON::ConvertToJSON(&fMenuItems, cl);

            fMenuItems.clear();
         }

         conn.fGetMenu = "";
      } else
       if (conn.fModified) {
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

ROOT::Experimental::Internal::TDrawable *TCanvasPainter::FindDrawable(const ROOT::Experimental::TCanvas &can, const std::string &id)
{

   for (auto &&drawable: can.GetPrimitives()) {

      std::string subid = ROOT::Experimental::TDisplayItem::MakeIDFromPtr(&(*drawable));

      if (subid == id) return &(*drawable);
   }

   return nullptr;
}


std::string TCanvasPainter::CreateSnapshot(const ROOT::Experimental::TCanvas& can)
{

   fDisplayList.Clear();

   fDisplayList.SetObjectIDAsPtr((void*)&can);

   TPad *dummy = new TPad(); // just to keep information we need for different ranges now

   auto *snap = new ROOT::Experimental::TUniqueDisplayItem<TPad>(dummy);
   snap->SetObjectIDAsPtr((void*)&can);
   fDisplayList.Add(snap);

   for (auto &&drawable: can.GetPrimitives()) {

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





//void TCanvasPainter::SendCanvas() {
//  for (auto &&drawable: fCanvas.GetPrimitives()) {
//    drawable->Paint(*this);
//  }
//}
