// Author: Sergey Linev   7/12/2016

/*************************************************************************
 * Copyright (C) 2016, Sergey Linev                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebCanvas.h"

#include "THttpCallArg.h"
#include "THttpEngine.h"
#include "TWebSnapshot.h"
#include "TWebPadPainter.h"
#include "TWebVirtualX.h"
#include "TWebMenuItem.h"

#include "TSystem.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TList.h"
#include "TBufferJSON.h"
#include "Riostream.h"

#include <stdio.h>
#include <string.h>

#ifdef WEBGUI_WITH_CEF
#include "../cef/simple_app.h"
#include "TApplication.h"
#include "TTimer.h"

class TCefTimer : public TTimer {
public:
   TCefTimer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode)
   {
      // construtor
   }
   virtual ~TCefTimer()
   {
      // destructor
   }
   virtual void Timeout()
   {
      // just dummy workaround
      CefDoMessageLoopWork();
   }
};


#endif

ClassImp(TWebCanvas)

TWebCanvas::TWebCanvas() :
   THttpWSHandler("", ""),
   TCanvasImp(),
   fWebConn(),
   fAddr(),
   fHasSpecials(kFALSE)
{
}

TWebCanvas::TWebCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height, TString addr) :
   THttpWSHandler(name, "title"),
   TCanvasImp(c, name, x, y, width, height),
   fWebConn(),
   fAddr(addr),
   fHasSpecials(kFALSE)
{
   UInt_t hash = TString::Hash(&c, sizeof(c));
   SetName(Form("0x%u", hash)); // make name very screwed
}

TWebCanvas::~TWebCanvas()
{
   for (WebConnList::iterator iter = fWebConn.begin(); iter != fWebConn.end(); ++iter) {
      if (iter->fHandle) {
         iter->fHandle->ClearHandle();
         delete iter->fHandle;
         iter->fHandle = 0;
      }
   }
}

Int_t TWebCanvas::InitWindow()
{
   TWebVirtualX *vx = dynamic_cast<TWebVirtualX *> (gVirtualX);
   if (vx) vx->SetWebCanvasSize(Canvas()->GetWw(), Canvas()->GetWh());

   // at this place canvas is not yet register to the list of canvases - we cannot start browser
   return 777111777; // magic number, should be catch by TWebVirtualX
}

TVirtualPadPainter* TWebCanvas::CreatePadPainter()
{
   return new TWebPadPainter();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE when object is fully supported on JSROOT side
/// In ROOT7 Paint function will just return appropriate flag that object can be displayed on JSROOT side

Bool_t TWebCanvas::IsJSSupportedClass(TObject* obj)
{
   if (!obj) return kTRUE;

   if (obj->InheritsFrom("TH1") || obj->InheritsFrom("TGraph") || obj->InheritsFrom("TPave") || obj->InheritsFrom("TArrow")) return kTRUE;

   return kFALSE;
}

TObject* TWebCanvas::FindPrimitive(UInt_t id, TPad *pad)
{
   // search of object with given id in list of primitives

   if (!pad) pad = Canvas();

   if (TString::Hash(&pad, sizeof(pad)) == id) return pad;

   TIter iter(pad->GetListOfPrimitives());
   TObject* obj = 0;
   while ((obj = iter()) != 0) {
      if (obj->InheritsFrom(TPad::Class())) obj = FindPrimitive(id, (TPad*) obj);
      else if (TString::Hash(&obj, sizeof(obj)) != id) obj = 0;

      if (obj) return obj;
   }

   return 0;
}


TPadWebSnapshot* TWebCanvas::CreateSnapshot(TPad* pad)
{
   TPadWebSnapshot *lst = new TPadWebSnapshot();

   TWebSnapshot* padshot = new TWebSnapshot();

   padshot->SetObjectIDAsPtr(pad);
   padshot->SetSnapshot(TWebSnapshot::kObject, pad);
   lst->Add(padshot);

   TIter iter(pad->GetListOfPrimitives());
   TObject* obj = 0;
   while ((obj = iter()) != 0) {
      if (obj->InheritsFrom(TPad::Class())) {
         TWebSnapshot* sub = CreateSnapshot((TPad*) obj);
         sub->SetObjectIDAsPtr(obj);
         lst->Add(sub);
      } else {
         TWebSnapshot *sub = new TWebSnapshot();
         sub->SetObjectIDAsPtr(obj);
         sub->SetOption(iter.GetOption());
         TWebPainting *p = 0;

         if (!IsJSSupportedClass(obj)) {
            TWebPadPainter *painter = dynamic_cast<TWebPadPainter *> (Canvas()->GetCanvasPainter());
            if (painter) painter->ResetPainting(); // ensure painter is created
            // call paint function of object itself
            obj->Paint(iter.GetOption());
            if (painter) p = painter->TakePainting();
            fHasSpecials = kTRUE;
         }

         // when paint method was used and resultative,

         if (p) {
            p->FixSize();
            sub->SetSnapshot(TWebSnapshot::kSVG, p);
         } else {
            sub->SetSnapshot(TWebSnapshot::kObject, obj);
         }

         lst->Add(sub);
      }
   }

   return lst;
}



void TWebCanvas::CheckModifiedFlag()
{
   if (!Canvas()) return;


   for (WebConnList::iterator citer = fWebConn.begin(); citer != fWebConn.end(); ++citer) {
      WebConn& conn = *citer;

      if (!conn.fReady || !conn.fHandle) continue;

      TString buf;

      if (conn.fGetMenu) {

         TObject* obj = FindPrimitive(conn.fGetMenu);
         if (!obj) obj = Canvas();

         TClass* cl = obj->IsA();

         TList* lst = new TList;
         cl->GetMenuItems(lst);
         // while there is no streamer for TMethod class, one needs own implementation

         // TBufferJSON::ConvertToJSON(lst, 3);

         TIter iter(lst);
         TMethod* m = 0;

         std::vector<TWebMenuItem> items;

         while ((m = (TMethod*) iter()) != 0) {

            TWebMenuItem item(m->GetName(), m->GetTitle());

            if (m->IsMenuItem() == kMenuToggle) {
               TString getter;
               if (m->Getter() && strlen(m->Getter()) > 0) {
                  getter = m->Getter();
               } else
                  if (strncmp(m->GetName(),"Set",3)==0) {
                     getter = TString(m->GetName())(3, strlen(m->GetName())-3);
                     if (cl->GetMethodAllAny(TString("Has") + getter)) getter = TString("Has") + getter;
                     else if (cl->GetMethodAllAny(TString("Get") + getter)) getter = TString("Get") + getter;
                     else if (cl->GetMethodAllAny(TString("Is") + getter)) getter = TString("Is") + getter;
                     else getter = "";
                  }

               if ((getter.Length()>0) && cl->GetMethodAllAny(getter)) {

                  TMethodCall* call = new TMethodCall(cl, getter, "");

                  if (call->ReturnType() == TMethodCall::kLong) {
                     Long_t l(0);
                     call->Execute(obj, l);
                     item.SetChecked(l!=0);
                     item.SetExec(Form("%s(%s)", m->GetName(), (l!=0) ? "0" : "1"));
                  } else {
                     Error("CheckModifiedFlag", "Cannot get toggle value with getter %s", getter.Data());
                  }

                  delete call;
               }
            } else {
               item.SetExec(Form("%s()", m->GetName()));
            }

            items.push_back(item);
         }

         buf = "MENU";
         buf += TBufferJSON::ConvertToJSON(&items, gROOT->GetClass("std::list<TWebMenuItem>"));

         printf("%s\n", buf.Data());

         delete lst;

         conn.fGetMenu = 0;
      } else
      if (conn.fModified) {
         // buf = "JSON";
         // buf  += TBufferJSON::ConvertToJSON(Canvas(), 3);

         buf = "SNAP";
         TPadWebSnapshot *snapshot = CreateSnapshot(Canvas());
         buf  += TBufferJSON::ConvertToJSON(snapshot, 23);
         delete snapshot;
         conn.fModified = kFALSE;
      }

      if (buf.Length() > 0) {
         // sending of data can be moved into separate thread - not to block user code
         conn.fReady = kFALSE;
         conn.fHandle->SendCharStar(buf.Data());
      }
   }
}

void TWebCanvas::Close()
{
   printf("Call TWebCanvas::Close\n");
}

void TWebCanvas::Show()
{
   TString addr;

   addr.Form("%s/webgui/%s/draw.htm?webcanvas", fAddr.Data(), GetName());
   // addr.Form("http://localhost:8080/Canvases/%s/draw.htm?longpollcanvas", Canvas()->GetName());

   Func_t symbol = gSystem->DynFindSymbol("*", "webgui_start_browser_new");
   if (symbol) {
      typedef void (*FunctionFunc)(const char*);

      addr.Form("example://dummy:8080/webgui/%s/draw.htm?longpollcanvas", GetName());
      //addr.Form("example://localhost:8080/Canvases/%s/draw.htm", Canvas()->GetName());

      Info("Show", "Call TWebCanvas::Show:  %s", addr.Data());

      FunctionFunc func = (FunctionFunc) symbol;
      func(addr.Data());
      return;
   }

   Info("Show", "Call TWebCanvas::Show:  %s", addr.Data());

   #ifdef WEBGUI_WITH_CEF
      const char* cef = gSystem->Getenv("CEF_PATH");
      Info("Show", "CEF %s", cef);
      if (cef && !gSystem->AccessPathName(cef)) {

         TApplication* root_app = gROOT->GetApplication();

         printf("Start cef window with addr %s app %p\n", addr.Data(), root_app);

         CefMainArgs main_args(root_app->Argc(), root_app->Argv());

         // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
         // that share the same executable. This function checks the command-line and,
         // if this is a sub-process, executes the appropriate logic.

/*         int exit_code = CefExecuteProcess(main_args, NULL, NULL);
         if (exit_code >= 0) {
           // The sub-process has completed so return here.
           return exit_code;
         }
*/

         // Install xlib error handlers so that the application won't be terminated
         // on non-fatal errors.
//         XSetErrorHandler(XErrorHandlerImpl);
//         XSetIOErrorHandler(XIOErrorHandlerImpl);

         // Specify CEF global settings here.
         CefSettings settings;

       //  settings.multi_threaded_message_loop = false; // not supported
       //  settings.external_message_pump = false;
         const char* rootsys = gSystem->Getenv("ROOTSYS");

         printf("Get build dir %s\n", rootsys);

         TString path, path2, cef_main;
         path.Form("%s/Resources/", cef);
         path2.Form("%s/Resources/locales/", cef);
         cef_main.Form("%s/bin/cef_main", rootsys);

          cef_string_ascii_to_utf16(path.Data(), path.Length(), &settings.resources_dir_path);

          cef_string_ascii_to_utf16(path2.Data(), path2.Length(), &settings.locales_dir_path);

          settings.no_sandbox = true;
          // settings.single_process = true;


         // SimpleApp implements application-level callbacks for the browser process.
         // It will create the first browser instance in OnContextInitialized() after
         // CEF has initialized.
         CefRefPtr<SimpleApp> *app = new CefRefPtr<SimpleApp>(new SimpleApp(addr.Data(), cef_main.Data()));

         // Initialize CEF for the browser process.
         CefInitialize(main_args, settings, app->get(), NULL);


         // let run event loop, should be improved later
         TCefTimer *timer = new TCefTimer(10, kTRUE);
         timer->TurnOn();


         // Run the CEF message loop. This will block until CefQuitMessageLoop() is
         // called.
         // CefRunMessageLoop();

       /*  int cnt = 0;
         while (true) {
           CefDoMessageLoopWork();
           ::sleep(1);
         }
       */
         // printf("doing shutdown\n");

         // Shut down CEF.
         //CefShutdown();



         return;
      }
   #endif

   // exec.Form("setsid chromium --app=http://localhost:8080/Canvases/%s/draw.htm?websocket </dev/null >/dev/null 2>/dev/null &", Canvas()->GetName());

   TString exec;
   if (gSystem->InheritsFrom("TMacOSXSystem"))
      exec.Form("open %s", addr.Data());
   else
      exec.Form("xdg-open %s &", addr.Data());
   printf("Exec %s\n", exec.Data());

   gSystem->Exec(exec);
}

Bool_t TWebCanvas::DecodePadRanges(TPad *pad, const char *arg)
{
   if (!arg || !*arg) return kFALSE;

   Double_t ux1,ux2,uy1,uy2,px1,px2,py1,py2;
   Int_t cnt = sscanf(arg, "%lf:%lf:%lf:%lf:%lf:%lf:%lf:%lf",&ux1,&ux2,&uy1,&uy2,&px1,&px2,&py1,&py2);
   if (cnt!=8) return kFALSE;

   Double_t ux1_,ux2_,uy1_,uy2_,px1_,px2_,py1_,py2_;

   pad->GetRange(px1_,py1_,px2_,py2_);
   pad->GetRangeAxis(ux1_,uy1_,ux2_,uy2_);

   if ((ux1==ux1_) && (ux2==ux2_) && (uy1==uy1_) && (uy2==uy2_) && (px1==px1_) && (px2==px2_) && (py1==py1_) && (py2==py2_)) {
      //Info("DecodePadRanges","Ranges not changed");
      return kFALSE;
   }

   pad->Range(px1,py1,px2,py2);
   pad->RangeAxis(ux1,uy1,ux2,uy2);

   Info("GetRanges", "Apply new ranges %s for pad %s", arg, pad->GetName());

   // without special objects no need for explicit update of the canvas
   if (!fHasSpecials) return kFALSE;

   pad->Modified(kTRUE);
   return kTRUE;
}

Bool_t TWebCanvas::DecodeAllRanges(const char *arg)
{
  if (!arg || !*arg) return kFALSE;
  Bool_t isany = kFALSE;

  const char *curr = arg, *pos = 0;

  while ((pos = strstr(curr, "id=")) != 0) {
     unsigned int id = 0;
     curr = pos + 3;
     if (sscanf(curr,"%u", &id)!=1) break;
     curr = strstr(curr,":");
     if (!curr) break;

     TPad *pad = dynamic_cast<TPad *>(FindPrimitive(id));

     if (pad && DecodePadRanges(pad, curr+1)) isany = kTRUE;
  }

  if (isany) PerformUpdate();
  return kTRUE;
}


Bool_t TWebCanvas::ProcessWS(THttpCallArg *arg)
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

   } else
   if (strcmp(arg->GetMethod(),"WS_READY")==0) {
      THttpWSEngine* wshandle = dynamic_cast<THttpWSEngine*> (arg->TakeWSHandle());

      if (conn != 0) Error("ProcessWSRequest","WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fHandle = wshandle;
      newconn.fModified = kTRUE;

      fWebConn.push_back(newconn);

      // if (gDebug>0) Info("ProcessRequest","Set WebSocket handle %p", wshandle);

      // connection is established
   } else
   if (strcmp(arg->GetMethod(),"WS_DATA")==0) {
      // process received data

      if (!conn) {
         Error("ProcessWSRequest","Get websocket data without valid connection - ignore!!!");
         return kFALSE;
      }

      if (conn->fHandle->PreviewData(arg)) return kTRUE;

      const char* cdata = (arg->GetPostDataLength()<=0) ? "" : (const char*) arg->GetPostData();

      if (strncmp(cdata,"READY",5)==0) {
         conn->fReady = kTRUE;
         CheckModifiedFlag();
      } else
      if (strncmp(cdata, "RREADY:", 7)==0) {
         conn->fReady = kTRUE;
         if (!DecodeAllRanges(cdata+7)) CheckModifiedFlag();
      } else
      if (strncmp(cdata,"GETMENU:",8)==0) {
         conn->fReady = kTRUE;
         conn->fGetMenu = (UInt_t) TString(cdata+8).Atoll();
         CheckModifiedFlag();
      } else
      if (strncmp(cdata,"GETMENU",7)==0) {
         void *ptr = this;
         conn->fReady = kTRUE;
         conn->fGetMenu = TString::Hash(ptr, sizeof(void *));
         CheckModifiedFlag();
      } else
      if (strncmp(cdata,"OBJEXEC:",8)==0) {
         TString buf(cdata+8);
         Int_t pos = buf.First(':');

         if (pos>0) {
            UInt_t id = (UInt_t) TString(buf(0,pos)).Atoll();
            buf.Remove(0, pos+1);

            TObject* obj = FindPrimitive(id);
            if (obj && (buf.Length()>0)) {
               TString exec;
               exec.Form("((%s*) %p)->%s;", obj->ClassName(), obj, buf.Data());
               Info("ProcessWSRequest", "Obj %s Execute %s", obj->GetName(), exec.Data());
               gROOT->ProcessLine(exec);
            }
         }
      } else
      if (strncmp(cdata,"EXEC:",5)==0) {
         if (Canvas()!=0) {
            TString exec;
            exec.Form("((%s*) %p)->%s;", Canvas()->ClassName(), Canvas(), cdata+5);
            gROOT->ProcessLine(exec);
         }
      } else
      if (strncmp(cdata,"GEXE:",5)==0) {
         gROOT->ProcessLine(cdata+5); // temporary solution, will be removed later
      } else
      if (strncmp(cdata,"GETIMG:",7)==0) {
         const char* img = cdata+7;

         const char* separ = strchr(img,':');
         if (separ) {
            TString filename(img, separ-img);
            img = separ+1;
            filename.Append(".svg"); // temporary - JSROOT returns SVG

            std::ofstream ofs(filename);
            ofs << "<?xml version=\"1.0\" standalone=\"no\"?>";
            ofs << img;
            ofs.close();

            Info("ProcessWSRequest", "SVG file %s has been created", filename.Data());
         }
         conn->fReady = kTRUE;
         CheckModifiedFlag();
      } else {
         Error("ProcessWSRequest", "GET unknown request %d %s", (int) strlen(cdata), cdata);
      }

   } else
   if (strcmp(arg->GetMethod(),"WS_CLOSE")==0) {
      // connection is closed, one can remove handle

      if (conn && conn->fHandle) {
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn) fWebConn.erase(iter);
   }

   return kTRUE;
}

Bool_t TWebCanvas::IsAnyPadModified(TPad *pad)
{
   // returns true when any pad or sub pad modified
   // reset modified flags

   Bool_t res = kFALSE;

   if (pad->IsModified()) {
      pad->Modified(kFALSE);
      res = kTRUE;
   }

   TIter iter(pad->GetListOfPrimitives());
   TObject* obj = 0;
   while ((obj = iter()) != 0) {
      if (obj->InheritsFrom(TPad::Class()) && IsAnyPadModified((TPad*) obj)) res = kTRUE;
   }

   return res;
}

UInt_t TWebCanvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // reset dimension in gVirtualX  - it will be requested immediately
   TWebVirtualX *vx = dynamic_cast<TWebVirtualX *> (gVirtualX);
   if (vx) vx->SetWebCanvasSize(Canvas()->GetWw(), Canvas()->GetWh());

   x = 0; y = 0;
   w = Canvas()->GetWw() + 4;
   h = Canvas()->GetWh() + 28;
   return 0;
}

Bool_t TWebCanvas::PerformUpdate()
{
   // check if canvas modified. If true and communication allowed,
   // It scan all primitives in the TCanvas and subpads and convert them into
   // the structure which will be delivered to JSROOT client

   if (IsAnyPadModified(Canvas())) {
      for (WebConnList::iterator iter = fWebConn.begin(); iter != fWebConn.end(); ++iter)
         iter->fModified = kTRUE;
   }

   CheckModifiedFlag();

   return kTRUE;
}
