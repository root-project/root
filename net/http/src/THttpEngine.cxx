// $Id$
// Author: Sergey Linev   21/12/2013

#include "THttpEngine.h"

#include <string.h>

#include "TCanvas.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TList.h"
#include "TROOT.h"
#include "THttpCallArg.h"
#include "TBufferJSON.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpEngine                                                          //
//                                                                      //
// Abstract class for implementing http protocol for THttpServer        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(THttpEngine)

//______________________________________________________________________________
THttpEngine::THttpEngine(const char *name, const char *title) :
   TNamed(name, title),
   fServer(0)
{
   // normal constructor
}

//______________________________________________________________________________
THttpEngine::~THttpEngine()
{
   // destructor

   fServer = 0;
}


ClassImp(THttpWSEngine)

//______________________________________________________________________________
THttpWSEngine::THttpWSEngine(const char* name, const char* title) :
   TNamed(name, title),
   fReady(kFALSE),
   fModified(kFALSE),
   fGetMenu(kFALSE),
   fCanv(0)
{
}

//______________________________________________________________________________
THttpWSEngine::~THttpWSEngine()
{
   AssignCanvas(0);
}

//______________________________________________________________________________
void THttpWSEngine::CanvasModified()
{
//   printf("Canvas modified\n");
   fModified = kTRUE;
   CheckModifiedFlag();
}

//______________________________________________________________________________
void THttpWSEngine::CheckModifiedFlag()
{
   if (!fReady || !fCanv) return;

   TString buf;

   if (fGetMenu) {
      TClass* cl = fCanv->IsA();

      TList* lst = new TList;
      cl->GetMenuItems(lst);
      // while there is no streamer for TMethod class, one needs own implementation

      // TBufferJSON::ConvertToJSON(lst, 3);

      TIter iter(lst);
      TMethod* m = 0;
      Int_t cnt = 0;

      buf = "MENU[";
      while ((m = (TMethod*) iter()) != 0) {
         if (cnt++ > 0) buf.Append(",");
         buf.Append("{");
         buf.Append(TString::Format("\"name\":\"%s\",\"title\":\"%s\"", m->GetName(), m->GetTitle()));

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
                 call->Execute(fCanv, l);
                 buf.Append(TString::Format(",\"chk\":%s", (l!=0) ? "true" : "false"));
                 buf.Append(TString::Format(",\"exec\":\"%s(%s)\"", m->GetName(), (l!=0) ? "0" : "1"));
                 // printf("Toggle %s getter %s chk: %s \n", m->GetName(), getter.Data(), (l!=0) ? "true" : "false");
               } else {
                 printf("Cannot get toggle value with getter %s \n", getter.Data());
               }
            }
         } else {
            buf.Append(TString::Format(",\"exec\":\"%s()\"", m->GetName()));
         }

         buf.Append("}");
      }
      buf  += "]";
      delete lst;

      fGetMenu = kFALSE;
   } else
   if (fModified) {
      buf = "JSON";
      buf  += TBufferJSON::ConvertToJSON(fCanv, 3);
      fModified = kFALSE;
   }

   if (buf.Length() > 0) {
      fReady = kFALSE;
      Send(buf.Data(), buf.Length());
   }
}


//______________________________________________________________________________
void THttpWSEngine::ProcessData(THttpCallArg* arg)
{
   if ((arg==0) && (arg->GetPostDataLength()<=0)) return;

   const char* cdata = (const char*) arg->GetPostData();

   if (strncmp(cdata,"READY",5)==0) {
      fReady = kTRUE;
      CheckModifiedFlag();
      return;
   }

   if (strncmp(cdata,"GETMENU",7)==0) {
      fGetMenu = kTRUE;
      CheckModifiedFlag();
      return;
   }

   if (strncmp(cdata,"EXEC",4)==0) {

      if (fCanv!=0) {

         TString exec;
         exec.Form("((%s*) %p)->%s;", fCanv->ClassName(), fCanv, cdata+4);
         // printf("Execute %s\n", exec.Data());

         gROOT->ProcessLine(exec);
      }

      return;
   }

}

//______________________________________________________________________________
void THttpWSEngine::AssignCanvas(TCanvas* canv)
{

   if (fCanv != 0) {
      fCanv->Disconnect("Modified()", this, "CanvasModified()");
      fCanv->GetListOfPrimitives()->Remove(this);
      fCanv = 0;
   }

   if (canv != 0) {
      SetName("websocket");
      canv->Connect("Modified()", "THttpWSEngine", this, "CanvasModified()");
      canv->GetListOfPrimitives()->Add(this);
      fCanv = canv;
   }

}


