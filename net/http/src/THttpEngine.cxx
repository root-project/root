// $Id$
// Author: Sergey Linev   21/12/2013

#include "THttpEngine.h"

#include <string.h>

#include "TCanvas.h"
#include "TClass.h"
#include "TList.h"
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
      buf = "MENU";
      TClass* cl = fCanv->IsA();

      TList* lst = new TList;
      cl->GetMenuItems(lst);
      buf  += TBufferJSON::ConvertToJSON(lst, 3);
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


