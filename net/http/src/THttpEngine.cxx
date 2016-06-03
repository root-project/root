// $Id$
// Author: Sergey Linev   21/12/2013

#include "THttpEngine.h"

#include <string.h>

#include "TCanvas.h"
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
   if (!fModified || !fReady || !fCanv) return;

   TString buf = "JSON";
   buf  += TBufferJSON::ConvertToJSON(fCanv, 3);

   fModified = kFALSE;
   fReady = kFALSE;

   Send(buf.Data(), buf.Length());
}


//______________________________________________________________________________
void THttpWSEngine::ProcessData(THttpCallArg* arg)
{
   if ((arg==0) && (arg->GetPostDataLength()<=0)) return;

   const char* cdata = (const char*) arg->GetPostData();

   if (strncmp(cdata,"READY",5)==0) {
      fReady = kTRUE;
      printf("GET READY FLAG\n");
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


