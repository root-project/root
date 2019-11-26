#include "THttpServer.h"
#include "THttpWSHandler.h"
#include "THttpCallArg.h"
#include "TString.h"
#include "TSystem.h"
#include "TDatime.h"
#include "TTimer.h"

#include <cstdio>

class TUserHandler : public THttpWSHandler {
   public:
      UInt_t fWSId;
      Int_t fServCnt;

      TUserHandler(const char *name = 0, const char *title = 0) : THttpWSHandler(name, title), fWSId(0), fServCnt(0) {}

      // load custom HTML page when open correpondent address
      TString GetDefaultPageContent() { return "file:ws.htm"; }

      virtual Bool_t ProcessWS(THttpCallArg *arg)
      {
         if (!arg || (arg->GetWSId()==0)) return kTRUE;

         // printf("Method %s\n", arg->GetMethod());

         if (arg->IsMethod("WS_CONNECT")) {
            // accept only if connection not established
            return fWSId == 0;
        }

        if (arg->IsMethod("WS_READY")) {
            fWSId = arg->GetWSId();
            printf("Client connected %d\n", fWSId);
            return kTRUE;
        }

        if (arg->IsMethod("WS_CLOSE")) {
           fWSId = 0;
           printf("Client disconnected\n");
           return kTRUE;
        }

        if (arg->IsMethod("WS_DATA")) {
           TString str;
           str.Append((const char *)arg->GetPostData(), arg->GetPostDataLength());
           printf("Client msg: %s\n", str.Data());
           TDatime now;
           SendCharStarWS(arg->GetWSId(), Form("Server replies:%s server counter:%d", now.AsString(), fServCnt++));
           return kTRUE;
        }

        return kFALSE;
      }

      /// per timeout sends data portion to the client
      virtual Bool_t HandleTimer(TTimer *)
      {
         TDatime now;
         if (fWSId) SendCharStarWS(fWSId, Form("Server sends data:%s server counter:%d", now.AsString(), fServCnt++));
         return kTRUE;
      }

};

void ws()
{
   THttpServer *serv = new THttpServer("http:8090");

   TUserHandler *handler = new TUserHandler("name1", "title1");

   serv->Register("/folder1", handler);

   const char *addr = "http://localhost:8090/folder1/name1/";

   printf("Starting browser with URL address %s\n", addr);
   printf("In browser content of ws.htm file should be loaded\n");

   if (gSystem->InheritsFrom("TMacOSXSystem"))
      gSystem->Exec(Form("open %s", addr));
   else
      gSystem->Exec(Form("xdg-open %s &", addr));

   // when connection will be established, data will be send to the client
   TTimer *tm = new TTimer(handler, 3700);
   tm->Start();
}
