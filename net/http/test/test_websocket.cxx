#include "gtest/gtest.h"

#include <string>
#include <fstream>

#include "THttpServer.h"
#include "THttpWSHandler.h"
#include "TROOT.h"

#include "TSystem.h"
#include "TRandom.h"

#include "ROOT/TestSupport.hxx"

Int_t httpport = 0;
TString server_url;

class TUserHandler : public THttpWSHandler {
   public:
      UInt_t fWSId{0};
      Int_t fServCnt{0};

      TUserHandler(const char *name = nullptr, const char *title = nullptr) : THttpWSHandler(name, title) {}

      // load custom HTML page when open correspondent address
      TString GetDefaultPageContent() override { return "dummy_page_content"; }

      // ignore threads safety - allow multi-threaded use of handler
      Bool_t AllowMTProcess() const override { return kFALSE; }

      Bool_t ProcessWS(THttpCallArg *arg) override
      {
         if (!arg || (arg->GetWSId() == 0))
            return kTRUE;

         if (arg->IsMethod("WS_CONNECT")) {
            // accept only if connection not established
            return fWSId == 0;
         }

         if (arg->IsMethod("WS_READY")) {
            fWSId = arg->GetWSId();
            return kTRUE;
         }

         if (arg->IsMethod("WS_CLOSE")) {
            fWSId = 0;
            return kTRUE;
         }

         if (arg->IsMethod("WS_DATA")) {
            TString str;
            str.Append((const char *)arg->GetPostData(), arg->GetPostDataLength());
            if ((str == "DataRequest") && (arg->GetWSId() == fWSId))
               SendCharStarWS(arg->GetWSId(), "DataResponse");
            else
               SendCharStarWS(arg->GetWSId(), "DataRejected");
            return kTRUE;
         }

         return kFALSE;
      }
};


// main http server
std::string execute_request(const char *url, const char *post = nullptr, Bool_t ws = kFALSE)
{
   TString fname = TString::Format("http_server_%d.output", httpport),
           pname, exec;

   if (post) {
      pname = TString::Format("http_server_%d.post", httpport);
      std::ofstream f(pname.Data());
      f << post;
   }

   if (post && ws)
      exec = TString::Format("curl --no-progress-meter --max-time 10 -T . -N ws://localhost:%d%s < %s > %s", httpport, url, pname.Data(), fname.Data());
   else if (post)
      exec = TString::Format("curl --no-progress-meter -X POST '%s%s' --data-binary @%s -o %s", server_url.Data(), url, pname.Data(), fname.Data());
   else
      exec = TString::Format("curl --no-progress-meter '%s%s' -o %s", server_url.Data(), url, fname.Data());

   printf("Execute %s\n", exec.Data());

   std::string res;

   // websocket ends with timeout - ignore failure for it
   if ((gSystem->Exec(exec) != 0) && !ws)
      res = "<fail>";
   else
      res = THttpServer::ReadFileContent(fname.Data());

   gSystem->Unlink(fname);
   if (!pname.IsNull())
      gSystem->Unlink(pname);

   return res;
}

// main http server
TEST(THttpServer, main)
{
   THttpServer serv("");

   gRandom->SetSeed(0);

   for(int ntry = 0; ntry < 100; ++ntry) {
      Int_t port = (Int_t) (25000 + gRandom->Rndm() * 1000);
      // only two threads, bind to loopback address only
      TString arg = TString::Format("http:%d?loopback&thrds=3", port);
      if (serv.CreateEngine(arg)) {
         httpport = port;
         break;
      }
   }

   EXPECT_NE(httpport, 0);

   if (!httpport)
      return;

   server_url = TString::Format("http:/localhost:%d", httpport);

   TUserHandler handler("ws", "Test WebSocket handler");

   serv.Register("/", &handler);

   // let process requests in separate thread
   serv.CreateServerThread();

   // test reply on html page request
   std::string res = execute_request("/ws/index.htm");
   EXPECT_EQ(res, "dummy_page_content") << "default html page for webcosket handler";

   // test reply on websocket request
   res = execute_request("/ws/root.websocket", "DataRequest", kTRUE);
   EXPECT_EQ(res, "DataResponse") << "check data after websocket communication";
}
