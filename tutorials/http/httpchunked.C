/// \file
/// \ingroup tutorial_http
///  This macro shows implementation of chunked requests
///  These are special kind of user-defined requests which let
///  split large http reply on many chunks so that server do not need to allocate
///  such memory at once
///
///  After macro started, try to execute request with
/// ~~~
///      wget http://localhost:8080/chunked.txt
/// ~~~
///
///  CAUTION: Example is not able to handle multiple requests at the same time
///  For this one should create counter for each THttpCallArg instance
///
/// \macro_code
///
/// \author Sergey Linev

#include "THttpServer.h"
#include "THttpCallArg.h"
#include <cstring>

class TChunkedHttpServer : public THttpServer {
   protected:

      int fCounter = 0;

      /** process only requests which are not handled by THttpServer itself */
      void MissedRequest(THttpCallArg *arg) override
      {
         if (strcmp(arg->GetFileName(), "chunked.txt"))
            return;

         arg->SetChunked();

         std::string content = "line" + std::to_string(fCounter++) + "   ";

         for (int n = 0; n < 2000; n++)
            content.append("-");
         content.append("\n");

         if (fCounter >= 1000) {
            // to stop chunk transfer either provide empty content or clear chunked flag
            fCounter = 0;
            arg->SetChunked(kFALSE);
         }

         arg->SetTextContent(std::move(content));
      }

   public:
      TChunkedHttpServer(const char *engine) : THttpServer(engine) {}

   ClassDefOverride(TChunkedHttpServer, 0)
};

void httpchunked()
{
   // start http server
   auto serv = new TChunkedHttpServer("http:8080");

   // reduce to minimum timeout for async requests processing
   serv->SetTimer(1);
}
