// $Id$
// Author: Sergey Linev   21/12/2013

#include "TCivetweb.h"

#include "../civetweb/civetweb.h"

#include <stdlib.h>

#include "THttpServer.h"
#include "TUrl.h"

static int begin_request_handler(struct mg_connection *conn)
{
   TCivetweb *engine = (TCivetweb *) mg_get_request_info(conn)->user_data;
   if (engine == 0) return 0;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return 0;

   const struct mg_request_info *request_info = mg_get_request_info(conn);

   TString filename;

   if (serv->IsFileRequested(request_info->uri, filename)) {
      mg_send_file(conn, filename.Data());
      return 1;
   }

   THttpCallArg arg;
   arg.SetPathAndFileName(request_info->uri); // path and file name
   arg.SetQuery(request_info->query_string);  //! additional arguments
   arg.SetTopName(engine->GetTopName());

   TString hdr;

   if (!serv->ExecuteHttp(&arg) || arg.Is404()) {
      arg.FillHttpHeader(hdr, "HTTP/1.1");
      mg_printf(conn, "%s", hdr.Data());
      return 1;
   }

   if (arg.IsFile()) {
      mg_send_file(conn, (const char *) arg.GetContent());
      return 1;
   }

   arg.FillHttpHeader(hdr, "HTTP/1.1");
   mg_printf(conn, "%s", hdr.Data());

   if (arg.GetContentLength() > 0)
      mg_write(conn, arg.GetContent(), (size_t) arg.GetContentLength());

   // Returning non-zero tells civetweb that our function has replied to
   // the client, and civetweb should not send client any more data.
   return 1;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCivetweb                                                            //
//                                                                      //
// http server implementation, based on civetweb embedded server        //
// It is default kind of engine, created for THttpServer                //
//                                                                      //
// Following additional options can be specified                        //
//    top=foldername - name of top folder, seen in the browser          //
//    thrds=N - use N threads to run civetweb server (default 5)        //
//    auth_file - global authentication file                            //
//    auth_domain - domain name, used for authentication                //
//                                                                      //
// Example:                                                             //
//    new THttpServer("http:8080/none?top=MyApp&thrds=3");              //
//                                                                      //
// Authentication:                                                      //
//    When auth_file and auth_domain parameters are specified, access   //
//    to running http server will be possible only after user           //
//    authentication, using so-call digest method. To generate          //
//    authentication file, htdigest routine should be used:             //
//                                                                      //
//        [shell] htdigest -c .htdigest domain_name user                //
//                                                                      //
//    When creating server, parameters should be:                       //
//                                                                      //
//       new THttpServer("http:8080/none?auth_file=.htdigets&auth_domain=domain_name");   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
TCivetweb::TCivetweb() :
   THttpEngine("civetweb", "compact embedded http server"),
   fCtx(0),
   fCallbacks(0),
   fTopName()
{
   // constructor
}

//______________________________________________________________________________
TCivetweb::~TCivetweb()
{
   // destructor

   if (fCtx != 0) mg_stop((struct mg_context *) fCtx);
   if (fCallbacks != 0) free(fCallbacks);
   fCtx = 0;
   fCallbacks = 0;
}

//______________________________________________________________________________
Bool_t TCivetweb::Create(const char *args)
{
   // Creates embedded civetweb server
   // As argument, http port should be specified in form "8090"

   fCallbacks = malloc(sizeof(struct mg_callbacks));
   memset(fCallbacks, 0, sizeof(struct mg_callbacks));
   ((struct mg_callbacks *) fCallbacks)->begin_request = begin_request_handler;

   TString sport = "8080";
   TString num_threads = "5";
   TString auth_file, auth_domain;

   // for the moment the only argument is port number
   if ((args != 0) && (strlen(args) > 0)) {
      TUrl url(TString::Format("http://localhost:%s", args));

      if (url.IsValid()) {
         url.ParseOptions();
         if (url.GetPort() > 0) sport.Form("%d", url.GetPort());

         const char *top = url.GetValueFromOptions("top");
         if (top != 0) fTopName = top;

         Int_t thrds = url.GetIntValueFromOptions("thrds");
         if (thrds > 0) num_threads.Form("%d", thrds);

         const char *afile = url.GetValueFromOptions("auth_file");
         if (afile != 0) auth_file = afile;
         const char *adomain = url.GetValueFromOptions("auth_domain");
         if (adomain != 0) auth_domain = adomain;
      }
   }

   const char *options[100];
   int op(0);

   Info("Create", "Starting HTTP server on port %s", sport.Data());

   options[op++] = "listening_ports";
   options[op++] = sport.Data();
   options[op++] = "num_threads";
   options[op++] = num_threads.Data();

   if ((auth_file.Length() > 0) && (auth_domain.Length() > 0)) {
      options[op++] = "global_auth_file";
      options[op++] = auth_file.Data();
      options[op++] = "authentication_domain";
      options[op++] = auth_domain.Data();
   }

   options[op++] = 0;

   // Start the web server.
   fCtx = mg_start((struct mg_callbacks *) fCallbacks, this, options);

   return kTRUE;
}

