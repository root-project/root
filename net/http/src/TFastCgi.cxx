// $Id$
// Author: Sergey Linev   28/12/2013

#include "TFastCgi.h"

#include "TThread.h"
#include "TUrl.h"
#include "THttpServer.h"

#include <string.h>

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifndef HTTP_WITHOUT_FASTCGI

#include "fcgiapp.h"

#include <stdlib.h>

void FCGX_ROOT_send_file(FCGX_Request *request, const char *fname)
{
   Int_t length = 0;

   char *buf = THttpServer::ReadFileContent(fname, length);

   if (buf == 0) {
      FCGX_FPrintF(request->out,
                   "Status: 404 Not Found\r\n"
                   "Content-Length: 0\r\n" // Always set Content-Length
                   "Connection: close\r\n\r\n");
   } else {

      FCGX_FPrintF(request->out,
                   "Status: 200 OK\r\n"
                   "Content-Type: %s\r\n"
                   "Content-Length: %d\r\n"     // Always set Content-Length
                   "\r\n", THttpServer::GetMimeType(fname), length);


      FCGX_PutStr(buf, length, request->out);

      free(buf);
   }
}


#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFastCgi                                                             //
//                                                                      //
// http engine implementation, based on fastcgi package                 //
// Allows to redirect http requests from normal web server like         //
// Apache or lighttpd                                                   //
//                                                                      //
// Configuration example for lighttpd                                   //
//                                                                      //
// server.modules += ( "mod_fastcgi" )                                  //
// fastcgi.server = (                                                   //
//   "/remote_scripts/" =>                                              //
//     (( "host" => "192.168.1.11",                                     //
//        "port" => 9000,                                               //
//        "check-local" => "disable",                                   //
//        "docroot" => "/"                                              //
//     ))                                                               //
// )                                                                    //
//                                                                      //
// When creating THttpServer, one should specify:                       //
//                                                                      //
//  THttpServer* serv = new THttpServer("fastcgi:9000");                //
//                                                                      //
// In this case, requests to lighttpd server will be                    //
// redirected to ROOT session. Like:                                    //
//    http://lighttpdhost/remote_scripts/root.cgi/                      //
//                                                                      //
// Following additional options can be specified                        //
//    top=foldername - name of top folder, seen in the browser          //
//    debug=1 - run fastcgi server in debug mode                        //
// Example:                                                             //
//    serv->CreateEngine("fastcgi:9000?top=fastcgiserver");             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
TFastCgi::TFastCgi() :
   THttpEngine("fastcgi", "fastcgi interface to webserver"),
   fSocket(0),
   fDebugMode(kFALSE),
   fTopName(),
   fThrd(0)
{
   // normal constructor
}

//______________________________________________________________________________
TFastCgi::~TFastCgi()
{
   // destructor

   if (fThrd) {
      // running thread will be killed
      fThrd->Kill();
      delete fThrd;
      fThrd = 0;
   }

   if (fSocket > 0) {
      // close opened socket
      close(fSocket);
      fSocket = 0;
   }
}

//______________________________________________________________________________
Bool_t TFastCgi::Create(const char *args)
{
   // initializes fastcgi variables and start thread,
   // which will process incoming http requests

#ifndef HTTP_WITHOUT_FASTCGI
   FCGX_Init();

//   Info("Create", "Analyze url %s", s.Data());

   TString sport = ":9000";

   if ((args != 0) && (strlen(args) > 0)) {

      // first extract port number
      sport = ":";
      while ((*args!=0) && (*args>='0') && (*args<='9'))
         sport.Append(*args++);

      // than search for extra parameters
      while ((*args!=0) && (*args!='?')) args++;

      if (*args=='?') {
         TUrl url(TString::Format("http://localhost/folder%s", args));

         if (url.IsValid()) {

            url.ParseOptions();

            if (url.GetValueFromOptions("debug") != 0) fDebugMode = kTRUE;

            const char *top = url.GetValueFromOptions("top");
            if (top != 0) fTopName = top;
         }
      }

//      Info("Create", "valid url opt %s debug = %d", url.GetOptions(), fDebugMode);
   }

   Info("Create", "Starting FastCGI server on port %s", sport.Data()+1);

   fSocket = FCGX_OpenSocket(sport.Data(), 10);
   fThrd = new TThread("FastCgiThrd", TFastCgi::run_func, this);
   fThrd->Run();

   return kTRUE;
#else
   (void)args;
   Error("Create", "ROOT compiled without fastcgi support");
   return kFALSE;
#endif
}


//______________________________________________________________________________
void *TFastCgi::run_func(void *args)
{
#ifndef HTTP_WITHOUT_FASTCGI

   TFastCgi *engine = (TFastCgi *) args;

   FCGX_Request request;

   FCGX_InitRequest(&request, engine->GetSocket(), 0);

   int count = 0;

   while (1) {

      int rc = FCGX_Accept_r(&request);

      if (rc != 0) continue;

      count++;

      const char *inp_path = FCGX_GetParam("PATH_INFO", request.envp);
      const char *inp_query = FCGX_GetParam("QUERY_STRING", request.envp);

      THttpCallArg arg;
      if (inp_path != 0) arg.SetPathAndFileName(inp_path);
      if (inp_query != 0) arg.SetQuery(inp_query);
      if (engine->fTopName.Length() > 0) arg.SetTopName(engine->fTopName.Data());

      if (engine->fDebugMode) {
         FCGX_FPrintF(request.out,
                      "Status: 200 OK\r\n"
                      "Content-type: text/html\r\n"
                      "\r\n"
                      "<title>FastCGI echo</title>"
                      "<h1>FastCGI echo</h1>\n"
                      "Request number %d<p>\n", count);

         char *contentLength = FCGX_GetParam("CONTENT_LENGTH", request.envp);
         int len = 0;

         if (contentLength != NULL)
            len = strtol(contentLength, NULL, 10);

         if (len <= 0) {
            FCGX_FPrintF(request.out, "No data from standard input.<p>\n");
         } else {
            int i, ch;

            FCGX_FPrintF(request.out, "Standard input:<br/>\n<pre>\n");
            for (i = 0; i < len; i++) {
               if ((ch = FCGX_GetChar(request.in)) < 0) {
                  FCGX_FPrintF(request.out,
                               "Error: Not enough bytes received on standard input<p>\n");
                  break;
               }
               FCGX_PutChar(ch, request.out);
            }
            FCGX_FPrintF(request.out, "\n</pre><p>\n");
         }

         FCGX_FPrintF(request.out, "PATHNAME: %s<p>\n", arg.GetPathName());
         FCGX_FPrintF(request.out, "FILENAME: %s<p>\n", arg.GetFileName());
         FCGX_FPrintF(request.out, "QUERY:    %s<p>\n", arg.GetQuery());
         FCGX_FPrintF(request.out, "<p>\n");

         FCGX_FPrintF(request.out, "Environment:<br/>\n<pre>\n");
         for (char **envp = request.envp; *envp != NULL; envp++) {
            FCGX_FPrintF(request.out, "%s\n", *envp);
         }
         FCGX_FPrintF(request.out, "</pre><p>\n");

         FCGX_Finish_r(&request);
         continue;
      }

      TString fname;

      if (engine->GetServer()->IsFileRequested(inp_path, fname)) {
         FCGX_ROOT_send_file(&request, fname.Data());
         FCGX_Finish_r(&request);
         continue;
      }

//      printf("PATHNAME %s FILENAME %s QUERY %s \n",
//             arg.GetPathName(), arg.GetFileName(), arg.GetQuery());

      TString hdr;

      if (!engine->GetServer()->ExecuteHttp(&arg) || arg.Is404()) {
         arg.FillHttpHeader(hdr, "Status:");
         FCGX_FPrintF(request.out, hdr.Data());
      } else if (arg.IsFile()) {
         FCGX_ROOT_send_file(&request, (const char *) arg.GetContent());
      } else {

         // TODO: check in request header that gzip encoding is supported
         if (arg.GetZipping()>0) arg.CompressWithGzip();

         arg.FillHttpHeader(hdr, "Status:");
         FCGX_FPrintF(request.out, hdr.Data());

         FCGX_PutStr((const char *) arg.GetContent(),
                     (int) arg.GetContentLength(), request.out);
      }

      FCGX_Finish_r(&request);

   } /* while */

   return 0;

#else
   return args;
#endif
}

