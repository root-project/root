// $Id$
// Author: Sergey Linev   28/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
   std::string buf = THttpServer::ReadFileContent(fname);

   if (buf.empty()) {
      FCGX_FPrintF(request->out,
                   "Status: 404 Not Found\r\n"
                   "Content-Length: 0\r\n" // Always set Content-Length
                   "Connection: close\r\n\r\n");
   } else {

      FCGX_FPrintF(request->out,
                   "Status: 200 OK\r\n"
                   "Content-Type: %s\r\n"
                   "Content-Length: %d\r\n" // Always set Content-Length
                   "\r\n",
                   THttpServer::GetMimeType(fname), (int) buf.length());

      FCGX_PutStr(buf.c_str(), buf.length(), request->out);
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

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

TFastCgi::TFastCgi()
   : THttpEngine("fastcgi", "fastcgi interface to webserver"), fSocket(0), fDebugMode(kFALSE), fTopName(),
     fThrd(nullptr), fTerminating(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TFastCgi::~TFastCgi()
{
   fTerminating = kTRUE;

   if (fThrd) {
      // running thread will be killed
      fThrd->Kill();
      delete fThrd;
      fThrd = nullptr;
   }

   if (fSocket > 0) {
      // close opened socket
      close(fSocket);
      fSocket = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// initializes fastcgi variables and start thread,
/// which will process incoming http requests

Bool_t TFastCgi::Create(const char *args)
{
#ifndef HTTP_WITHOUT_FASTCGI
   FCGX_Init();

   TString sport = ":9000";

   if ((args != 0) && (strlen(args) > 0)) {

      // first extract port number
      sport = ":";
      while ((*args != 0) && (*args >= '0') && (*args <= '9'))
         sport.Append(*args++);

      // than search for extra parameters
      while ((*args != 0) && (*args != '?'))
         args++;

      if (*args == '?') {
         TUrl url(TString::Format("http://localhost/folder%s", args));

         if (url.IsValid()) {

            url.ParseOptions();

            if (url.GetValueFromOptions("debug") != 0)
               fDebugMode = kTRUE;

            const char *top = url.GetValueFromOptions("top");
            if (top != 0)
               fTopName = top;
         }
      }
   }

   Info("Create", "Starting FastCGI server on port %s", sport.Data() + 1);

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

////////////////////////////////////////////////////////////////////////////////

void *TFastCgi::run_func(void *args)
{
#ifndef HTTP_WITHOUT_FASTCGI

   TFastCgi *engine = (TFastCgi *)args;

   FCGX_Request request;

   FCGX_InitRequest(&request, engine->GetSocket(), 0);

   int count = 0;

   while (!engine->fTerminating) {

      int rc = FCGX_Accept_r(&request);

      if (rc != 0)
         continue;

      count++;

      const char *inp_path = FCGX_GetParam("PATH_INFO", request.envp);
      if (!inp_path) inp_path = FCGX_GetParam("SCRIPT_FILENAME", request.envp);
      const char *inp_query = FCGX_GetParam("QUERY_STRING", request.envp);
      const char *inp_method = FCGX_GetParam("REQUEST_METHOD", request.envp);
      const char *inp_length = FCGX_GetParam("CONTENT_LENGTH", request.envp);

      auto arg = std::make_shared<THttpCallArg>();
      if (inp_path)
         arg->SetPathAndFileName(inp_path);
      if (inp_query)
         arg->SetQuery(inp_query);
      if (inp_method)
         arg->SetMethod(inp_method);
      if (engine->fTopName.Length() > 0)
         arg->SetTopName(engine->fTopName.Data());
      int len = 0;
      if (inp_length)
         len = strtol(inp_length, NULL, 10);
      if (len > 0) {
         std::string buf;
         buf.resize(len);
         int nread = FCGX_GetStr((char *)buf.data(), len, request.in);
         if (nread == len)
            arg->SetPostData(std::move(buf));
      }

      TString header;
      for (char **envp = request.envp; *envp != NULL; envp++) {
         TString entry = *envp;
         for (Int_t n = 0; n < entry.Length(); n++)
            if (entry[n] == '=') {
               entry[n] = ':';
               break;
            }
         header.Append(entry);
         header.Append("\r\n");
      }
      arg->SetRequestHeader(header);

      TString username = arg->GetRequestHeader("REMOTE_USER");
      if ((username.Length() > 0) && (arg->GetRequestHeader("AUTH_TYPE").Length() > 0))
         arg->SetUserName(username);

      if (engine->fDebugMode) {
         FCGX_FPrintF(request.out, "Status: 200 OK\r\n"
                                   "Content-type: text/html\r\n"
                                   "\r\n"
                                   "<title>FastCGI echo</title>"
                                   "<h1>FastCGI echo</h1>\n");

         FCGX_FPrintF(request.out, "Request %d:<br/>\n<pre>\n", count);
         FCGX_FPrintF(request.out, "  Method   : %s\n", arg->GetMethod());
         FCGX_FPrintF(request.out, "  PathName : %s\n", arg->GetPathName());
         FCGX_FPrintF(request.out, "  FileName : %s\n", arg->GetFileName());
         FCGX_FPrintF(request.out, "  Query    : %s\n", arg->GetQuery());
         FCGX_FPrintF(request.out, "  PostData : %ld\n", arg->GetPostDataLength());
         FCGX_FPrintF(request.out, "</pre><p>\n");

         FCGX_FPrintF(request.out, "Environment:<br/>\n<pre>\n");
         for (char **envp = request.envp; *envp != NULL; envp++) {
            FCGX_FPrintF(request.out, "  %s\n", *envp);
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

      if (!engine->GetServer()->ExecuteHttp(arg) || arg->Is404()) {
         std::string hdr = arg->FillHttpHeader("Status:");
         FCGX_FPrintF(request.out, hdr.c_str());
      } else if (arg->IsFile()) {
         FCGX_ROOT_send_file(&request, (const char *)arg->GetContent());
      } else {

         // TODO: check in request header that gzip encoding is supported
         if (arg->GetZipping() != THttpCallArg::kNoZip)
            arg->CompressWithGzip();

         std::string hdr = arg->FillHttpHeader("Status:");
         FCGX_FPrintF(request.out, hdr.c_str());

         FCGX_PutStr((const char *)arg->GetContent(), (int)arg->GetContentLength(), request.out);
      }

      FCGX_Finish_r(&request);

   } /* while */

   return nullptr;

#else
   return args;
#endif
}
