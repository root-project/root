// $Id$
// Author: Sergey Linev   21/12/2013

#include "TCivetweb.h"

#include "../civetweb/civetweb.h"

#include <stdlib.h>
#include <string.h>

#include "THttpServer.h"
#include "TUrl.h"

#include <string>

static int log_message_handler(const struct mg_connection *conn, const char *message)
{
   const struct mg_context *ctx = mg_get_context(conn);

   TCivetweb* engine = (TCivetweb*) mg_get_user_data(ctx);

   if (engine) return engine->ProcessLog(message);

   // provide debug output
   if ((gDebug>0) || (strstr(message,"cannot bind to")!=0))
      fprintf(stderr, "Error in <TCivetweb::Log> %s\n",message);

   return 0;
}


static int begin_request_handler(struct mg_connection *conn, void*)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   TCivetweb *engine = (TCivetweb *) request_info->user_data;
   if (engine == 0) return 0;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return 0;

   THttpCallArg arg;

   TString filename;

   Bool_t execres = kTRUE, debug = engine->IsDebugMode();

   if (!debug && serv->IsFileRequested(request_info->uri, filename)) {
      if ((filename.Index(".js") != kNPOS) || (filename.Index(".css") != kNPOS)) {
         Int_t length = 0;
         char *buf = THttpServer::ReadFileContent(filename.Data(), length);
         if (buf == 0) {
            arg.Set404();
         } else {
            arg.SetContentType(THttpServer::GetMimeType(filename.Data()));
            arg.SetBinData(buf, length);
            arg.AddHeader("Cache-Control", "max-age=3600");
            arg.SetZipping(2);
         }
      } else {
         arg.SetFile(filename.Data());
      }
   } else {
      arg.SetPathAndFileName(request_info->uri); // path and file name
      arg.SetQuery(request_info->query_string);  // query arguments
      arg.SetTopName(engine->GetTopName());
      arg.SetMethod(request_info->request_method); // method like GET or POST
      if (request_info->remote_user!=0)
         arg.SetUserName(request_info->remote_user);

      TString header;
      for (int n = 0; n < request_info->num_headers; n++)
         header.Append(TString::Format("%s: %s\r\n", request_info->http_headers[n].name, request_info->http_headers[n].value));
      arg.SetRequestHeader(header);

      const char* len = mg_get_header(conn, "Content-Length");
      Int_t ilen = len!=0 ? TString(len).Atoi() : 0;

      if (ilen>0) {
         void* buf = malloc(ilen+1); // one byte more for null-termination
         Int_t iread = mg_read(conn, buf, ilen);
         if (iread==ilen) arg.SetPostData(buf, ilen);
                     else free(buf);
      }

      if (debug) {
         TString cont;
         cont.Append("<title>Civetweb echo</title>");
         cont.Append("<h1>Civetweb echo</h1>\n");

         static int count = 0;

         cont.Append(TString::Format("Request %d:<br/>\n<pre>\n", ++count));
         cont.Append(TString::Format("  Method   : %s\n", arg.GetMethod()));
         cont.Append(TString::Format("  PathName : %s\n", arg.GetPathName()));
         cont.Append(TString::Format("  FileName : %s\n", arg.GetFileName()));
         cont.Append(TString::Format("  Query    : %s\n", arg.GetQuery()));
         cont.Append(TString::Format("  PostData : %ld\n", arg.GetPostDataLength()));
         if (arg.GetUserName())
         cont.Append(TString::Format("  User     : %s\n", arg.GetUserName()));

         cont.Append("</pre><p>\n");

         cont.Append("Environment:<br/>\n<pre>\n");
         for (int n = 0; n < request_info->num_headers; n++)
            cont.Append(TString::Format("  %s = %s\n", request_info->http_headers[n].name, request_info->http_headers[n].value));
         cont.Append("</pre><p>\n");

         arg.SetContentType("text/html");

         arg.SetContent(cont);

      } else {
         execres = serv->ExecuteHttp(&arg);
      }
   }

   if (!execres || arg.Is404()) {
      TString hdr;
      arg.FillHttpHeader(hdr, "HTTP/1.1");
      mg_printf(conn, "%s", hdr.Data());
   } else if (arg.IsFile()) {
      mg_send_file(conn, (const char *) arg.GetContent());
   } else {

      Bool_t dozip = arg.GetZipping() > 0;
      switch (arg.GetZipping()) {
         case 2:
            if (arg.GetContentLength() < 10000) {
               dozip = kFALSE;
               break;
            }
         case 1:
            // check if request header has Accept-Encoding
            dozip = kFALSE;
            for (int n = 0; n < request_info->num_headers; n++) {
               TString name = request_info->http_headers[n].name;
               if (name.Index("Accept-Encoding", 0, TString::kIgnoreCase) != 0) continue;
               TString value = request_info->http_headers[n].value;
               dozip = (value.Index("gzip", 0, TString::kIgnoreCase) != kNPOS);
               break;
            }

            break;
         case 3:
            dozip = kTRUE;
            break;
      }

      if (dozip) arg.CompressWithGzip();

      TString hdr;
      arg.FillHttpHeader(hdr, "HTTP/1.1");
      mg_printf(conn, "%s", hdr.Data());

      if (arg.GetContentLength() > 0)
         mg_write(conn, arg.GetContent(), (size_t) arg.GetContentLength());
   }

   // Returning non-zero tells civetweb that our function has replied to
   // the client, and civetweb should not send client any more data.
   return 1;
}

//_____________________________________________________________________

class TCivetwebWSEngine : public THttpWSEngine {
   protected:

      struct mg_connection *fWSconn;

   public:

      TCivetwebWSEngine(const char* name, const char* title, struct mg_connection *conn) :
         THttpWSEngine(name, title),
         fWSconn(conn)
      {
      }

      virtual ~TCivetwebWSEngine()
      {
      }

      virtual void ClearHandle()
      {
         fWSconn = 0;
      }

      virtual void Send(const void* buf, int len)
      {
         if (fWSconn)
            mg_websocket_write(fWSconn, WEBSOCKET_OPCODE_TEXT, (const char*) buf, len);
      }

};


int websocket_connect_handler(const struct mg_connection *conn, void*)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);
   if (request_info == 0) return 1;

   // printf("Request websocket for uri:%s\n", request_info->uri);

   TCivetweb *engine = (TCivetweb *) request_info->user_data;
   if (engine == 0) return 1;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return 1;

   THttpCallArg arg;
   arg.SetPathAndFileName(request_info->uri); // path and file name
   arg.SetQuery(request_info->query_string);  // query arguments
   arg.SetMethod("WS_CONNECT");

   Bool_t execres = serv->ExecuteHttp(&arg);

   // printf("res %d 404 %d\n", execres, arg.Is404());

   return execres && !arg.Is404() ? 0 : 1;
}

void websocket_ready_handler(struct mg_connection *conn, void*)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   // printf("Websocket connection established url:%s\n", request_info->uri);

   TCivetweb *engine = (TCivetweb *) request_info->user_data;
   if (engine == 0) return;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return;

   THttpCallArg arg;
   arg.SetPathAndFileName(request_info->uri); // path and file name
   arg.SetQuery(request_info->query_string);  // query arguments
   arg.SetMethod("WS_READY");

   arg.SetWSHandle(new TCivetwebWSEngine("websocket", "title", conn));

   serv->ExecuteHttp(&arg);
}

//static int wscnt = 0;

int websocket_data_handler(struct mg_connection *conn, int, char *data, size_t len, void*)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   TCivetweb *engine = (TCivetweb *) request_info->user_data;
   if (engine == 0) return 1;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return 1;

   THttpCallArg arg;
   arg.SetPathAndFileName(request_info->uri); // path and file name
   arg.SetQuery(request_info->query_string);  // query arguments
   arg.SetMethod("WS_DATA");

   void* buf = malloc(len+1); // one byte more for null-termination
   memcpy(buf, data, len);
   arg.SetPostData(buf, len);

   //if ((bits & 0xF) == 1)
   //   printf("Get string len %d %s\n", (int) len, (char*) arg.GetPostData());
   //else
   //   printf("Get data from web socket len bits %d %d\n", bits, (int) len);

   serv->ExecuteHttp(&arg);

   //if (++wscnt >= 20000) {
   //   const char* reply = "Send close message";
   //   mg_websocket_write(conn, WEBSOCKET_OPCODE_CONNECTION_CLOSE, reply, strlen(reply));
   //}

   return 1;
}

void websocket_close_handler(const struct mg_connection *conn, void*)
{
   const struct mg_request_info *request_info = mg_get_request_info(conn);

   // printf("Websocket connection closed url:%s\n", request_info->uri);
   // wscnt = 0;

   TCivetweb *engine = (TCivetweb *) request_info->user_data;
   if (engine == 0) return;
   THttpServer *serv = engine->GetServer();
   if (serv == 0) return;

   THttpCallArg arg;
   arg.SetPathAndFileName(request_info->uri); // path and file name
   arg.SetQuery(request_info->query_string);  // query arguments
   arg.SetMethod("WS_CLOSE");

   serv->ExecuteHttp(&arg);
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
//    new THttpServer("http:8080?top=MyApp&thrds=3");                   //
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
//       new THttpServer("http:8080?auth_file=.htdigets&auth_domain=domain_name");  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(TCivetweb)

//______________________________________________________________________________
TCivetweb::TCivetweb() :
   THttpEngine("civetweb", "compact embedded http server"),
   fCtx(0),
   fCallbacks(0),
   fTopName(),
   fDebug(kFALSE)
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
Int_t TCivetweb::ProcessLog(const char* message)
{
   // process civetweb log message, can be used to detect critical errors

   if ((gDebug>0) || (strstr(message,"cannot bind to")!=0)) Error("Log", "%s", message);

   return 0;
}

//______________________________________________________________________________
Bool_t TCivetweb::Create(const char *args)
{
   // Creates embedded civetweb server
   // As main argument, http port should be specified like "8090".
   // Or one can provide combination of ipaddress and portnumber like 127.0.0.1:8090
   // Extra parameters like in URL string could be specified after '?' mark:
   //    thrds=N   - there N is number of threads used by the civetweb (default is 5)
   //    top=name  - configure top name, visible in the web browser
   //    auth_file=filename  - authentication file name, created with htdigets utility
   //    auth_domain=domain   - authentication domain
   //    loopback  - bind specified port to loopback 127.0.0.1 address
   //    debug  - enable debug mode, server always returns html page with request info

   fCallbacks = malloc(sizeof(struct mg_callbacks));
   memset(fCallbacks, 0, sizeof(struct mg_callbacks));
   //((struct mg_callbacks *) fCallbacks)->begin_request = begin_request_handler;
   ((struct mg_callbacks *) fCallbacks)->log_message = log_message_handler;
   TString sport = "8080";
   TString num_threads = "5";
   TString auth_file, auth_domain, log_file;

   // extract arguments
   if ((args != 0) && (strlen(args) > 0)) {

      // first extract port number
      sport = "";
      while ((*args != 0) && (*args != '?') && (*args != '/'))
         sport.Append(*args++);

      // than search for extra parameters
      while ((*args != 0) && (*args != '?')) args++;

      if (*args == '?') {
         TUrl url(TString::Format("http://localhost/folder%s", args));

         if (url.IsValid()) {
            url.ParseOptions();

            const char *top = url.GetValueFromOptions("top");
            if (top != 0) fTopName = top;

            const char *log = url.GetValueFromOptions("log");
            if (log != 0) log_file = log;

            Int_t thrds = url.GetIntValueFromOptions("thrds");
            if (thrds > 0) num_threads.Form("%d", thrds);

            const char *afile = url.GetValueFromOptions("auth_file");
            if (afile != 0) auth_file = afile;

            const char *adomain = url.GetValueFromOptions("auth_domain");
            if (adomain != 0) auth_domain = adomain;

            if (url.HasOption("debug")) fDebug = kTRUE;

            if (url.HasOption("loopback") && (sport.Index(":")==kNPOS))
               sport = TString("127.0.0.1:") + sport;
         }
      }
   }

   const char *options[20];
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

   if (log_file.Length() > 0) {
      options[op++] = "error_log_file";
      options[op++] = log_file.Data();
   }

   options[op++] = 0;

   // Start the web server.
   fCtx = mg_start((struct mg_callbacks *) fCallbacks, this, options);

   if (fCtx == 0) return kFALSE;

   mg_set_request_handler((struct mg_context *) fCtx, "/", begin_request_handler, 0);

   mg_set_websocket_handler((struct mg_context *) fCtx,
                            "**root.websocket$",
                             websocket_connect_handler,
                             websocket_ready_handler,
                             websocket_data_handler,
                             websocket_close_handler,
                             0);

   return kTRUE;
}

