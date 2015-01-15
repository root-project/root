// $Id$
// Author: Sergey Linev   21/12/2013

#include "THttpServer.h"

#include "TTimer.h"
#include "TSystem.h"
#include "TImage.h"
#include "TROOT.h"
#include "TClass.h"
#include "RVersion.h"
#include "RConfigure.h"

#include "THttpEngine.h"
#include "TRootSniffer.h"
#include "TRootSnifferStore.h"

#include <string>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#ifdef COMPILED_WITH_DABC
   extern "C" unsigned long crc32(unsigned long crc, const unsigned char* buf, unsigned int buflen);
   extern "C" unsigned long R__memcompress(char* tgt, unsigned long tgtsize, char* src, unsigned long srcsize);

   unsigned long R__crc32(unsigned long crc, const unsigned char* buf, unsigned int buflen)
   { return crc32(crc, buf, buflen); }
#else
   #include "RZip.h"
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpCallArg                                                         //
//                                                                      //
// Contains arguments for single HTTP call                              //
// Must be used in THttpEngine to process incoming http requests        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
THttpCallArg::THttpCallArg() :
   TObject(),
   fTopName(),
   fPathName(),
   fFileName(),
   fQuery(),
   fCond(),
   fContentType(),
   fHeader(),
   fContent(),
   fZipping(0),
   fBinData(0),
   fBinDataLength(0)
{
   // constructor
}

//______________________________________________________________________________
THttpCallArg::~THttpCallArg()
{
   // destructor

   if (fBinData) {
      free(fBinData);
      fBinData = 0;
   }
}

//______________________________________________________________________________
void THttpCallArg::SetBinData(void* data, Long_t length)
{
   // set binary data, which will be returned as reply body

   if (fBinData) free(fBinData);
   fBinData = data;
   fBinDataLength = length;

   // string content must be cleared in any case
   fContent.Clear();
}

//______________________________________________________________________________
void THttpCallArg::SetPathAndFileName(const char *fullpath)
{
   // set complete path of requested http element
   // For instance, it could be "/folder/subfolder/get.bin"
   // Here "/folder/subfolder/" is element path and "get.bin" requested file.
   // One could set path and file name separately

   fPathName.Clear();
   fFileName.Clear();

   if (fullpath == 0) return;

   const char *rslash = strrchr(fullpath, '/');
   if (rslash == 0) {
      fFileName = fullpath;
   } else {
      while ((fullpath != rslash) && (*fullpath == '/')) fullpath++;
      fPathName.Append(fullpath, rslash - fullpath);
      if (fPathName == "/") fPathName.Clear();
      fFileName = rslash + 1;
   }
}

//______________________________________________________________________________
void THttpCallArg::FillHttpHeader(TString &hdr, const char* kind)
{
   // fill HTTP header

   if (kind==0) kind = "HTTP/1.1";

   if ((fContentType.Length() == 0) || Is404()) {
      hdr.Form("%s 404 Not Found\r\n"
               "Content-Length: 0\r\n"
               "Connection: close\r\n\r\n", kind);
   } else {
      hdr.Form("%s 200 OK\r\n"
               "Content-Type: %s\r\n"
               "Connection: keep-alive\r\n"
               "Content-Length: %ld\r\n"
               "%s\r\n",
               kind,
               GetContentType(),
               GetContentLength(),
               fHeader.Data());
   }
}

//______________________________________________________________________________
Bool_t THttpCallArg::CompressWithGzip()
{
   // compress reply data with gzip compression

   char *objbuf = (char*) GetContent();
   Long_t objlen = GetContentLength();

   unsigned long objcrc = R__crc32(0, NULL, 0);
   objcrc = R__crc32(objcrc, (const unsigned char*) objbuf, objlen);

   // 10 bytes (ZIP header), compressed data, 8 bytes (CRC and original length)
   Int_t buflen = 10 + objlen + 8;
   if (buflen<512) buflen = 512;

   void* buffer = malloc(buflen);

   char *bufcur = (char*) buffer;

   *bufcur++ = 0x1f;  // first byte of ZIP identifier
   *bufcur++ = 0x8b;  // second byte of ZIP identifier
   *bufcur++ = 0x08;  // compression method
   *bufcur++ = 0x00;  // FLAG - empty, no any file names
   *bufcur++ = 0;    // empty timestamp
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    // XFL (eXtra FLags)
   *bufcur++ = 3;    // OS   3 means Unix
   //strcpy(bufcur, "get.json");
   //bufcur += strlen("get.json")+1;

   char dummy[8];
   memcpy(dummy, bufcur-6, 6);

   // R__memcompress fills first 6 bytes with own header, therefore just overwrite them
   unsigned long ziplen = R__memcompress(bufcur-6, objlen + 6, objbuf, objlen);

   memcpy(bufcur-6, dummy, 6);

   bufcur += (ziplen-6); // jump over compressed data (6 byte is extra ROOT header)

   *bufcur++ = objcrc & 0xff;    // CRC32
   *bufcur++ = (objcrc >> 8) & 0xff;
   *bufcur++ = (objcrc >> 16) & 0xff;
   *bufcur++ = (objcrc >> 24) & 0xff;

   *bufcur++ = objlen & 0xff;  // original data length
   *bufcur++ = (objlen >> 8) & 0xff;  // original data length
   *bufcur++ = (objlen >> 16) & 0xff;  // original data length
   *bufcur++ = (objlen >> 24) & 0xff;  // original data length

   SetBinData(buffer, bufcur - (char*) buffer);

   SetEncoding("gzip");

   return kTRUE;

}

// ====================================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpTimer                                                           //
//                                                                      //
// Specialized timer for THttpServer                                    //
// Provides regular call of THttpServer::ProcessRequests() method       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
class THttpTimer : public TTimer {
public:

   THttpServer *fServer;  //!

   THttpTimer(Long_t milliSec, Bool_t mode, THttpServer *serv) :
      TTimer(milliSec, mode), fServer(serv)
   {
      // construtor
   }
   virtual ~THttpTimer()
   {
      // destructor
   }
   virtual void Timeout()
   {
      // timeout handler
      // used to process http requests in main ROOT thread

      if (fServer) fServer->ProcessRequests();
   }
};

// =======================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpServer                                                          //
//                                                                      //
// Online http server for arbitrary ROOT application                    //
//                                                                      //
// Idea of THttpServer - provide remote http access to running          //
// ROOT application and enable HTML/JavaScript user interface.          //
// Any registered object can be requested and displayed in the browser. //
// There are many benefits of such approach:                            //
//     * standard http interface to ROOT application                    //
//     * no any temporary ROOT files when access data                   //
//     * user interface running in all browsers                         //
//                                                                      //
// Starting HTTP server                                                 //
//                                                                      //
// To start http server, at any time  create instance                   //
// of the THttpServer class like:                                       //
//    serv = new THttpServer("http:8080");                              //
//                                                                      //
// This will starts civetweb-based http server with http port 8080.     //
// Than one should be able to open address "http://localhost:8080"      //
// in any modern browser (IE, Firefox, Chrome) and browse objects,      //
// created in application. By default, server can access files,         //
// canvases and histograms via gROOT pointer. All such objects          //
// can be displayed with JSROOT graphics.                               //
//                                                                      //
// At any time one could register other objects with the command:       //
//                                                                      //
// TGraph* gr = new TGraph(10);                                         //
// gr->SetName("gr1");                                                  //
// serv->Register("graphs/subfolder", gr);                              //
//                                                                      //
// If objects content is changing in the application, one could         //
// enable monitoring flag in the browser - than objects view            //
// will be regularly updated.                                           //
//                                                                      //
// More information: http://root.cern.ch/drupal/content/users-guide     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
THttpServer::THttpServer(const char *engine) :
   TNamed("http", "ROOT http server"),
   fEngines(),
   fTimer(0),
   fSniffer(0),
   fMainThrdId(0),
   fJsRootSys(),
   fTopName("ROOT"),
   fDefaultPage(),
   fDefaultPageCont(),
   fDrawPage(),
   fDrawPageCont(),
   fMutex(),
   fCallArgs()
{
   // constructor

   // As argument, one specifies engine kind which should be
   // created like "http:8080". One could specify several engines
   // at once, separating them with ; like "http:8080;fastcgi:9000"
   // One also can configure readonly flag for sniffer like
   // "http:8080;readonly" or "http:8080;readwrite"
   //
   // Also searches for JavaScript ROOT sources, which are used in web clients
   // Typically JSROOT sources located in $ROOTSYS/etc/http directory,
   // but one could set JSROOTSYS variable to specify alternative location

   fMainThrdId = TThread::SelfId();

   // Info("THttpServer", "Create %p in thrd %ld", this, (long) fMainThrdId);

#ifdef COMPILED_WITH_DABC
   const char *dabcsys = gSystem->Getenv("DABCSYS");
   if (dabcsys != 0)
      fJsRootSys = TString::Format("%s/plugins/root/js", dabcsys);
#endif

   const char *jsrootsys = gSystem->Getenv("JSROOTSYS");
   if (jsrootsys != 0) fJsRootSys = jsrootsys;

   if (fJsRootSys.Length() == 0) {
#ifdef ROOTETCDIR
      TString jsdir = TString::Format("%s/http", ROOTETCDIR);
#else
      TString jsdir("$(ROOTSYS)/etc/http");
#endif
      if (gSystem->ExpandPathName(jsdir)) {
         Warning("THttpServer", "problems resolving '%s', use JSROOTSYS to specify $ROOTSYS/etc/http location", jsdir.Data());
         fJsRootSys = ".";
      } else {
         fJsRootSys = jsdir;
      }
   }

   fDefaultPage = fJsRootSys + "/files/online.htm";
   fDrawPage = fJsRootSys + "/files/draw.htm";

   SetSniffer(new TRootSniffer("sniff"));

   // start timer
   SetTimer(20, kTRUE);

   if (strchr(engine,';')==0) {
      CreateEngine(engine);
   } else {
      TObjArray* lst = TString(engine).Tokenize(";");

      for (Int_t n=0;n<=lst->GetLast();n++) {
         const char* opt = lst->At(n)->GetName();
         if ((strcmp(opt,"readonly")==0) || (strcmp(opt,"ro")==0)) {
            GetSniffer()->SetReadOnly(kTRUE);
         } else
         if ((strcmp(opt,"readwrite")==0) || (strcmp(opt,"rw")==0)) {
            GetSniffer()->SetReadOnly(kFALSE);
         } else
            CreateEngine(opt);
      }

      delete lst;
   }
}

//______________________________________________________________________________
THttpServer::~THttpServer()
{
   // destructor
   // delete all http engines and sniffer

   fEngines.Delete();

   SetSniffer(0);

   SetTimer(0);
}

//______________________________________________________________________________
void THttpServer::SetSniffer(TRootSniffer *sniff)
{
   // Set TRootSniffer to the server
   // Server takes ownership over sniffer

   if (fSniffer) delete fSniffer;
   fSniffer = sniff;
}

//______________________________________________________________________________
Bool_t THttpServer::IsReadOnly() const
{
   // returns read-only mode

   return fSniffer ? fSniffer->IsReadOnly() : kTRUE;
}

//______________________________________________________________________________
void THttpServer::SetReadOnly(Bool_t readonly)
{
   // Set read-only mode for the server (default on)
   // In read-only server is not allowed to change any ROOT object, registered to the server
   // Server also cannot execute objects method via exe.json request

   if (fSniffer) fSniffer->SetReadOnly(readonly);
}

//______________________________________________________________________________
Bool_t THttpServer::CreateEngine(const char *engine)
{
   // factory method to create different http engines
   // At the moment two engine kinds are supported:
   //  civetweb (default) and fastcgi
   // Examples:
   //   "civetweb:8080" or "http:8080" or ":8080" - creates civetweb web server with http port 8080
   //   "fastcgi:9000" - creates fastcgi server with port 9000
   //   "dabc:1237"    - create DABC server with port 1237 (only available with DABC installed)
   //   "dabc:master_host:port" - attach to DABC master, running on master_host:port (only available with DABC installed)

   if (engine == 0) return kFALSE;

   const char *arg = strchr(engine, ':');
   if (arg == 0) return kFALSE;

   TString clname;
   if (arg != engine) clname.Append(engine, arg - engine);

   if ((clname.Length() == 0) || (clname == "http") || (clname == "civetweb"))
      clname = "TCivetweb";
   else if (clname == "fastcgi")
      clname = "TFastCgi";
   else if (clname == "dabc")
      clname = "TDabcEngine";

   // ensure that required engine class exists before we try to create it
   TClass *engine_class = gROOT->LoadClass(clname.Data());
   if (engine_class == 0) return kFALSE;

   THttpEngine *eng = (THttpEngine *) engine_class->New();
   if (eng == 0) return kFALSE;

   eng->SetServer(this);

   if (!eng->Create(arg + 1)) {
      delete eng;
      return kFALSE;
   }

   fEngines.Add(eng);

   return kTRUE;
}

//______________________________________________________________________________
void THttpServer::SetTimer(Long_t milliSec, Bool_t mode)
{
   // create timer which will invoke ProcessRequests() function periodically
   // Timer is required to perform all actions in main ROOT thread
   // Method arguments are the same as for TTimer constructor
   // By default, sync timer with 100 ms period is created
   //
   // If milliSec == 0, no timer will be created.
   // In this case application should regularly call ProcessRequests() method.

   if (fTimer) {
      fTimer->Stop();
      delete fTimer;
      fTimer = 0;
   }
   if (milliSec > 0) {
      fTimer = new THttpTimer(milliSec, mode, this);
      fTimer->TurnOn();
   }
}

//______________________________________________________________________________
Bool_t THttpServer::VerifyFilePath(const char* fname)
{
   // Checked that filename does not contains relative path below current directory
   // Used to prevent access to files below current directory

   if ((fname==0) || (*fname==0)) return kFALSE;

   Int_t level = 0;

   while (*fname != 0) {

      // find next slash or backslash
      const char* next = strpbrk(fname, "/\\");
      if (next==0) return kTRUE;

      // most important - change to parent dir
      if ((next == fname + 2) && (*fname == '.') && (*(fname+1) == '.')) {
         fname += 3; level--;
         if (level<0) return kFALSE;
         continue;
      }

      // ignore current directory
      if ((next == fname + 1) && (*fname == '.'))  {
         fname += 2;
         continue;
      }

      // ignore slash at the front
      if (next==fname) {
         fname ++;
         continue;
      }

      fname = next+1;
      level++;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t THttpServer::IsFileRequested(const char *uri, TString &res) const
{
   // Verifies that request is just file name
   // File names typically contains prefix like "jsrootsys/"
   // If true, method returns real name of the file,
   // which should be delivered to the client
   // Method is thread safe and can be called from any thread

   if ((uri == 0) || (strlen(uri) == 0)) return kFALSE;

   TString fname = uri;

   Ssiz_t pos = fname.Index("jsrootsys/");
   if (pos != kNPOS) {
      fname.Remove(0, pos + 9);
      // check that directory below jsrootsys will not be accessed
      if (!VerifyFilePath(fname.Data())) {
         // Error("IsFileRequested","Prevent access to filepath %s", fname.Data());
         return kFALSE;
      }
      res = fJsRootSys + fname;
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t THttpServer::ExecuteHttp(THttpCallArg *arg)
{
   // Executes http request, specified in THttpCallArg structure
   // Method can be called from any thread
   // Actual execution will be done in main ROOT thread, where analysis code is running.

   if (fMainThrdId == TThread::SelfId()) {
      // should not happen, but one could process requests directly without any signaling

      ProcessRequest(arg);

      return kTRUE;
   }

   // add call arg to the list
   fMutex.Lock();
   fCallArgs.Add(arg);
   fMutex.UnLock();

   // and now wait until request is processed
   arg->fCond.Wait();

   return kTRUE;
}

//______________________________________________________________________________
void THttpServer::ProcessRequests()
{
   // Process requests, submitted for execution
   // Regularly invoked by THttpTimer, when somewhere in the code
   // gSystem->ProcessEvents() is called.
   // User can call serv->ProcessRequests() directly, but only from main analysis thread.

   if (fMainThrdId != TThread::SelfId()) {
      Error("ProcessRequests", "Should be called only from main ROOT thread");
      return;
   }

   while (true) {
      THttpCallArg *arg = 0;

      fMutex.Lock();
      if (fCallArgs.GetSize() > 0) {
         arg = (THttpCallArg *) fCallArgs.First();
         fCallArgs.RemoveFirst();
      }
      fMutex.UnLock();

      if (arg == 0) break;

      ProcessRequest(arg);

      arg->fCond.Signal();
   }

   // regularly call Process() method of engine to let perform actions in ROOT context
   TIter iter(&fEngines);
   THttpEngine *engine = 0;
   while ((engine = (THttpEngine *)iter()) != 0)
      engine->Process();
}

//______________________________________________________________________________
void THttpServer::ProcessRequest(THttpCallArg *arg)
{
   // Process single http request
   // Depending from requested path and filename different actions will be performed.
   // In most cases information is provided by TRootSniffer class

   if (arg->fFileName.IsNull() || (arg->fFileName == "index.htm")) {

      if (fDefaultPageCont.Length()==0) {
         Int_t len = 0;
         char* buf = ReadFileContent(fDefaultPage.Data(), len);
         if (len>0) fDefaultPageCont.Append(buf, len);
         delete buf;
      }

      if (fDefaultPageCont.Length()==0) {
         arg->Set404();
      } else {
         const char* hjsontag = "\"$$$h.json$$$\"";

         Ssiz_t pos = fDefaultPageCont.Index(hjsontag);
         if (pos==kNPOS) {
            arg->fContent = fDefaultPageCont;
         } else {
            TString h_json;
            TRootSnifferStoreJson store(h_json, kTRUE);
            const char *topname = fTopName.Data();
            if (arg->fTopName.Length() > 0) topname = arg->fTopName.Data();
            fSniffer->ScanHierarchy(topname, arg->fPathName.Data(), &store);

            arg->fContent.Clear();
            arg->fContent.Append(fDefaultPageCont, pos);
            arg->fContent.Append(h_json);
            arg->fContent.Append(fDefaultPageCont.Data() + pos + strlen(hjsontag));

            arg->AddHeader("Cache-Control", "private, no-cache, no-store, must-revalidate, max-age=0, proxy-revalidate, s-maxage=0");
            if (arg->fQuery.Index("nozip")==kNPOS) arg->SetZipping(2);
         }
         arg->SetContentType("text/html");
      }
      return;
   }

   if (arg->fFileName == "draw.htm") {
      if (fDrawPageCont.Length()==0) {
         Int_t len = 0;
         char* buf = ReadFileContent(fDrawPage.Data(), len);
         if (len>0) fDrawPageCont.Append(buf, len);
         delete buf;
      }

      if (fDrawPageCont.Length()==0) {
         arg->Set404();
      } else {
         const char* rootjsontag = "\"$$$root.json$$$\"";

         Ssiz_t pos = fDrawPageCont.Index(rootjsontag);
         if (pos==kNPOS) {
            arg->fContent = fDrawPageCont;
         } else {
            void* bindata(0);
            Long_t bindatalen(0);

            if (fSniffer->Produce(arg->fPathName.Data(), "root.json", "compact=3", bindata, bindatalen)) {
               arg->fContent.Clear();
               arg->fContent.Append(fDrawPageCont, pos);
               arg->fContent.Append((char*) bindata, bindatalen);
               arg->fContent.Append(fDrawPageCont.Data() + pos + strlen(rootjsontag));
               arg->AddHeader("Cache-Control", "private, no-cache, no-store, must-revalidate, max-age=0, proxy-revalidate, s-maxage=0");
               if (arg->fQuery.Index("nozip")==kNPOS) arg->SetZipping(2);
            } else {
               arg->fContent = fDrawPageCont;
            }
            free(bindata);
         }
         arg->SetContentType("text/html");
      }
      return;
   }

   TString filename;
   if (IsFileRequested(arg->fFileName.Data(), filename)) {
      arg->SetFile(filename);
      return;
   }

   filename = arg->fFileName;
   Bool_t iszip = kFALSE;
   if (filename.EndsWith(".gz")) {
      filename.Resize(filename.Length()-3);
      iszip = kTRUE;
   }

   if (filename == "h.xml")  {

      arg->fContent.Form(
         "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         "<root>\n");

      {
         TRootSnifferStoreXml store(arg->fContent, arg->fQuery.Index("compact")!=kNPOS);

         const char *topname = fTopName.Data();
         if (arg->fTopName.Length() > 0) topname = arg->fTopName.Data();

         fSniffer->ScanHierarchy(topname, arg->fPathName.Data(), &store);
      }

      arg->fContent.Append("</root>\n");
      arg->SetXml();
   } else

   if (filename == "h.json")  {

      TRootSnifferStoreJson store(arg->fContent, arg->fQuery.Index("compact")!=kNPOS);

      const char *topname = fTopName.Data();
      if (arg->fTopName.Length() > 0) topname = arg->fTopName.Data();

      fSniffer->ScanHierarchy(topname, arg->fPathName.Data(), &store);

      arg->SetJson();
   } else

   if (fSniffer->Produce(arg->fPathName.Data(), filename.Data(), arg->fQuery.Data(), arg->fBinData, arg->fBinDataLength)) {
      // define content type base on extension
      arg->SetContentType(GetMimeType(filename.Data()));
   } else {
      // request is not processed
      arg->Set404();
   }

   if (arg->Is404()) return;

   if (iszip) arg->SetZipping(3);

   if (filename == "root.bin") {
      // only for binary data master version is important
      // it allows to detect if streamer info was modified
      const char* parname = fSniffer->IsStreamerInfoItem(arg->fPathName.Data()) ? "BVersion" : "MVersion";
      arg->AddHeader(parname, Form("%u", (unsigned) fSniffer->GetStreamerInfoHash()));
   }

   // try to avoid caching on the browser
   arg->AddHeader("Cache-Control", "private, no-cache, no-store, must-revalidate, max-age=0, proxy-revalidate, s-maxage=0");
}

//______________________________________________________________________________
Bool_t THttpServer::Register(const char *subfolder, TObject *obj)
{
   // Register object in folders hierarchy
   //
   // See TRootSniffer::RegisterObject() for more details

   return fSniffer->RegisterObject(subfolder, obj);
}

//______________________________________________________________________________
Bool_t THttpServer::Unregister(TObject *obj)
{
   // Unregister object in folders hierarchy
   //
   // See TRootSniffer::UnregisterObject() for more details

   return fSniffer->UnregisterObject(obj);
}


//______________________________________________________________________________
const char *THttpServer::GetMimeType(const char *path)
{
   // Returns MIME type base on file extension

   static const struct {
      const char *extension;
      int ext_len;
      const char *mime_type;
   } builtin_mime_types[] = {
      {".xml", 4, "text/xml"},
      {".json", 5, "application/json"},
      {".bin", 4, "application/x-binary"},
      {".gif", 4, "image/gif"},
      {".jpg", 4, "image/jpeg"},
      {".png", 4, "image/png"},
      {".html", 5, "text/html"},
      {".htm", 4, "text/html"},
      {".shtm", 5, "text/html"},
      {".shtml", 6, "text/html"},
      {".css", 4, "text/css"},
      {".js",  3, "application/x-javascript"},
      {".ico", 4, "image/x-icon"},
      {".jpeg", 5, "image/jpeg"},
      {".svg", 4, "image/svg+xml"},
      {".txt", 4, "text/plain"},
      {".torrent", 8, "application/x-bittorrent"},
      {".wav", 4, "audio/x-wav"},
      {".mp3", 4, "audio/x-mp3"},
      {".mid", 4, "audio/mid"},
      {".m3u", 4, "audio/x-mpegurl"},
      {".ogg", 4, "application/ogg"},
      {".ram", 4, "audio/x-pn-realaudio"},
      {".xslt", 5, "application/xml"},
      {".xsl", 4, "application/xml"},
      {".ra",  3, "audio/x-pn-realaudio"},
      {".doc", 4, "application/msword"},
      {".exe", 4, "application/octet-stream"},
      {".zip", 4, "application/x-zip-compressed"},
      {".xls", 4, "application/excel"},
      {".tgz", 4, "application/x-tar-gz"},
      {".tar", 4, "application/x-tar"},
      {".gz",  3, "application/x-gunzip"},
      {".arj", 4, "application/x-arj-compressed"},
      {".rar", 4, "application/x-arj-compressed"},
      {".rtf", 4, "application/rtf"},
      {".pdf", 4, "application/pdf"},
      {".swf", 4, "application/x-shockwave-flash"},
      {".mpg", 4, "video/mpeg"},
      {".webm", 5, "video/webm"},
      {".mpeg", 5, "video/mpeg"},
      {".mov", 4, "video/quicktime"},
      {".mp4", 4, "video/mp4"},
      {".m4v", 4, "video/x-m4v"},
      {".asf", 4, "video/x-ms-asf"},
      {".avi", 4, "video/x-msvideo"},
      {".bmp", 4, "image/bmp"},
      {".ttf", 4, "application/x-font-ttf"},
      {NULL,  0, NULL}
   };

   int path_len = strlen(path);

   for (int i = 0; builtin_mime_types[i].extension != NULL; i++) {
      if (path_len <= builtin_mime_types[i].ext_len) continue;
      const char *ext = path + (path_len - builtin_mime_types[i].ext_len);
      if (strcmp(ext, builtin_mime_types[i].extension) == 0) {
         return builtin_mime_types[i].mime_type;
      }
   }

   return "text/plain";
}

//______________________________________________________________________________
char* THttpServer::ReadFileContent(const char* filename, Int_t& len)
{
   // reads file content

   len = 0;

   std::ifstream is(filename);
   if (!is) return 0;

   is.seekg(0, is.end);
   len = is.tellg();
   is.seekg(0, is.beg);

   char *buf = (char *) malloc(len);
   is.read(buf, len);
   if (!is) {
      free(buf);
      len = 0;
      return 0;
   }

   return buf;
}
