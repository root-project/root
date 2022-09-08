// @(#)root/net:$Id$
// Author: Fons Rademakers   17/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebFile                                                             //
//                                                                      //
// A TWebFile is like a normal TFile except that it reads its data      //
// via a standard apache web server. A TWebFile is a read-only file.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWebFile.h"
#include "TROOT.h"
#include "TSocket.h"
#include "Bytes.h"
#include "TError.h"
#include "TSystem.h"
#include "TBase64.h"
#include "TVirtualPerfStats.h"
#ifdef R__SSL
#include "TSSLSocket.h"
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
# ifndef EADDRINUSE
#  define EADDRINUSE  10048
# endif
# ifndef EISCONN
#  define EISCONN     10056
# endif
#endif

static const char *gUserAgent = "User-Agent: ROOT-TWebFile/1.1";

TUrl TWebFile::fgProxy;

Long64_t TWebFile::fgMaxFullCacheSize = 500000000;


// Internal class used to manage the socket that may stay open between
// calls when HTTP/1.1 protocol is used
class TWebSocket {
private:
   TWebFile *fWebFile;           // associated web file
public:
   TWebSocket(TWebFile *f);
   ~TWebSocket();
   void ReOpen();
};

////////////////////////////////////////////////////////////////////////////////
/// Open web file socket.

TWebSocket::TWebSocket(TWebFile *f)
{
   fWebFile = f;
   if (!f->fSocket)
      ReOpen();
}

////////////////////////////////////////////////////////////////////////////////
/// Close socket in case not HTTP/1.1 protocol or when explicitly requested.

TWebSocket::~TWebSocket()
{
   if (!fWebFile->fHTTP11) {
      delete fWebFile->fSocket;
      fWebFile->fSocket = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Re-open web file socket.

void TWebSocket::ReOpen()
{
   if (fWebFile->fSocket) {
      delete fWebFile->fSocket;
      fWebFile->fSocket = 0;
   }

   TUrl connurl;
   if (fWebFile->fProxy.IsValid())
      connurl = fWebFile->fProxy;
   else
      connurl = fWebFile->fUrl;

   for (Int_t i = 0; i < 5; i++) {
      if (strcmp(connurl.GetProtocol(), "https") == 0) {
#ifdef R__SSL
         fWebFile->fSocket = new TSSLSocket(connurl.GetHost(), connurl.GetPort());
#else
         ::Error("TWebSocket::ReOpen", "library compiled without SSL, https not supported");
         return;
#endif
      } else
         fWebFile->fSocket = new TSocket(connurl.GetHost(), connurl.GetPort());

      if (!fWebFile->fSocket || !fWebFile->fSocket->IsValid()) {
         delete fWebFile->fSocket;
         fWebFile->fSocket = 0;
         if (gSystem->GetErrno() == EADDRINUSE || gSystem->GetErrno() == EISCONN) {
            gSystem->Sleep(i*10);
         } else {
            ::Error("TWebSocket::ReOpen", "cannot connect to host %s (errno=%d)",
                    fWebFile->fUrl.GetHost(), gSystem->GetErrno());
            return;
         }
      } else
         return;
   }
}


ClassImp(TWebFile);

////////////////////////////////////////////////////////////////////////////////
/// Create a Web file object. A web file is the same as a read-only
/// TFile except that it is being read via a HTTP server. The url
/// argument must be of the form: http://host.dom.ain/file.root.
/// The opt can be "NOPROXY", to bypass any set "http_proxy" shell
/// variable. The proxy can be specified as (in sh, or equivalent csh):
///   export http_proxy=http://pcsalo.cern.ch:3128
/// The proxy can also be specified via the static method TWebFile::SetProxy().
/// Basic authentication (AuthType Basic) is supported. The user name and
/// passwd can be specified in the url like this:
///   http://username:mypasswd@pcsalo.cern.ch/files/aap.root
/// If the file specified in the URL does not exist or is not accessible
/// the kZombie bit will be set in the TWebFile object. Use IsZombie()
/// to see if the file is accessible. The preferred interface to this
/// constructor is via TFile::Open().

TWebFile::TWebFile(const char *url, Option_t *opt)
   : TFile(url, strstr(opt, "_WITHOUT_GLOBALREGISTRATION") != nullptr ? "WEB_WITHOUT_GLOBALREGISTRATION" : "WEB"),
     fSocket(0)
{
   TString option = opt;
   fNoProxy = kFALSE;
   if (option.Contains("NOPROXY", TString::kIgnoreCase))
      fNoProxy = kTRUE;
   CheckProxy();

   Bool_t headOnly = kFALSE;
   if (option.Contains("HEADONLY", TString::kIgnoreCase))
      headOnly = kTRUE;

   if (option == "IO")
      return;

   Init(headOnly);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a Web file object. A web file is the same as a read-only
/// TFile except that it is being read via a HTTP server. Make sure url
/// is a valid TUrl object.
/// The opt can be "NOPROXY", to bypass any set "http_proxy" shell
/// variable. The proxy can be specified as (in sh, or equivalent csh):
///   export http_proxy=http://pcsalo.cern.ch:3128
/// The proxy can also be specified via the static method TWebFile::SetProxy().
/// Basic authentication (AuthType Basic) is supported. The user name and
/// passwd can be specified in the url like this:
///   http://username:mypasswd@pcsalo.cern.ch/files/aap.root
/// If the file specified in the URL does not exist or is not accessible
/// the kZombie bit will be set in the TWebFile object. Use IsZombie()
/// to see if the file is accessible.

TWebFile::TWebFile(TUrl url, Option_t *opt) : TFile(url.GetUrl(), "WEB"), fSocket(0)
{
   TString option = opt;
   fNoProxy = kFALSE;
   if (option.Contains("NOPROXY", TString::kIgnoreCase))
      fNoProxy = kTRUE;
   CheckProxy();

   Bool_t headOnly = kFALSE;
   if (option.Contains("HEADONLY", TString::kIgnoreCase))
      headOnly = kTRUE;

   Init(headOnly);
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup.

TWebFile::~TWebFile()
{
   delete fSocket;
   if (fFullCache) {
      free(fFullCache);
      fFullCache = 0;
      fFullCacheSize = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize a TWebFile object.

void TWebFile::Init(Bool_t readHeadOnly)
{
   char buf[4];
   int  err;

   fSocket     = 0;
   fSize       = -1;
   fHasModRoot = kFALSE;
   fHTTP11     = kFALSE;
   fFullCache  = 0;
   fFullCacheSize = 0;
   SetMsgReadBuffer10();

   if ((err = GetHead()) < 0) {
      if (readHeadOnly) {
         fD = -1;
         fWritten = err;
         return;
      }
      if (err == -2) {
         Error("TWebFile", "%s does not exist", fBasicUrl.Data());
         MakeZombie();
         gDirectory = gROOT;
         return;
      }
      // err == -3 HEAD not supported, fall through and try ReadBuffer()
   }
   if (readHeadOnly) {
      fD = -1;
      return;
   }

   if (fIsRootFile) {
      Seek(0);
      if (ReadBuffer(buf, 4)) {
         MakeZombie();
         gDirectory = gROOT;
         return;
      }

      if (strncmp(buf, "root", 4) && strncmp(buf, "PK", 2)) {  // PK is zip file
         Error("TWebFile", "%s is not a ROOT file", fBasicUrl.Data());
         MakeZombie();
         gDirectory = gROOT;
         return;
      }
   }

   TFile::Init(kFALSE);
   fD = -2;   // so TFile::IsOpen() will return true when in TFile::~TFile
}

////////////////////////////////////////////////////////////////////////////////
/// Set GET command for use by ReadBuffer(s)10(), handle redirection if
/// needed. Give full URL so Apache's virtual hosts solution works.

void TWebFile::SetMsgReadBuffer10(const char *redirectLocation, Bool_t tempRedirect)
{
   TUrl oldUrl;
   TString oldBasicUrl;

   if (redirectLocation) {
      if (tempRedirect) { // temp redirect
         fUrlOrg      = fUrl;
         fBasicUrlOrg = fBasicUrl;
      } else {             // permanent redirect
         fUrlOrg      = "";
         fBasicUrlOrg = "";
      }

      oldUrl = fUrl;
      oldBasicUrl = fBasicUrl;

      fUrl.SetUrl(redirectLocation);
      fBasicUrl = fUrl.GetProtocol();
      fBasicUrl += "://";
      fBasicUrl += fUrl.GetHost();
      fBasicUrl += ":";
      fBasicUrl += fUrl.GetPort();
      fBasicUrl += "/";
      fBasicUrl += fUrl.GetFile();
      // add query string again
      TString rdl(redirectLocation);
      if (rdl.Index("?") >= 0) {
         rdl = rdl(rdl.Index("?"), rdl.Length());
         fBasicUrl += rdl;
      }
   }

   if (fMsgReadBuffer10 != "") {
      // patch up existing command
      if (oldBasicUrl != "") {
         // change to redirection location
         fMsgReadBuffer10.ReplaceAll(oldBasicUrl, fBasicUrl);
         fMsgReadBuffer10.ReplaceAll(TString("Host: ")+oldUrl.GetHost(), TString("Host: ")+fUrl.GetHost());
      } else if (fBasicUrlOrg != "") {
         // change back from temp redirection location
         fMsgReadBuffer10.ReplaceAll(fBasicUrl, fBasicUrlOrg);
         fMsgReadBuffer10.ReplaceAll(TString("Host: ")+fUrl.GetHost(), TString("Host: ")+fUrlOrg.GetHost());
         fUrl         = fUrlOrg;
         fBasicUrl    = fBasicUrlOrg;
         fUrlOrg      = "";
         fBasicUrlOrg = "";
      }
   }

   if (fBasicUrl == "") {
      fBasicUrl += fUrl.GetProtocol();
      fBasicUrl += "://";
      fBasicUrl += fUrl.GetHost();
      fBasicUrl += ":";
      fBasicUrl += fUrl.GetPort();
      fBasicUrl += "/";
      fBasicUrl += fUrl.GetFile();
      fBasicUrl += "?";
      fBasicUrl += fUrl.GetOptions();
   }

   if (fMsgReadBuffer10 == "") {
      fMsgReadBuffer10 = "GET ";
      fMsgReadBuffer10 += fBasicUrl;
      if (fHTTP11)
         fMsgReadBuffer10 += " HTTP/1.1";
      else
         fMsgReadBuffer10 += " HTTP/1.0";
      fMsgReadBuffer10 += "\r\n";
      if (fHTTP11) {
         fMsgReadBuffer10 += "Host: ";
         fMsgReadBuffer10 += fUrl.GetHost();
         fMsgReadBuffer10 += "\r\n";
      }
      fMsgReadBuffer10 += BasicAuthentication();
      fMsgReadBuffer10 += gUserAgent;
      fMsgReadBuffer10 += "\r\n";
      fMsgReadBuffer10 += "Range: bytes=";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if shell var "http_proxy" has been set and should be used.

void TWebFile::CheckProxy()
{
   if (fNoProxy)
      return;

   if (fgProxy.IsValid()) {
      fProxy = fgProxy;
      return;
   }

   TString proxy = gSystem->Getenv("http_proxy");
   if (proxy != "") {
      TUrl p(proxy);
      if (strcmp(p.GetProtocol(), "http")) {
         Error("CheckProxy", "protocol must be HTTP in proxy URL %s",
               proxy.Data());
         return;
      }
      fProxy = p;
      if (gDebug > 0)
         Info("CheckProxy", "using HTTP proxy %s", fProxy.GetUrl());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// A TWebFile that has been correctly constructed is always considered open.

Bool_t TWebFile::IsOpen() const
{
   return IsZombie() ? kFALSE : kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Reopen a file with a different access mode, like from READ to
/// UPDATE or from NEW, CREATE, RECREATE, UPDATE to READ. Thus the
/// mode argument can be either "READ" or "UPDATE". The method returns
/// 0 in case the mode was successfully modified, 1 in case the mode
/// did not change (was already as requested or wrong input arguments)
/// and -1 in case of failure, in which case the file cannot be used
/// anymore. A TWebFile cannot be reopened in update mode.

Int_t TWebFile::ReOpen(Option_t *mode)
{
   TString opt = mode;
   opt.ToUpper();

   if (opt != "READ" && opt != "UPDATE")
      Error("ReOpen", "mode must be either READ or UPDATE, not %s", opt.Data());

   if (opt == "UPDATE")
      Error("ReOpen", "update mode not allowed for a TWebFile");

   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via HTTP daemon. This
/// routine connects to the remote host, sends the request and returns
/// the buffer. Returns kTRUE in case of error.

Bool_t TWebFile::ReadBuffer(char *buf, Int_t len)
{
   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   if (!fHasModRoot)
      return ReadBuffer10(buf, len);

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
   if (fMsgReadBuffer == "") {
      fMsgReadBuffer = "GET ";
      fMsgReadBuffer += fBasicUrl;
      fMsgReadBuffer += "?";
   }
   TString msg = fMsgReadBuffer;
   msg += fOffset;
   msg += ":";
   msg += len;
   msg += "\r\n";

   if (GetFromWeb(buf, len, msg) == -1)
      return kTRUE;

   fOffset += len;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via HTTP daemon. This
/// routine connects to the remote host, sends the request and returns
/// the buffer. Returns kTRUE in case of error.

Bool_t TWebFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   SetOffset(pos);
   return ReadBuffer(buf, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte range from remote file via HTTP 1.0 daemon (without
/// mod-root installed). This routine connects to the remote host, sends the
/// request and returns the buffer. Returns kTRUE in case of error.

Bool_t TWebFile::ReadBuffer10(char *buf, Int_t len)
{
   SetMsgReadBuffer10();

   TString msg = fMsgReadBuffer10;
   msg += fOffset;
   msg += "-";
   msg += fOffset+len-1;
   msg += "\r\n\r\n";

   Long64_t apos = fOffset - fArchiveOffset;

   // in case when server does not support segments, let chance to recover
   Int_t n = GetFromWeb10(buf, len, msg, 1, &apos, &len);
   if (n == -1)
      return kTRUE;
   // The -2 error condition typically only happens when
   // GetHead() failed because not implemented, in the first call to
   // ReadBuffer() in Init(), it is not checked in ReadBuffers10().
   if (n == -2) {
      Error("ReadBuffer10", "%s does not exist", fBasicUrl.Data());
      MakeZombie();
      gDirectory = gROOT;
      return kTRUE;
   }

   fOffset += len;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte ranges from remote file via HTTP daemon.
/// Reads the nbuf blocks described in arrays pos and len,
/// where pos[i] is the seek position of block i of length len[i].
/// Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer
/// This function is overloaded by TNetFile, TWebFile, etc.
/// Returns kTRUE in case of failure.

Bool_t TWebFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
   if (!fHasModRoot)
      return ReadBuffers10(buf, pos, len, nbuf);

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
   if (fMsgReadBuffer == "") {
      fMsgReadBuffer = "GET ";
      fMsgReadBuffer += fBasicUrl;
      fMsgReadBuffer += "?";
   }
   TString msg = fMsgReadBuffer;

   Int_t k = 0, n = 0, cnt = 0;
   for (Int_t i = 0; i < nbuf; i++) {
      if (n) msg += ",";
      msg += pos[i] + fArchiveOffset;
      msg += ":";
      msg += len[i];
      n   += len[i];
      cnt++;
      if ((msg.Length() > 8000) || (cnt >= 200)) {
         msg += "\r\n";
         if (GetFromWeb(&buf[k], n, msg) == -1)
            return kTRUE;
         msg = fMsgReadBuffer;
         k += n;
         n = 0;
         cnt = 0;
      }
   }

   msg += "\r\n";

   if (GetFromWeb(&buf[k], n, msg) == -1)
      return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read specified byte ranges from remote file via HTTP 1.0 daemon (without
/// mod-root installed). Read the nbuf blocks described in arrays pos and len,
/// where pos[i] is the seek position of block i of length len[i].
/// Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer
/// This function is overloaded by TNetFile, TWebFile, etc.
/// Returns kTRUE in case of failure.

Bool_t TWebFile::ReadBuffers10(char *buf,  Long64_t *pos, Int_t *len, Int_t nbuf)
{
   SetMsgReadBuffer10();

   TString msg = fMsgReadBuffer10;

   Int_t k = 0, n = 0, r, cnt = 0;
   for (Int_t i = 0; i < nbuf; i++) {
      if (n) msg += ",";
      msg += pos[i] + fArchiveOffset;
      msg += "-";
      msg += pos[i] + fArchiveOffset + len[i] - 1;
      n   += len[i];
      cnt++;
      if ((msg.Length() > 8000) || (cnt >= 200) || (i+1 == nbuf)) {
         msg += "\r\n\r\n";
         r = GetFromWeb10(&buf[k], n, msg, cnt, pos + (i+1-cnt), len + (i+1-cnt));
         if (r == -1)
            return kTRUE;
         msg = fMsgReadBuffer10;
         k += n;
         n = 0;
         cnt = 0;
      }
   }

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Extract requested segments from the cached content.
/// Such cache can be produced when server suddenly returns full data instead of segments
/// Returns -1 in case of error, 0 in case of success

Int_t TWebFile::GetFromCache(char *buf, Int_t len, Int_t nseg, Long64_t *seg_pos, Int_t *seg_len)
{
   if (!fFullCache) return -1;

   if (gDebug > 0)
      Info("GetFromCache", "Extract %d segments total len %d from cached data", nseg, len);

   Int_t curr = 0;
   for (Int_t cnt=0;cnt<nseg;cnt++) {
      // check that target buffer has enough space
      if (curr + seg_len[cnt] > len) return -1;
      // check that segment is inside cached area
      if (fArchiveOffset + seg_pos[cnt] + seg_len[cnt] > fFullCacheSize) return -1;
      char* src = (char*) fFullCache + fArchiveOffset + seg_pos[cnt];
      memcpy(buf + curr, src, seg_len[cnt]);
      curr += seg_len[cnt];
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read request from web server. Returns -1 in case of error,
/// 0 in case of success.

Int_t TWebFile::GetFromWeb(char *buf, Int_t len, const TString &msg)
{
   TSocket *s;

   if (!len) return 0;

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   TUrl connurl;
   if (fProxy.IsValid())
      connurl = fProxy;
   else
      connurl = fUrl;

   if (strcmp(connurl.GetProtocol(), "https") == 0) {
#ifdef R__SSL
      s = new TSSLSocket(connurl.GetHost(), connurl.GetPort());
#else
      Error("GetFromWeb", "library compiled without SSL, https not supported");
      return -1;
#endif
   } else
      s = new TSocket(connurl.GetHost(), connurl.GetPort());

   if (!s->IsValid()) {
      Error("GetFromWeb", "cannot connect to host %s", fUrl.GetHost());
      delete s;
      return -1;
   }

   if (s->SendRaw(msg.Data(), msg.Length()) == -1) {
      Error("GetFromWeb", "error sending command to host %s", fUrl.GetHost());
      delete s;
      return -1;
   }

   if (s->RecvRaw(buf, len) == -1) {
      Error("GetFromWeb", "error receiving data from host %s", fUrl.GetHost());
      delete s;
      return -1;
   }

   // collect statistics
   fBytesRead += len;
   fReadCalls++;
#ifdef R__WIN32
   SetFileBytesRead(GetFileBytesRead() + len);
   SetFileReadCalls(GetFileReadCalls() + 1);
#else
   fgBytesRead += len;
   fgReadCalls++;
#endif

   if (gPerfStats)
      gPerfStats->FileReadEvent(this, len, start);

   delete s;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read multiple byte range request from web server.
/// Uses HTTP 1.0 daemon wihtout mod-root.
/// Returns -2 in case file does not exist, -1 in case
/// of error and 0 in case of success.

Int_t TWebFile::GetFromWeb10(char *buf, Int_t len, const TString &msg, Int_t nseg, Long64_t *seg_pos, Int_t *seg_len)
{
   if (!len) return 0;

   // if file content was cached, reuse it
   if (fFullCache && (nseg>0))
       return GetFromCache(buf, len, nseg, seg_pos, seg_len);

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   // open fSocket and close it when going out of scope
   TWebSocket ws(this);

   if (!fSocket || !fSocket->IsValid()) {
      Error("GetFromWeb10", "cannot connect to host %s", fUrl.GetHost());
      return -1;
   }

   if (gDebug > 0)
      Info("GetFromWeb10", "sending HTTP request:\n%s", msg.Data());

   if (fSocket->SendRaw(msg.Data(), msg.Length()) == -1) {
      Error("GetFromWeb10", "error sending command to host %s", fUrl.GetHost());
      return -1;
   }

   char line[8192];
   Int_t n, ret = 0, nranges = 0, ltot = 0, redirect = 0;
   TString boundary, boundaryEnd;
   Long64_t first = -1, last = -1, tot, fullsize = 0;
   TString redir;

   while ((n = GetLine(fSocket, line, sizeof(line))) >= 0) {
      if (n == 0) {
         if (ret < 0)
            return ret;
         if (redirect) {
            if (redir.IsNull()) {
              // Some sites (s3.amazonaws.com) do not return a Location field on 301
              Error("GetFromWeb10", "error - redirect without location from host %s", fUrl.GetHost());
              return -1;
            }

            ws.ReOpen();
            // set message to reflect the redirectLocation and add bytes field
            TString msg_1 = fMsgReadBuffer10;
            msg_1 += fOffset;
            msg_1 += "-";
            msg_1 += fOffset+len-1;
            msg_1 += "\r\n\r\n";
            return GetFromWeb10(buf, len, msg_1);
         }

         if (first >= 0) {
            Int_t ll = Int_t(last - first) + 1;
            Int_t rsize;
            if ((rsize = fSocket->RecvRaw(&buf[ltot], ll)) == -1) {
               Error("GetFromWeb10", "error receiving data from host %s", fUrl.GetHost());
               return -1;
            }
            else if (ll != rsize) {
               Error("GetFromWeb10", "expected %d bytes, got %d", ll, rsize);
               return -1;
            }
            ltot += ll;

            first = -1;

            if (boundary == "")
               break;  // not a multipart response
         }

         if (fullsize > 0) {

            if (nseg <= 0) {
               Error("GetFromWeb10","Need segments data to extract parts from full size %lld", fullsize);
               return -1;
            }

            if (len > fullsize) {
               Error("GetFromWeb10","Requested part %d longer than full size %lld", len, fullsize);
               return -1;
            }

            if ((fFullCache == 0) && (fullsize <= GetMaxFullCacheSize())) {
              // try to read file content into cache and than reuse it, limit cache by 2 GB
               fFullCache = malloc(fullsize);
               if (fFullCache != 0) {
                  if (fSocket->RecvRaw(fFullCache, fullsize) != fullsize) {
                     Error("GetFromWeb10", "error receiving data from host %s", fUrl.GetHost());
                     free(fFullCache); fFullCache = 0;
                     return -1;
                  }
                  fFullCacheSize = fullsize;
                  return GetFromCache(buf, len, nseg, seg_pos, seg_len);
               }
               // when cache allocation failed, try without cache
            }

            // check all segemnts are inside range and in sorted order
            for (Int_t cnt=0;cnt<nseg;cnt++) {
               if (fArchiveOffset + seg_pos[cnt] + seg_len[cnt] > fullsize) {
                  Error("GetFromWeb10","Requested segment %lld len %d is outside of full range %lld",  seg_pos[cnt], seg_len[cnt], fullsize);
                  return -1;
               }
               if ((cnt>0) &&  (seg_pos[cnt-1] + seg_len[cnt-1] > seg_pos[cnt])) {
                  Error("GetFromWeb10","Requested segments are not in sorted order");
                  return -1;
               }
            }

            Long64_t pos = 0;
            char* curr = buf;
            char dbuf[2048]; // dummy buffer for skip data

            // now read complete file and take only requested segments into the buffer
            for (Int_t cnt=0; cnt<nseg; cnt++) {
               // first skip data before segment
               while (pos < fArchiveOffset + seg_pos[cnt]) {
                  Long64_t ll = fArchiveOffset + seg_pos[cnt] - pos;
                  if (ll > Int_t(sizeof(dbuf))) ll = sizeof(dbuf);
                  if (fSocket->RecvRaw(dbuf, ll) != ll) {
                     Error("GetFromWeb10", "error receiving data from host %s", fUrl.GetHost());
                     return -1;
                  }
                  pos += ll;
               }

               // reading segment itself
               if (fSocket->RecvRaw(curr, seg_len[cnt]) != seg_len[cnt]) {
                  Error("GetFromWeb10", "error receiving data from host %s", fUrl.GetHost());
                  return -1;
               }
               curr += seg_len[cnt];
               pos += seg_len[cnt];
               ltot += seg_len[cnt];
            }

            // now read file to the end
            while (pos < fullsize) {
               Long64_t ll = fullsize - pos;
               if (ll > Int_t(sizeof(dbuf))) ll = sizeof(dbuf);
               if (fSocket->RecvRaw(dbuf, ll) != ll) {
                  Error("GetFromWeb10", "error receiving data from host %s", fUrl.GetHost());
                  return -1;
               }
               pos += ll;
            }

            if (gDebug>0) Info("GetFromWeb10","Complete reading %d bytes in %d segments out of full size %lld", len, nseg, fullsize);

            break;
         }

         continue;
      }

      if (gDebug > 0)
         Info("GetFromWeb10", "header: %s", line);

      if (boundaryEnd == line) {
         if (gDebug > 0)
            Info("GetFromWeb10", "got all headers");
         break;
      }
      if (boundary == line) {
         nranges++;
         if (gDebug > 0)
            Info("GetFromWeb10", "get new multipart byte range (%d)", nranges);
      }

      TString res = line;

      if (res.BeginsWith("HTTP/1.")) {
         if (res.BeginsWith("HTTP/1.1")) {
            if (!fHTTP11)
               fMsgReadBuffer10  = "";
            fHTTP11 = kTRUE;
         }
         TString scode = res(9, 3);
         Int_t code = scode.Atoi();
         if (code >= 500) {
            ret = -1;
            TString mess = res(13, 1000);
            Error("GetFromWeb10", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
         } else if (code >= 400) {
            if (code == 404)
               ret = -2;   // file does not exist
            else {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetFromWeb10", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code >= 300) {
            if (code == 301 || code == 303) {
               redirect = 1;   // permanent redirect
            } else if (code == 302 || code == 307) {
               // treat 302 as 303: permanent redirect
               redirect = 1;
               //redirect = 2; // temp redirect
            } else {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetFromWeb10", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code > 200) {
            if (code != 206) {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetFromWeb10", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code == 200) {
            fullsize = -200; // make indication of code 200
            Warning("GetFromWeb10",
                    "Server %s response with complete file, but only part of it was requested.\n"
                    "Check MaxRanges configuration parameter (if Apache is used)",
                    fUrl.GetHost());

         }
      } else if (res.BeginsWith("Content-Type: multipart")) {
         boundary = res(res.Index("boundary=")+9, 1000);
         if (boundary[0]=='"' && boundary[boundary.Length()-1]=='"') {
            boundary = boundary(1,boundary.Length()-2);
         }
         boundary = "--" + boundary;
         boundaryEnd = boundary + "--";
      } else if (res.BeginsWith("Content-range:")) {
#ifdef R__WIN32
         sscanf(res.Data(), "Content-range: bytes %I64d-%I64d/%I64d", &first, &last, &tot);
#else
         sscanf(res.Data(), "Content-range: bytes %lld-%lld/%lld", &first, &last, &tot);
#endif
         if (fSize == -1) fSize = tot;
      } else if (res.BeginsWith("Content-Range:")) {
#ifdef R__WIN32
         sscanf(res.Data(), "Content-Range: bytes %I64d-%I64d/%I64d", &first, &last, &tot);
#else
         sscanf(res.Data(), "Content-Range: bytes %lld-%lld/%lld", &first, &last, &tot);
#endif
         if (fSize == -1) fSize = tot;
      } else if (res.BeginsWith("Content-Length:") && (fullsize == -200)) {
#ifdef R__WIN32
         sscanf(res.Data(), "Content-Length: %I64d", &fullsize);
#else
         sscanf(res.Data(), "Content-Length: %lld", &fullsize);
#endif
      } else if (res.BeginsWith("Location:") && redirect) {
         redir = res(10, 1000);
         if (redirect == 2)   // temp redirect
            SetMsgReadBuffer10(redir, kTRUE);
         else               // permanent redirect
            SetMsgReadBuffer10(redir, kFALSE);
      }
   }

   if (redirect && redir.IsNull()) {
      Error("GetFromWeb10", "error - redirect without location from host %s", fUrl.GetHost());
   }

   if (n == -1 && fHTTP11) {
      if (gDebug > 0)
         Info("GetFromWeb10", "HTTP/1.1 socket closed, reopen");
      if (fBasicUrlOrg != "") {
         // if we have to close temp redirection, set back to original url
         SetMsgReadBuffer10();
      }
      ws.ReOpen();
      return GetFromWeb10(buf, len, msg);
   }

   if (ltot != len) {
      Error("GetFromWeb10", "error receiving expected amount of data (got %d, expected %d) from host %s",
            ltot, len, fUrl.GetHost());
      return -1;
   }

   // collect statistics
   fBytesRead += len;
   fReadCalls++;
#ifdef R__WIN32
   SetFileBytesRead(GetFileBytesRead() + len);
   SetFileReadCalls(GetFileReadCalls() + 1);
#else
   fgBytesRead += len;
   fgReadCalls++;
#endif

   if (gPerfStats)
      gPerfStats->FileReadEvent(this, len, start);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set position from where to start reading.

void TWebFile::Seek(Long64_t offset, ERelativeTo pos)
{
   switch (pos) {
   case kBeg:
      fOffset = offset + fArchiveOffset;
      break;
   case kCur:
      fOffset += offset;
      break;
   case kEnd:
      // this option is not used currently in the ROOT code
      if (fArchiveOffset)
         Error("Seek", "seeking from end in archive is not (yet) supported");
      fOffset = fEND - offset;  // is fEND really EOF or logical EOF?
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return maximum file size.

Long64_t TWebFile::GetSize() const
{
   if (!fHasModRoot || fSize >= 0)
      return fSize;

   Long64_t size;
   char     asize[64];

   TString msg = "GET ";
   msg += fBasicUrl;
   msg += "?";
   msg += -1;
   msg += "\r\n";

   if (const_cast<TWebFile*>(this)->GetFromWeb(asize, 64, msg) == -1)
      return kMaxInt;

#ifndef R__WIN32
   size = atoll(asize);
#else
   size = _atoi64(asize);
#endif

   fSize = size;

   return size;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the HTTP header. Depending on the return code we can see if
/// the file exists and if the server uses mod_root.
/// Returns -1 in case of an error, -2 in case the file does not exists,
/// -3 in case HEAD is not supported (dCache HTTP door) and
/// 0 in case of success.

Int_t TWebFile::GetHead()
{
   // Give full URL so Apache's virtual hosts solution works.
   if (fMsgGetHead == "") {
      fMsgGetHead = "HEAD ";
      fMsgGetHead += fBasicUrl;
      if (fHTTP11)
         fMsgGetHead += " HTTP/1.1";
      else
         fMsgGetHead += " HTTP/1.0";
      fMsgGetHead += "\r\n";
      if (fHTTP11) {
         fMsgGetHead += "Host: ";
         fMsgGetHead += fUrl.GetHost();
         fMsgGetHead += "\r\n";
      }
      fMsgGetHead += BasicAuthentication();
      fMsgGetHead += gUserAgent;
      fMsgGetHead += "\r\n\r\n";
   }
   TString msg = fMsgGetHead;

   TUrl connurl;
   if (fProxy.IsValid())
      connurl = fProxy;
   else
      connurl = fUrl;

   TSocket *s = 0;
   for (Int_t i = 0; i < 5; i++) {
      if (strcmp(connurl.GetProtocol(), "https") == 0) {
#ifdef R__SSL
         s = new TSSLSocket(connurl.GetHost(), connurl.GetPort());
#else
         Error("GetHead", "library compiled without SSL, https not supported");
         return -1;
#endif
      } else
         s = new TSocket(connurl.GetHost(), connurl.GetPort());

      if (!s->IsValid()) {
         delete s;
         if (gSystem->GetErrno() == EADDRINUSE || gSystem->GetErrno() == EISCONN) {
            s = 0;
            gSystem->Sleep(i*10);
         } else {
            Error("GetHead", "cannot connect to host %s (errno=%d)", fUrl.GetHost(),
                  gSystem->GetErrno());
            return -1;
         }
      } else
         break;
   }
   if (!s)
      return -1;

   if (gDebug > 0) {
      Info("GetHead", "connected to host %s", connurl.GetHost());
      Info("GetHead", "sending HTTP request:\n%s", msg.Data());
   }

   if (s->SendRaw(msg.Data(), msg.Length()) == -1) {
      Error("GetHead", "error sending command to host %s", fUrl.GetHost());
      delete s;
      return -1;
   }

   char line[8192];
   Int_t n, ret = 0, redirect = 0;
   TString redir;

   while ((n = GetLine(s, line, sizeof(line))) >= 0) {
      if (n == 0) {
         if (gDebug > 0)
            Info("GetHead", "got all headers");
         delete s;
         if (fBasicUrlOrg != "" && !redirect) {
            // set back to original url in case of temp redirect
            SetMsgReadBuffer10();
            fMsgGetHead = "";
         }
         if (ret < 0)
            return ret;
         if (redirect) {
            if (redir.IsNull()) {
               // Some sites (s3.amazonaws.com) do not return a Location field on 301
               Error("GetHead", "error - redirect without location from host %s", fUrl.GetHost());
               return -1;
            }
            return GetHead();
         }
         return 0;
      }

      if (gDebug > 0)
         Info("GetHead", "header: %s", line);

      TString res = line;
      ProcessHttpHeader(res);
      if (res.BeginsWith("HTTP/1.")) {
         if (res.BeginsWith("HTTP/1.1")) {
            if (!fHTTP11) {
               fMsgGetHead = "";
               fMsgReadBuffer10 = "";
            }
            fHTTP11 = kTRUE;
         }
         TString scode = res(9, 3);
         Int_t code = scode.Atoi();
         if (code >= 500) {
            if (code == 500)
               fHasModRoot = kTRUE;
            else {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetHead", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code >= 400) {
            if (code == 400)
               ret = -3;   // command not supported
            else if (code == 404)
               ret = -2;   // file does not exist
            else {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetHead", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code >= 300) {
            if (code == 301 || code == 303)
               redirect = 1;   // permanent redirect
            else if (code == 302 || code == 307)
               redirect = 2;   // temp redirect
            else {
               ret = -1;
               TString mess = res(13, 1000);
               Error("GetHead", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
            }
         } else if (code > 200) {
            ret = -1;
            TString mess = res(13, 1000);
            Error("GetHead", "%s: %s (%d)", fBasicUrl.Data(), mess.Data(), code);
         }
      } else if (res.BeginsWith("Content-Length:")) {
         TString slen = res(16, 1000);
         fSize = slen.Atoll();
      } else if (res.BeginsWith("Location:") && redirect) {
         redir = res(10, 1000);
         if (redirect == 2)   // temp redirect
            SetMsgReadBuffer10(redir, kTRUE);
         else               // permanent redirect
            SetMsgReadBuffer10(redir, kFALSE);
         fMsgGetHead = "";
      }
   }

   delete s;

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Read a line from the socket. Reads at most one less than the number of
/// characters specified by maxsize. Reading stops when a newline character
/// is found, The newline (\n) and cr (\r), if any, are removed.
/// Returns -1 in case of error, or the number of characters read (>= 0)
/// otherwise.

Int_t TWebFile::GetLine(TSocket *s, char *line, Int_t maxsize)
{
   Int_t n = GetHunk(s, line, maxsize);
   if (n < 0) {
      if (!fHTTP11 || gDebug > 0)
         Error("GetLine", "error receiving data from host %s", fUrl.GetHost());
      return -1;
   }

   if (n > 0 && line[n-1] == '\n') {
      n--;
      if (n > 0 && line[n-1] == '\r')
         n--;
      line[n] = '\0';
   }
   return n;
}

////////////////////////////////////////////////////////////////////////////////
/// Read a hunk of data from the socket, up until a terminator. The hunk is
/// limited by whatever the TERMINATOR callback chooses as its
/// terminator. For example, if terminator stops at newline, the hunk
/// will consist of a line of data; if terminator stops at two
/// newlines, it can be used to read the head of an HTTP response.
/// Upon determining the boundary, the function returns the data (up to
/// the terminator) in hunk.
///
/// In case of read error, -1 is returned. In case of having read some
/// data, but encountering EOF before seeing the terminator, the data
/// that has been read is returned, but it will (obviously) not contain the
/// terminator.
///
/// The TERMINATOR function is called with three arguments: the
/// beginning of the data read so far, the beginning of the current
/// block of peeked-at data, and the length of the current block.
/// Depending on its needs, the function is free to choose whether to
/// analyze all data or just the newly arrived data. If TERMINATOR
/// returns 0, it means that the terminator has not been seen.
/// Otherwise it should return a pointer to the character immediately
/// following the terminator.
///
/// The idea is to be able to read a line of input, or otherwise a hunk
/// of text, such as the head of an HTTP request, without crossing the
/// boundary, so that the next call to RecvRaw() etc. reads the data
/// after the hunk. To achieve that, this function does the following:
///
/// 1. Peek at incoming data.
///
/// 2. Determine whether the peeked data, along with the previously
///    read data, includes the terminator.
///
/// 3a. If yes, read the data until the end of the terminator, and
///     exit.
///
/// 3b. If no, read the peeked data and goto 1.
///
/// The function is careful to assume as little as possible about the
/// implementation of peeking.  For example, every peek is followed by
/// a read. If the read returns a different amount of data, the
/// process is retried until all data arrives safely.
///
/// Reads at most one less than the number of characters specified by maxsize.

Int_t TWebFile::GetHunk(TSocket *s, char *hunk, Int_t maxsize)
{
   if (maxsize <= 0) return 0;

   Int_t bufsize = maxsize;
   Int_t tail = 0;                 // tail position in HUNK

   while (1) {
      const char *end;
      Int_t pklen, rdlen, remain;

      // First, peek at the available data.
      pklen = s->RecvRaw(hunk+tail, bufsize-1-tail, kPeek);
      if (pklen < 0) {
         return -1;
      }
      end = HttpTerminator(hunk, hunk+tail, pklen);
      if (end) {
         // The data contains the terminator: we'll drain the data up
         // to the end of the terminator.
         remain = end - (hunk + tail);
         if (remain == 0) {
            // No more data needs to be read.
            hunk[tail] = '\0';
            return tail;
         }
         if (bufsize - 1 < tail + remain) {
            Error("GetHunk", "hunk buffer too small for data from host %s (%d bytes needed)",
                  fUrl.GetHost(), tail + remain + 1);
            hunk[tail] = '\0';
            return -1;
         }
      } else {
         // No terminator: simply read the data we know is (or should
         // be) available.
         remain = pklen;
      }

      // Now, read the data. Note that we make no assumptions about
      // how much data we'll get. (Some TCP stacks are notorious for
      // read returning less data than the previous MSG_PEEK.)
      rdlen = s->RecvRaw(hunk+tail, remain, kDontBlock);
      if (rdlen < 0) {
         return -1;
      }
      tail += rdlen;
      hunk[tail] = '\0';

      if (rdlen == 0) {
         // in case of EOF: return the data we've read.
         return tail;
      }
      if (end && rdlen == remain) {
         // The terminator was seen and the remaining data drained --
         // we got what we came for.
         return tail;
      }

      // Keep looping until all the data arrives.

      if (tail == bufsize - 1) {
         Error("GetHunk", "hunk buffer too small for data from host %s",
               fUrl.GetHost());
         return -1;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Determine whether [START, PEEKED + PEEKLEN) contains an HTTP new
/// line [\\r]\\n. If so, return the pointer to the position after the line,
/// otherwise return 0. This is used as callback to GetHunk(). The data
/// between START and PEEKED has been read and cannot be "unread"; the
/// data after PEEKED has only been peeked.

const char *TWebFile::HttpTerminator(const char *start, const char *peeked,
                                     Int_t peeklen)
{
#if 0
   const char *p, *end;

   // Look for "[\r]\n", and return the following position if found.
   // Start one char before the current to cover the possibility that
   // part of the terminator (e.g. "\r") arrived in the previous batch.
   p = peeked - start < 1 ? start : peeked - 1;
   end = peeked + peeklen;

   // Check for \r\n anywhere in [p, end-2).
   for (; p < end - 1; p++)
      if (p[0] == '\r' && p[1] == '\n')
         return p + 2;

   // p==end-1: check for \r\n directly preceding END.
   if (p[0] == '\r' && p[1] == '\n')
      return p + 2;
#else
   (void) start;   // start unused, silence compiler
   if (peeked) {
      const char *p = (const char*) memchr(peeked, '\n', peeklen);
      if (p)
         // p+1 because the line must include '\n'
         return p + 1;
   }
#endif
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return basic authentication scheme, to be added to the request.

TString TWebFile::BasicAuthentication()
{
   TString msg;
   if (strlen(fUrl.GetUser())) {
      TString auth = fUrl.GetUser();
      if (strlen(fUrl.GetPasswd())) {
         auth += ":";
         auth += fUrl.GetPasswd();
      }
      msg += "Authorization: Basic ";
      msg += TBase64::Encode(auth);
      msg += "\r\n";
   }
   return msg;
}

////////////////////////////////////////////////////////////////////////////////
/// Static method setting global proxy URL.

void TWebFile::SetProxy(const char *proxy)
{
   if (proxy && *proxy) {
      TUrl p(proxy);
      if (strcmp(p.GetProtocol(), "http")) {
         :: Error("TWebFile::SetProxy", "protocol must be HTTP in proxy URL %s",
                  proxy);
         return;
      }
      fgProxy = p;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Static method returning the global proxy URL.

const char *TWebFile::GetProxy()
{
   if (fgProxy.IsValid())
      return fgProxy.GetUrl();
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// Process the HTTP header in the argument. This method is intended to be
/// overwritten by subclasses that exploit the information contained in the
/// HTTP headers.

void TWebFile::ProcessHttpHeader(const TString&)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Static method returning maxmimal size of full cache,
/// which can be preserved by file instance

Long64_t TWebFile::GetMaxFullCacheSize()
{
   return fgMaxFullCacheSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Static method, set maxmimal size of full cache,
// which can be preserved by file instance

void TWebFile::SetMaxFullCacheSize(Long64_t sz)
{
   fgMaxFullCacheSize = sz;
}


////////////////////////////////////////////////////////////////////////////////
/// Create helper class that allows directory access via httpd.
/// The name must start with '-' to bypass the TSystem singleton check.

TWebSystem::TWebSystem() : TSystem("-http", "HTTP Helper System")
{
   SetName("http");

   fDirp = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Make a directory via httpd. Not supported.

Int_t TWebSystem::MakeDirectory(const char *)
{
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Open a directory via httpd. Returns an opaque pointer to a dir
/// structure. Returns 0 in case of error.

void *TWebSystem::OpenDirectory(const char *)
{
   if (fDirp) {
      Error("OpenDirectory", "invalid directory pointer (should never happen)");
      fDirp = 0;
   }

   fDirp = 0;   // not implemented for the time being

   return fDirp;
}

////////////////////////////////////////////////////////////////////////////////
/// Free directory via httpd.

void TWebSystem::FreeDirectory(void *dirp)
{
   if (dirp != fDirp) {
      Error("FreeDirectory", "invalid directory pointer (should never happen)");
      return;
   }

   fDirp = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get directory entry via httpd. Returns 0 in case no more entries.

const char *TWebSystem::GetDirEntry(void *dirp)
{
   if (dirp != fDirp) {
      Error("GetDirEntry", "invalid directory pointer (should never happen)");
      return 0;
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get info about a file. Info is returned in the form of a FileStat_t
/// structure (see TSystem.h).
/// The function returns 0 in case of success and 1 if the file could
/// not be stat'ed.

Int_t TWebSystem::GetPathInfo(const char *path, FileStat_t &buf)
{
   TWebFile *f = new TWebFile(path, "HEADONLY");

   if (f->fWritten == 0) {

      buf.fDev    = 0;
      buf.fIno    = 0;
      buf.fMode   = 0;
      buf.fUid    = 0;
      buf.fGid    = 0;
      buf.fSize   = f->GetSize();
      buf.fMtime  = 0;
      buf.fIsLink = kFALSE;

      delete f;
      return 0;
   }

   delete f;
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns FALSE if one can access a file using the specified access mode.
/// Mode is the same as for the Unix access(2) function.
/// Attention, bizarre convention of return value!!

Bool_t TWebSystem::AccessPathName(const char *path, EAccessMode)
{
   TWebFile *f = new TWebFile(path, "HEADONLY");
   if (f->fWritten == 0) {
      delete f;
      return kFALSE;
   }
   delete f;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlink, i.e. remove, a file or directory. Returns 0 when successful,
/// -1 in case of failure. Not supported for httpd.

Int_t TWebSystem::Unlink(const char *)
{
   return -1;
}
