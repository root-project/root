// @(#)root/netx:$Id$
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetFile                                                            //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
// Interfaced to the standalone client (XrdClient): G. Ganis, CERN      //
//                                                                      //
// TXNetFile is an extension of TNetFile able to deal with new xrootd   //
// server. Its new features are:                                        //
//  - Automatic server kind recognition (xrootd load balancer, xrootd   //
//    data server, old rootd)                                           //
//  - Backward compatibility with old rootd server (acts as an old      //
//    TNetFile)                                                         //
//  - Fault tolerance for read/write operations (read/write timeouts    //
//    and retry)                                                        //
//  - Internal connection timeout (tunable indipendently from the OS    //
//    one) handled by threads                                           //
//  - handling of redirections from server                              //
//  - Single TCP physical channel for multiple TXNetFile's instances    //
//    inside the same application                                       //
//    So, each TXNetFile object client must send messages containing    //
//    its ID (streamid). The server, of course, will respond with       //
//    messages containing the client's ID, in order to make the client  //
//    able to recognize its message by matching its streamid with that  //
//    one contained in the server's response.                           //
//  - Tunable log verbosity level (0 = nothing, 3 = dump read/write     //
//    buffers too!)                                                     //
//  - Many parameters configurable via TEnv facility (see SetParm()     //
//    methods)                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Bytes.h"

#include "TError.h"
#include "TEnv.h"
#include "TSocket.h"
#include "TXNetFile.h"
#include "TROOT.h"
#include "TVirtualMonitoring.h"
#include "TFileStager.h"
#include "TFileCacheRead.h"
#include "TTimeStamp.h"
#include "TVirtualPerfStats.h"

#include <XrdClient/XrdClient.hh>
#include <XrdClient/XrdClientConst.hh>
#include <XrdClient/XrdClientEnv.hh>
#include <XProtocol/XProtocol.hh>
#ifdef OLDXRDOUC
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif

ClassImp(TXNetFile);

Bool_t TXNetFile::fgInitDone = kFALSE;
Bool_t TXNetFile::fgRootdBC = kTRUE;
TFileStager *TXNetFile::fgFileStager = 0;

//_____________________________________________________________________________
TXNetFile::TXNetFile(const char *url, Option_t *option, const char* ftitle,
                     Int_t compress, Int_t netopt, Bool_t parallelopen,
                     const char *logicalurl) :
            TNetFile((logicalurl ? logicalurl : url), ftitle, compress, kFALSE)
{
   // Create a TXNetFile object. A TXNetFile object is the same as a TNetFile
   // (from which the former derives) except that the protocol is extended to
   // support dealing with new xrootd data server or xrootd load balancer
   // server.
   //
   // The "url" argument must be of the form
   //
   //   root://server1:port1[,server2:port2,...,serverN:portN]/pathfile,
   //
   // Note that this means that multiple servers (>= 1) can be specified in
   // the url. The connection will try to connect to the first server:port
   // and if that does not succeed, it will try the second one, and so on
   // until it finds a server that will respond.
   //
   // See the TNetFile documentation for the description of the other arguments.
   //
   // The creation consists of internal variable settings (most important is
   // the client's domain), creation of a TXUrl array containing all specified
   // urls (a single url is serverX:portX/pathfile), trying to connect to the
   // servers calling Connect() method, getting a valid access to the remote
   // server the client is connected to using GetAccessToSrv() method,
   // recognizing the remote server (if an old rootd the TNetFile's Create
   // method will be called).
   //
   // The options field of the URL can be used for the following purposes:
   //   a. open a non-ROOT generic file
   //      "root://server1:port1[,server2:port2,...]/pathfile?filetype=raw"
   //   b. re-check the environment variables
   //      "root://server1:port1[,server2:port2,...]/pathfile?checkenv"
   //   c. set the cache size (in bytes)
   //      "root://server1:port1[,server2:port2,...]/pathfile?cachesz=20000000"
   //   d. set the read-ahead size (in bytes)
   //      "root://server1:port1[,server2:port2,...]/pathfile?readaheadsz=100000"
   //   e. set the cache remove policy
   //      "root://server1:port1[,server2:port2,...]/pathfile?rmpolicy=1"
   //   f. set the max number of redirections
   //      "root://server1:port1[,server2:port2,...]/pathfile?mxredir=2"
   // (multiple options can be set concurrently)
   TUrl urlnoanchor(url);
   // Set debug level
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", -1));

   // Set environment, if needed
   if (!fgInitDone || strstr(urlnoanchor.GetOptions(),"checkenv")) {
      SetEnv();
      fgInitDone = kTRUE;

      // Print the tag, if required (only once)
      if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
         Info("TXNetFile","(eXtended TNetFile) %s",
              gROOT->GetVersion());
   }

   // Remove anchors from the URL!
   urlnoanchor.SetAnchor("");

   // Init mutex used in the asynchronous open machinery
   fInitMtx = new XrdSysRecMutex();

   if (gMonitoringWriter) {
      // Init the monitoring system
      if (!fOpenPhases) {
         fOpenPhases = new TList;
         fOpenPhases->SetOwner();
      }
      // Should not be null instead of "xrdopen" to init the thing ?
      gMonitoringWriter->SendFileOpenProgress(this, fOpenPhases, "xrdopen", kFALSE);
   }

   // Create an instance
   CreateXClient(urlnoanchor.GetUrl(), option, netopt, parallelopen);
}

//_____________________________________________________________________________
TXNetFile::~TXNetFile()
{
   // Destructor.

   if (IsOpen())
      Close(0);

   SafeDelete(fClient);
   SafeDelete(fInitMtx);
}

//_____________________________________________________________________________
void TXNetFile::FormUrl(TUrl uu, TString &uus)
{
   // Form url for rootd socket.

   // Protocol
   uus = "root://";

   // User, if any
   if (strlen(uu.GetUser()) > 0) {
      uus += uu.GetUser();
      uus += "@";
   }

   // Host, if any
   if (strlen(uu.GetHost()) > 0) {
      uus += uu.GetHost();
   }

   // Port, if any
   if (uu.GetPort() > 0) {
      uus += ":";
      uus += uu.GetPort();
   }

   // End of string
   uus += "/";
}

//_____________________________________________________________________________
Int_t TXNetFile::ParseOptions(const char *opts,
                              Int_t &cachesz, Int_t &readaheadsz,
                              Int_t &rmpolicy, Int_t &mxredir, Int_t &rastrategy, Int_t &readtrimblksz)
{
   // Parse input options for cache parameters
   static const char *keys[6] = { "cachesz=", "readaheadsz=", "rmpolicy=",
                                  "mxredir=", "readaheadstrategy=", "readtrimblksz=" };
   Int_t fo = 0;
   TString s(opts);

   UInt_t i = 0;
   for (i = 0; i < (sizeof(keys)/sizeof(keys[0])); i++) {
      Int_t j = s.Index(keys[i]);
      if (j != kNPOS) {
         TString val(s(j+strlen(keys[i]), s.Length()));
         // Cut of non digits
         Int_t k = 0;
         while (k < val.Length())
            if (!TString(val(k++)).IsDigit())
               break;
         if (k < val.Length())
            val.Remove(--k);
         if (val.IsDigit()) {
            fo++;
            if (i == 0)
               cachesz = val.Atoi();
            else if (i == 1)
               readaheadsz = val.Atoi();
            else if (i == 2)
               rmpolicy = val.Atoi();
            else if (i == 3)
               mxredir = val.Atoi();
            else if (i == 4)
               rastrategy = val.Atoi();
            else if (i == 5)
               readtrimblksz = val.Atoi();
         }
      }
   }

   // Notify
   if (gDebug > 0)
      Info("ParseCacheOptions","found: cachesz = %d, readaheadsz = %d, "
           "rmpolicy = %d, mxredir = %d, rastrategy = %d, readtrimblksz = %d",
           cachesz, readaheadsz, rmpolicy, mxredir, rastrategy, readtrimblksz);

   // Done
   return fo;
}

//_____________________________________________________________________________
void TXNetFile::CreateXClient(const char *url, Option_t *option, Int_t netopt,
                              Bool_t parallelopen)
{
   // The real creation work is done here.

   Int_t cachesz = -1, readaheadsz = -1, rmpolicy = -1, mxredir = -1, np = 0;
   Int_t readaheadstrategy = -1, readtrimblksz = -1;

   fClient = 0;
   fNetopt = netopt;

   // Set the timeout (default 999999999 secs, i.e. far, far in the future)
   gSystem->Setenv("XRDCLIENTMAXWAIT", Form("%d",TFile::GetOpenTimeout()));

   if (GetOnlyStaged()) {
      // Check if the file is staged before opening it
      if (!fgFileStager || !(fgFileStager->Matches(url))) {
         SafeDelete(fgFileStager);
         fgFileStager = TFileStager::Open(url);
      }
      if (fgFileStager) {
         if (!(fgFileStager->IsStaged(url))) {
            ::Warning("TXNetFile","<%s> is not staged - StageOnly flag is set!",url);
            goto zombie;
         }
      }
   }

   // Init members
   fIsRootd = kFALSE;

   // The parallel open can be forced to true in the config
   if (gEnv->GetValue("XNet.ForceParallelOpen", 0))
      parallelopen = kTRUE;
   fAsyncOpenStatus = (parallelopen) ? kAOSInProgress : fAsyncOpenStatus ;

   Bool_t isRootd;
   isRootd = kFALSE;
   //
   // Setup a client instance
   fClient = new XrdClient(url);
   if (!fClient) {
      fAsyncOpenStatus = (parallelopen) ? kAOSFailure : fAsyncOpenStatus ;
      Error("CreateXClient","fatal error: new object creation failed -"
            " out of system resources.");
      gSystem->Abort();
      goto zombie;
   }

   // Get client (cache, redir) parameters, if any
   np = ParseOptions(TUrl(url).GetOptions(),
                     cachesz, readaheadsz, rmpolicy, mxredir,
                     readaheadstrategy, readtrimblksz);

   // Set max redir, if asked
   if (mxredir > 0) {
      if (fClient->GetClientConn()) {
         if (gDebug > 0)
            Info("CreateXClient", "setting maxredir = %d", mxredir);
         fClient->GetClientConn()->SetMaxRedirCnt(mxredir);
      }
      np--;
   }
   // Set the cache parameters, if any
   if (np > 0) {
      if (gDebug > 0)
         Info("CreateXClient", "setting cachesz = %d, readaheadsz = %d, "
                               "rmpolicy = %d",
                               cachesz, readaheadsz, rmpolicy);
      fClient->SetCacheParameters(cachesz, readaheadsz, rmpolicy);

      if (readaheadstrategy >= 0) {
         if (gDebug > 0)
            Info("CreateXClient", "setting readaheadstrategy = %d", readaheadstrategy);
         fClient->SetReadAheadStrategy(readaheadstrategy);
      }

      if (readtrimblksz >= 0) {
         if (gDebug > 0)
            Info("CreateXClient", "setting readtrimblksz = %d", readtrimblksz);
         fClient->SetBlockReadTrimming(readtrimblksz);
      }

   }

   //
   // Now try opening the file
   if (!Open(option, parallelopen)) {
      if (!fClient->IsOpen_wait()) {
         if (gDebug > 1)
            Info("CreateXClient", "remote file could not be open");

         // If the server is a rootd we need to create a TNetFile
         isRootd = (fClient->GetClientConn()->GetServerType() == kSTRootd);

         if (isRootd) {
            if (fgRootdBC) {

               Int_t sd = fClient->GetClientConn()->GetOpenSockFD();
               if (sd > -1) {
                  //
                  // Create a TSocket on the open connection
                  TSocket *s = new TSocket(sd);

                  s->SetOption(kNoBlock, 0);

                  // Find out the remote protocol (send the client protocol first)
                  Int_t rproto = GetRootdProtocol(s);
                  if (rproto < 0) {
                     Error("CreateXClient", "getting rootd server protocol");
                     goto zombie;
                  }

                  // Finalize TSocket initialization
                  s->SetRemoteProtocol(rproto);
                  TUrl uut((fClient->GetClientConn()
                                   ->GetCurrentUrl()).GetUrl().c_str());
                  TString uu;
                  FormUrl(uut,uu);

                  if (gDebug > 2)
                     Info("CreateXClient"," url: %s",uu.Data());
                  s->SetUrl(uu.Data());
                  s->SetService("rootd");
                  s->SetServType(TSocket::kROOTD);
                  //
                  // Set rootd flag
                  fIsRootd = kTRUE;
                  //
                  // Now we can check if we can create a TNetFile on the
                  // open connection
                  if (rproto > 13) {
                     //
                     // Remote support for reuse of open connection
                     TNetFile::Create(s, option, netopt);
                  } else {
                     //
                     // Open connection has been closed because could
                     // not be reused; TNetFile will open a new connection
                     TNetFile::Create(uu.Data(), option, netopt);
                  }

                  return;
               } else {
                  Error("CreateXClient", "rootd: underlying socket undefined");
                  goto zombie;
               }
            } else {
               if (gDebug > 0)
                  Info("CreateXClient", "rootd: fall back not enabled - closing");
               goto zombie;
            }
         } else {
            Error("CreateXClient", "open attempt failed on %s", fUrl.GetUrl());
            goto zombie;
         }
      }
   }

   return;

zombie:
   // error in file opening occured, make this object a zombie
   SafeDelete(fClient);
   MakeZombie();
   gDirectory = gROOT;
}

//_____________________________________________________________________________
Int_t TXNetFile::GetRootdProtocol(TSocket *s)
{
   // Find out the remote rootd protocol version.
   // Returns -1 in case of error.

   Int_t rproto = -1;

   UInt_t cproto = 0;
   Int_t len = sizeof(cproto);
   memcpy((char *)&cproto,
      Form(" %d", TSocket::GetClientProtocol()),len);
   Int_t ns = s->SendRaw(&cproto, len);
   if (ns != len) {
      ::Error("TXNetFile::GetRootdProtocol",
              "sending %d bytes to rootd server [%s:%d]",
              len, (s->GetInetAddress()).GetHostName(), s->GetPort());
      return -1;
   }

   // Get the remote protocol
   Int_t ibuf[2] = {0};
   len = sizeof(ibuf);
   Int_t nr = s->RecvRaw(ibuf, len);
   if (nr != len) {
      ::Error("TXNetFile::GetRootdProtocol",
              "reading %d bytes from rootd server [%s:%d]",
              len, (s->GetInetAddress()).GetHostName(), s->GetPort());
      return -1;
   }
   Int_t kind = net2host(ibuf[0]);
   if (kind == kROOTD_PROTOCOL) {
      rproto = net2host(ibuf[1]);
   } else {
      kind = net2host(ibuf[1]);
      if (kind == kROOTD_PROTOCOL) {
         len = sizeof(rproto);
         nr = s->RecvRaw(&rproto, len);
         if (nr != len) {
            ::Error("TXNetFile::GetRootdProtocol",
                    "reading %d bytes from rootd server [%s:%d]",
                    len, (s->GetInetAddress()).GetHostName(), s->GetPort());
            return -1;
         }
         rproto = net2host(rproto);
      }
   }
   if (gDebug > 2)
      ::Info("TXNetFile::GetRootdProtocol",
             "remote rootd: buf1: %d, buf2: %d rproto: %d",
             net2host(ibuf[0]),net2host(ibuf[1]),rproto);

   // We are done
   return rproto;
}

//_____________________________________________________________________________
Bool_t TXNetFile::Open(Option_t *option, Bool_t doitparallel)
{
   // The real creation work is done here.

   //
   // Parse options
   kXR_unt16 openOpt = 0;
   memset(&openOpt, 0, sizeof(openOpt));
   TString opt = option;
   opt.ToUpper();
   //
   // Check force, accepting 'f'/'F' for backward compatibility,
   // and special read syntax
   if (opt.BeginsWith("-") || opt.BeginsWith("F") || (opt == "+READ")) {
      opt.Remove(0,1);
      openOpt |= kXR_force;
   }
   //
   // Read flag
   Bool_t read = (opt == "READ");
   //
   // Create flag ("NEW" == "CREATE")
   Bool_t create = (opt == "CREATE" || opt == "NEW");
   //
   // Recreate flag
   Bool_t recreate = (opt == "RECREATE");
   //
   // Update flag
   Bool_t update = (opt == "UPDATE");
   //
   // Default is Read
   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      opt = "READ";
   }
   //
   // Save effective options
   fOption = opt;
   if (create || update || recreate)
      fWritable = 1;
   //
   // Update requires the file existing: check that and switch to create,
   // if the file is not found.
   if (update) {
      if (gSystem->AccessPathName(fUrl.GetUrl(), kFileExists)) {
         update = kFALSE;
         create = kTRUE;
      }
      if (update) {
         if (gSystem->AccessPathName(fUrl.GetUrl(), kWritePermission)) {
            Error("Open", "no write permission, could not open file %s",
                          fUrl.GetUrl());
            fAsyncOpenStatus = (doitparallel) ? kAOSFailure : fAsyncOpenStatus ;
            return kFALSE;
         }
         openOpt |= kXR_open_updt;
      }
   }

   //
   // Create and Recreate are correlated
   if (create)
      openOpt |= kXR_new;
   if (recreate) {
      openOpt |= kXR_delete;
      create = kTRUE;
   }

   Bool_t mkpath = (gEnv->GetValue("XNet.Mkpath", 0) == 1) ? kTRUE : kFALSE;
   char *p = (char*)strstr(fUrl.GetOptions(), "mkpath=");
   if (p)
     mkpath = (*(p + strlen("mkpath=")) == '1') ? kTRUE : kFALSE;
   if (mkpath)
     openOpt |= kXR_mkpath;

   if (read)
      openOpt |= kXR_open_read;

   //
   // Set open mode to rw-r-r
   kXR_unt16 openMode = kXR_or | kXR_gr | kXR_ur | kXR_uw;

   //
   // Open file (FileOpenerThread disabled for the time being)
   if (!fClient->Open(openMode, openOpt, doitparallel)) {
      if (gDebug > 1)
         Info("Open", "remote file could not be open");
      fAsyncOpenStatus = (doitparallel) ? kAOSFailure : fAsyncOpenStatus ;
      return kFALSE;
   } else {
      // Initialize the file
      // If we are using the parallel open, the init phase is
      // performed later. In checking for the IsOpen or
      // asynchronously in a callback func
      if (!doitparallel) {
         // Mutex serialization is done inside
         Init(create);
         // If initialization failed close everything
         if (TFile::IsZombie()) {
            fClient->Close();
            // To avoid problems in final deletion of object not completely
            // initialized
            fWritable = 0;
            // Notify failure
            return kFALSE;
         }
      }
   }

   // We are done
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXNetFile::ReadBuffer(char *buffer, Int_t bufferLength)
{
   // Override TNetFile::ReadBuffer to deal with the xrootd server.
   // Returns kTRUE in case of errors.

   if (IsZombie()) {
      Error("ReadBuffer", "ReadBuffer is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("ReadBuffer","Calling TNetFile::ReadBuffer");
      return TNetFile::ReadBuffer(buffer, bufferLength);
   }

   if (!IsOpen()) {
      Error("ReadBuffer","The remote file is not open");
      return kTRUE;
   }

   Bool_t result = kFALSE;

   if (bufferLength==0)
      return 0;

   // This returns:
   //  2 if errors
   //  1 it looks like the block has already been prefetched
   //  0 it looks like the block has not been prefetched
   // But we don't want it to return the buffer, to avoid recursion
   Int_t st = 0;

   //using the new method to read
   if (GetCacheRead() && GetCacheRead()->IsEnablePrefetching()) {
      st = ReadBufferViaCache(buffer, bufferLength);   //modify to "buffer" so that it work with the ne version!!!
      if (st == 1){
         fOffset -= bufferLength;
         return kFALSE;
      }
   }
   else{ //using the old method to read
   if (GetCacheRead() && GetCacheRead()->IsAsyncReading()) {
      st = ReadBufferViaCache(0, bufferLength);
      if (st == 1)
         fOffset -= bufferLength;
   } else {
      if (GetCacheRead()) {
         st = ReadBufferViaCache(buffer, bufferLength);
         if (st == 1)
            return kFALSE;
      }
   }
   }

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   // Read from the remote xrootd
   Int_t nr = fClient->Read(buffer, fOffset, bufferLength);

   if (!nr)
      return kTRUE;

   if (gDebug > 1)
      Info("ReadBuffer", "%d bytes of data read from offset"
                         " %lld (%d requested)", nr, fOffset, bufferLength);

   fOffset += bufferLength;

     fBytesRead += nr;
     fReadCalls++;
#ifdef WIN32
     SetFileBytesRead(GetFileBytesRead() + nr);
     SetFileReadCalls(GetFileReadCalls() + 1);
#else
     fgBytesRead += nr;
     fgReadCalls++;
#endif

     if (gPerfStats)
        gPerfStats->FileReadEvent(this, bufferLength, start);

   if (gMonitoringWriter)
      gMonitoringWriter->SendFileReadProgress(this);

   return result;
}

//_____________________________________________________________________________
Bool_t TXNetFile::ReadBuffer(char *buffer, Long64_t pos, Int_t bufferLength)
{
   // Pass through to TNetFile implementation which will call back eventually
   // to our ReadBuffer with 2 arguments to deal with xrootd errors.

   return TNetFile::ReadBuffer(buffer, pos, bufferLength);
}

//______________________________________________________________________________
Bool_t TXNetFile::ReadBufferAsync(Long64_t offs, Int_t bufferLength)
{
   // Implementation dealing with the xrootd server.
   // Returns kTRUE in case of errors.
   // This is the same as TXNetFile::ReadBuffer but using the async
   // call from xrootd

   if (IsZombie()) {
      Error("ReadBuffer", "ReadBuffer is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Error("ReadBufferAsync","Not supported for rootd");
      return kTRUE;
   }

   if (!IsOpen()) {
      Error("ReadBuffer","The remote file is not open");
      return kTRUE;
   }

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   Bool_t result = kFALSE;

   if (bufferLength==0)
      return 0;

   SynchronizeCacheSize();

   // Read for the remote xrootd
   // This doesnt return the number of bytes read...
   // and even if it did we dont want to update fBytesRead
   // because that would be updated in the real read
   XReqErrorType nr = fClient->Read_Async(offs+fArchiveOffset, bufferLength);

   if (nr != kOK)
      return kTRUE;

   fBytesRead += bufferLength;
   fReadCalls++;
#ifdef WIN32
   SetFileBytesRead(GetFileBytesRead() + bufferLength);
   SetFileReadCalls(GetFileReadCalls() + 1);
#else
   fgBytesRead += bufferLength;
   fgReadCalls++;
#endif

   if (gPerfStats)
      gPerfStats->FileReadEvent(this, bufferLength, start);

   if (gDebug > 1)
      Info("ReadBufferAsync", "%d bytes of data read request from offset"
                              " %lld", bufferLength, offs);
   return result;
}

//______________________________________________________________________________
Bool_t TXNetFile::ReadBuffers(char *buf,  Long64_t *pos, Int_t *len, Int_t nbuf)
{
   // Read the nbuf blocks described in arrays pos and len,
   // where pos[i] is the seek position of block i of length len[i].
   // Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer
   // This function is overloaded by TNetFile, TWebFile, etc.
   // Returns kTRUE in case of failure.
   // Note: This is the overloading made in TXNetFile, If ReadBuffers
   // is supported by xrootd it will try to gt the whole list from one single
   // call avoiding the latency of multiple calls

   if (IsZombie()) {
      Error("ReadBuffers", "ReadBuffers is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("ReadBuffers","Calling TNetFile::ReadBuffers");
      return TNetFile::ReadBuffers(buf, pos, len, nbuf);
   }

   if (!IsOpen()) {
      Error("ReadBuffers","The remote file is not open");
      return kTRUE;
   }

   Double_t start = 0;
   if (gPerfStats) start = TTimeStamp();

   if (fArchiveOffset)
      for (Int_t i = 0; i < nbuf; i++)
         pos[i] += fArchiveOffset;

   // A null buffer means that we want to use the async stuff
   //  hence we have to sync the cache size in XrdClient with the supposed
   //  size in TFile.
   if (!buf) {
      // Null buffer + 0 blocks means 'reset cache'
      if (!nbuf) ResetCache();
      SynchronizeCacheSize();
   }

   // Read for the remote xrootd
   Long64_t nr = fClient->ReadV(buf, pos, len, nbuf);

   if (gDebug > 1)
      Info("ReadBuffers", "response from ReadV(%d) nr: %lld", nbuf, nr);

   if (nr > 0) {

      if (gDebug > 1)
         Info("ReadBuffers", "%lld bytes of data read from a list of %d buffers",
              nr, nbuf);

      if (GetCacheRead() && GetCacheRead()->GetBufferSize() < nr)
         Info("ReadBuffers", "%lld bytes of data read with a smaller (%d) TFileCacheRead buffer size?",
              nr, GetCacheRead()->GetBufferSize());

      // Where should we leave the offset ?
      // fOffset += bufferLength;
      fBytesRead += nr;
      fReadCalls++;
#ifdef WIN32
      SetFileBytesRead(GetFileBytesRead() + nr);
      SetFileReadCalls(GetFileReadCalls() + 1);
#else
      fgBytesRead += nr;
      fgReadCalls++;
#endif

      if (gPerfStats) {
         fOffset = pos[0];
         gPerfStats->FileReadEvent(this, pos[nbuf-1]+len[nbuf-1]-pos[0], start);
      }

      if (gMonitoringWriter)
         gMonitoringWriter->SendFileReadProgress(this);

      return kFALSE;
   }

   if (gDebug > 1)
      Info("ReadBuffers", "XrdClient->ReadV failed, executing TFile::ReadBuffers");

   // If it wasnt able to use the specialized call
   // then use the generic one that is a plain loop
   // of individual requests
   if (buf && nbuf)
      return TFile::ReadBuffers(buf, pos, len, nbuf);
   // If the async call was needed (buf == 0) and it got an error,
   // just return error
   else return kTRUE;
}

//_____________________________________________________________________________
Bool_t TXNetFile::WriteBuffer(const char *buffer, Int_t bufferLength)
{
   // Override TNetFile::WriteBuffer to deal with the xrootd server.
   // Returns kTRUE in case of errors.

   if (IsZombie()) {
      Error("WriteBuffer", "WriteBuffer is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   if (!fWritable) {
      if (gDebug > 1)
         Info("WriteBuffer","file not writable");
      return kTRUE;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("WriteBuffer","Calling TNetFile::WriteBuffer");
      return TNetFile::WriteBuffer(buffer, bufferLength );
   }

   if (!IsOpen()) {
      Error("WriteBuffer","The remote file is not open");
      return kTRUE;
   }

   Int_t st;
   if ((st = WriteBufferViaCache(buffer, bufferLength))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   // Read for the remote xrootd
   if (!fClient->Write(buffer, fOffset, bufferLength)) {
      if (gDebug > 0)
         Info("WriteBuffer",
              "error writing %d bytes of data wrote to offset %lld",
              bufferLength , fOffset);
      return kTRUE;
   }

   if (gDebug > 1)
      Info("WriteBuffer", " %d bytes of data wrote to offset"
                         " %lld", bufferLength , fOffset);

   fOffset += bufferLength;
   fBytesWrite += bufferLength;
#ifdef WIN32
   SetFileBytesWritten(GetFileBytesWritten() + bufferLength);
#else
   fgBytesWrite += bufferLength;
#endif

   return kFALSE;
}

//_____________________________________________________________________________
void TXNetFile::Init(Bool_t create)
{
   // Initialize the file. Makes sure that the file is really open before
   // calling TFile::Init. It may block.

   if (fInitDone) {
      // TFile::Init already called once
      if (gDebug > 1)
         Info("Init","TFile::Init already called once");
      return;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("Init","rootd: calling directly TFile::Init");
      return TNetFile::Init(create);
   }

   if (fClient) {
      // A mutex serializes this very delicate section
      XrdSysMutexHelper m(fInitMtx);

      // To safely perform the Init() we must make sure that
      // the file is successfully open; this call may block
      if (fClient->IsOpen_wait()) {

         // Notify the monitoring system
         if (gMonitoringWriter)
            gMonitoringWriter->SendFileOpenProgress(this, fOpenPhases, "rootinit", kFALSE);

         // Avoid big transfers at this level
         bool usecachesave = fClient->UseCache(0);
         // Note that Init will trigger recursive calls
         TFile::Init(create);
         // so TFile::IsOpen() returns true when in TFile::~TFile
         fD = -2;
         // Restore requested behaviour
         fClient->UseCache(usecachesave);

            // Notify the monitoring system
         if (gMonitoringWriter)
            gMonitoringWriter->SendFileOpenProgress(this, fOpenPhases, "endopen", kTRUE);

         // Set the Endpoint Url we are now connected to. Unless there was some opaque info
         // which cannot be re-used
         if (fClient->GetClientConn() && fClient->GetClientConn()->fRedirOpaque.length() <= 0) {
            fEndpointUrl = fClient->GetClientConn()->GetCurrentUrl().GetUrl().c_str();
            // Check equivalence of initial and end-point Url to see whether we have
            // been redirected
            if (fEndpointUrl.GetPort() != fUrl.GetPort() ||
               strcmp(fEndpointUrl.GetHostFQDN(), fUrl.GetHostFQDN()))
               SetBit(TFile::kRedirected);
         }
      } else {
         if (gDebug > 0)
            Info("Init","open request failed!");
         SafeDelete(fClient);
         MakeZombie();
         gDirectory = gROOT;
      }
   }
}

//_____________________________________________________________________________
Bool_t TXNetFile::IsOpen() const
{
   // Return kTRUE if the file is open, kFALSE otherwise.

   if (fIsRootd) {
      if (gDebug > 1)
         Info("IsOpen","Calling TNetFile::IsOpen");
      return TNetFile::IsOpen();
   }

   if (!fClient)
      return kFALSE;

   // We are done
   return ((fClient && fInitDone) ? fClient->IsOpen() : kFALSE);
}

//_____________________________________________________________________________
TFile::EAsyncOpenStatus TXNetFile::GetAsyncOpenStatus()
{
   // Return status of asynchronous request

   if (fAsyncOpenStatus != TFile::kAOSNotAsync) {
      if (fClient->IsOpen_inprogress()) {
         return TFile::kAOSInProgress;
      } else {
         if (fClient->IsOpen())
            return TFile::kAOSSuccess;
         else
            return TFile::kAOSFailure;
      }
   }

   // Not asynchronous
   return TFile::kAOSNotAsync;
}

//_____________________________________________________________________________
Int_t TXNetFile::ReOpen(const Option_t *Mode)
{
   // Re-open the file (see TNetFile::ReOpen() or TFile::ReOpen()
   // for more details).

   if (fIsRootd) {
      if (gDebug > 1)
         Info("ReOpen","Calling TNetFile::ReOpen");
      return TNetFile::ReOpen(Mode);
   }

   return TFile::ReOpen(Mode);
}

//_____________________________________________________________________________
void TXNetFile::Close(const Option_t *opt)
{
   // Close the file (see TNetFile::Close() or TFile::Close()
   // for more details).

   if (fIsRootd) {
      if (gDebug > 1)
         Info("Close","Calling TNetFile::Close");
      TNetFile::Close(opt);
      return;
   }

   if (!fClient) return;

   TFile::Close(opt);

   fIsRootd = kFALSE;

   if (IsOpen())
      fClient->Close();

   fD = -1;  // so TFile::IsOpen() returns false when in TFile::~TFile
}

//_____________________________________________________________________________
void TXNetFile::Flush()
{
   // Flushes un-written data.
   if (IsZombie()) {
      Error("Flush", "Flush is not possible because object is"
            " in 'zombie' state");
      return;
   }

   if (!fWritable) {
      if (gDebug > 1)
         Info("Flush", "file not writable - do nothing");
      return;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("Flush","Calling TNetFile::Flush");
      TNetFile::Flush();
      return;
   }

   if (!IsOpen()) {
      Error("Flush","The remote file is not open");
      return;
   }

   FlushWriteCache();

   //
   // Flush via the remote xrootd
   fClient->Sync();
   if (gDebug > 1)
      Info("Flush", "XrdClient::Sync called.");
}

//_____________________________________________________________________________
Int_t TXNetFile::SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags,
                          Long_t *modtime)
{
   // Override TNetFile::SysStat (see parent's method for more details).

   if (IsZombie()) {
      Error("SysStat", "SysStat is not possible because object is"
            " in 'zombie' state");
      *size = 0;
      return 1;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("SysStat", "calling TNetFile::SysStat");
      return TNetFile::SysStat(fd, id, size, flags, modtime);
   }

   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo().
   struct XrdClientStatInfo stinfo;
   if (fClient && fClient->Stat(&stinfo)) {
      *id = (Long_t)(stinfo.id);
      *size = (Long64_t)(stinfo.size);
      *flags = (Long_t)(stinfo.flags);
      *modtime = (Long_t)(stinfo.modtime);
      if (gDebug > 1)
         Info("SysStat", "got stats = %ld %lld %ld %ld",
                         *id, *size, *flags, *modtime);
   } else {

      if (gDebug > 1) {
         if (!IsOpen()) Info("SysStat", "could not stat remote file. Not opened.");
         else
            Info("SysStat", "could not stat remote file");
      }


      *id = -1;
      return 1;
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysClose(Int_t fd)
{
   // Override TNetFile::SysClose (see parent's method for more details).

   if (IsZombie()) {
      Error("SysClose", "SysClose is not possible because object is"
            " in 'zombie' state");
      return 0;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("SysClose","Calling TNetFile::SysClose");
      return TNetFile::SysClose(fd);
   }

   // Send close to remote xrootd
   if (IsOpen())
      fClient->Close();

   return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysOpen(const char* pathname, Int_t flags, UInt_t mode)
{
   // Override TNetFile::SysOpen (see parent's method for more details).

   if (fIsRootd) {
      if (gDebug > 1)
         Info("SysOpen", "Calling TNetFile::SysOpen");
      return TNetFile::SysOpen(pathname, flags, mode);
   }

   if (!fClient) {

      // Create an instance of XrdClient
      CreateXClient(fUrl.GetUrl(), fOption, fNetopt, kFALSE);

   } else {

      // url is not needed because already stored
      // fOption is set in TFile::ReOpen
      Open(fOption.Data(), kFALSE);
   }

   // If not successful, flag it
   if (!IsOpen())
      return -1;

   // This means ok for net files
   return -2;  // set as fD in ReOpen
}

//_____________________________________________________________________________
void TXNetFile::SetEnv()
{
   // Set the relevant environment variables

   // List of domains where redirection is allowed
   TString allowRE = gEnv->GetValue("XNet.RedirDomainAllowRE", "");
   if (allowRE.Length() > 0)
      EnvPutString(NAME_REDIRDOMAINALLOW_RE, allowRE.Data());

   // List of domains where redirection is denied
   TString denyRE  = gEnv->GetValue("XNet.RedirDomainDenyRE", "");
   if (denyRE.Length() > 0)
      EnvPutString(NAME_REDIRDOMAINDENY_RE, denyRE.Data());

   // List of domains where connection is allowed
   TString allowCO = gEnv->GetValue("XNet.ConnectDomainAllowRE", "");
   if (allowCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINALLOW_RE, allowCO.Data());

   // List of domains where connection is denied
   TString denyCO  = gEnv->GetValue("XNet.ConnectDomainDenyRE", "");
   if (denyCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINDENY_RE, denyCO.Data());

   // Connect Timeout
   Int_t connTO = gEnv->GetValue("XNet.ConnectTimeout",
                                  DFLT_CONNECTTIMEOUT);
   EnvPutInt(NAME_CONNECTTIMEOUT, connTO);

   // Reconnect Timeout
   Int_t recoTO = gEnv->GetValue("XNet.ReconnectWait",
                                  DFLT_RECONNECTWAIT);
   if (recoTO == DFLT_RECONNECTWAIT) {
      // Check also the old variable name
      recoTO = gEnv->GetValue("XNet.ReconnectTimeout",
                                  DFLT_RECONNECTWAIT);
   }
   EnvPutInt(NAME_RECONNECTWAIT, recoTO);

   // Request Timeout
   Int_t requTO = gEnv->GetValue("XNet.RequestTimeout",
                                  DFLT_REQUESTTIMEOUT);
   EnvPutInt(NAME_REQUESTTIMEOUT, requTO);

   // Max number of redirections
   Int_t maxRedir = gEnv->GetValue("XNet.MaxRedirectCount",
                                    DFLT_MAXREDIRECTCOUNT);
   EnvPutInt(NAME_MAXREDIRECTCOUNT, maxRedir);


   // Read ahead size
   Int_t rAheadsiz = gEnv->GetValue("XNet.ReadAheadSize",
                                     DFLT_READAHEADSIZE);
   EnvPutInt(NAME_READAHEADSIZE, rAheadsiz);


   // Cache size (<= 0 disables cache)
   Int_t rCachesiz = gEnv->GetValue("XNet.ReadCacheSize",
                                     DFLT_READCACHESIZE);

   EnvPutInt(NAME_READCACHESIZE, rCachesiz);

   // Max number of retries on first connect
   Int_t maxRetries = gEnv->GetValue("XNet.FirstConnectMaxCnt",
                                     DFLT_FIRSTCONNECTMAXCNT);
   EnvPutInt(NAME_FIRSTCONNECTMAXCNT, maxRetries);

   // Parallel stream count
   Int_t parStreamsCnt = gEnv->GetValue("XNet.ParStreamsPerPhyConn",
                                        DFLT_MULTISTREAMCNT);
   EnvPutInt(NAME_MULTISTREAMCNT, parStreamsCnt);

   // Change the TCP window size (0 means 'scaling' on some platforms)
   Int_t tcpWindowSize = gEnv->GetValue("XNet.DfltTcpWindowSize",
                                        DFLT_DFLTTCPWINDOWSIZE);
   EnvPutInt(NAME_DFLTTCPWINDOWSIZE, tcpWindowSize);

   // Change the transaction timeout
   Int_t transactionTimeout = gEnv->GetValue("XNet.TransactionTimeout",
                                             DFLT_TRANSACTIONTIMEOUT);
   EnvPutInt(NAME_TRANSACTIONTIMEOUT, transactionTimeout);

   // Whether to activate automatic rootd backward-compatibility
   // (We override XrdClient default)
   fgRootdBC = gEnv->GetValue("XNet.RootdFallback", 1);
   EnvPutInt(NAME_KEEPSOCKOPENIFNOTXRD, fgRootdBC);

   // Dynamic forwarding (SOCKS4)
   TString socks4Host = gEnv->GetValue("XNet.SOCKS4Host","");
   Int_t socks4Port = gEnv->GetValue("XNet.SOCKS4Port",-1);
   if (socks4Port > 0) {
      if (socks4Host.IsNull())
         // Default
         socks4Host = "127.0.0.1";
      EnvPutString(NAME_SOCKS4HOST, socks4Host.Data());
      EnvPutInt(NAME_SOCKS4PORT, socks4Port);
   }

   const char *cenv = 0;

   // For password-based authentication
   TString autolog = gEnv->GetValue("XSec.Pwd.AutoLogin","1");
   if (autolog.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecPWDAUTOLOG")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecPWDAUTOLOG",autolog.Data());

   // Old style netrc file
   TString netrc;
   netrc.Form("%s/.rootnetrc",gSystem->HomeDirectory());
   gSystem->Setenv("XrdSecNETRC", netrc.Data());

   TString alogfile = gEnv->GetValue("XSec.Pwd.ALogFile","");
   if (alogfile.Length() > 0)
      gSystem->Setenv("XrdSecPWDALOGFILE",alogfile.Data());

   TString verisrv = gEnv->GetValue("XSec.Pwd.VerifySrv","1");
   if (verisrv.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecPWDVERIFYSRV")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecPWDVERIFYSRV",verisrv.Data());

   TString srvpuk = gEnv->GetValue("XSec.Pwd.ServerPuk","");
   if (srvpuk.Length() > 0)
      gSystem->Setenv("XrdSecPWDSRVPUK",srvpuk.Data());

   // For GSI authentication
   TString cadir = gEnv->GetValue("XSec.GSI.CAdir","");
   if (cadir.Length() > 0)
      gSystem->Setenv("XrdSecGSICADIR",cadir.Data());

   TString crldir = gEnv->GetValue("XSec.GSI.CRLdir","");
   if (crldir.Length() > 0)
      gSystem->Setenv("XrdSecGSICRLDIR",crldir.Data());

   TString crlext = gEnv->GetValue("XSec.GSI.CRLextension","");
   if (crlext.Length() > 0)
      gSystem->Setenv("XrdSecGSICRLEXT",crlext.Data());

   TString ucert = gEnv->GetValue("XSec.GSI.UserCert","");
   if (ucert.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERCERT",ucert.Data());

   TString ukey = gEnv->GetValue("XSec.GSI.UserKey","");
   if (ukey.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERKEY",ukey.Data());

   TString upxy = gEnv->GetValue("XSec.GSI.UserProxy","");
   if (upxy.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERPROXY",upxy.Data());

   TString valid = gEnv->GetValue("XSec.GSI.ProxyValid","");
   if (valid.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYVALID",valid.Data());

   TString deplen = gEnv->GetValue("XSec.GSI.ProxyForward","0");
   if (deplen.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSIPROXYDEPLEN")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSIPROXYDEPLEN",deplen.Data());

   TString pxybits = gEnv->GetValue("XSec.GSI.ProxyKeyBits","");
   if (pxybits.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYKEYBITS",pxybits.Data());

   TString crlcheck = gEnv->GetValue("XSec.GSI.CheckCRL","1");
   if (crlcheck.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSICRLCHECK")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSICRLCHECK",crlcheck.Data());

   TString delegpxy = gEnv->GetValue("XSec.GSI.DelegProxy","0");
   if (delegpxy.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSIDELEGPROXY")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSIDELEGPROXY",delegpxy.Data());

   TString signpxy = gEnv->GetValue("XSec.GSI.SignProxy","1");
   if (signpxy.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSISIGNPROXY")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSISIGNPROXY",signpxy.Data());

   // Using ROOT mechanism to IGNORE SIGPIPE signal
   gSystem->IgnoreSignal(kSigPipe);
}

//_____________________________________________________________________________
void TXNetFile::SynchronizeCacheSize()
{
   // Synchronize the cache size
   // Alternative purging policy

   if (fClient == 0) return;

   fClient->UseCache(TRUE);
   Int_t size;
   Long64_t bytessubmitted, byteshit, misscount, readreqcnt;
   Float_t  missrate, bytesusefulness;
   int newbsz = -1;
   if (fClient->GetCacheInfo(size, bytessubmitted,
                             byteshit, misscount,
                             missrate, readreqcnt,
                             bytesusefulness) ) {

      // To allow for some space for outstanding data
      TFileCacheRead *cacheRead = GetCacheRead();
      if (cacheRead) {
         newbsz = GetBufferSize() / 2 * 3;
         newbsz = TMath::Max(newbsz, size);
      } else {
         newbsz = size;
      }

   }

   if (newbsz > 0)
      fClient->SetCacheParameters(newbsz, 0, XrdClientReadCache::kRmBlk_FIFO);
}

//_____________________________________________________________________________
void TXNetFile::ResetCache()
{
   // Reset the cache

   if (fClient)
      fClient->RemoveAllDataFromCache();
}

//______________________________________________________________________________
Int_t TXNetFile::GetBytesToPrefetch() const
{
   // Max number of bytes to prefetch.

   Int_t size;
   Long64_t bytessubmitted, byteshit, misscount, readreqcnt;
   Float_t  missrate, bytesusefulness;
   Int_t bytes = 0;
   if (fClient && fClient->GetCacheInfo(size, bytessubmitted,
                                        byteshit, misscount,
                                        missrate, readreqcnt,
                                        bytesusefulness) )
   bytes = size;
   return ((bytes < 0) ? 0 : bytes);
}

//______________________________________________________________________________
void TXNetFile::Print(Option_t *option) const
{
   // Print the local statistics.

   Printf("TXNetFile caching information:");

   Int_t size;
   Long64_t bytessubmitted, byteshit, misscount, readreqcnt;
   Float_t  missrate, bytesusefulness;

   if (fClient && fClient->GetCacheInfo(size, bytessubmitted,
                                        byteshit, misscount,
                                        missrate, readreqcnt,
                                        bytesusefulness)) {
      Printf(" Max size:                  %d",   size);
      Printf(" Bytes submitted:           %lld", bytessubmitted);
      Printf(" Bytes hit (estimation):    %lld", byteshit);
      Printf(" Miss count:                %lld", misscount);
      Printf(" Miss rate:                 %f",   missrate);
      Printf(" Read requests count:       %lld", readreqcnt);
      Printf(" Bytes usefulness:          %f\n", bytesusefulness);
   } else
      Printf(" -- No Xrd client instance allocated --\n");

   TFile::Print(option);
}
