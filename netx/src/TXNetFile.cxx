// @(#)root/netx:$Name:  $:$Id: TXNetFile.cxx,v 1.9 2005/05/12 12:25:26 rdm Exp $
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
// Interfaced to the posix client: G. Ganis, CERN                       //
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

#include "TError.h"
#include "TEnv.h"
#include "TSocket.h"
#include "TXNetFile.h"

#include <XrdClient/XrdClient.hh>
#include <XrdClient/XrdClientConst.hh>
#include <XrdClient/XrdClientEnv.hh>
#include <XProtocol/XProtocol.hh>

ClassImp(TXNetFile);

Bool_t TXNetFile::fgInitDone = kFALSE;
Bool_t TXNetFile::fgRootdBC = kTRUE;
  
//_____________________________________________________________________________
TXNetFile::TXNetFile(const char *url, Option_t *option, const char* ftitle,
                       Int_t compress, Int_t netopt) :
                       TNetFile(url, ftitle, compress, kFALSE)
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
   // method will be called)

   if (!fgInitDone) {

      // Set debug level
      EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", 0));

      // Local domain
      TInetAddress loc(gSystem->GetHostByName(gSystem->HostName()));
      TString localDomain = loc.GetHostName();
      localDomain.Remove(0,localDomain.Index('.'));

      // List of domains where redirection is allowed
      TString allowRE = gEnv->GetValue("XNet.RedirDomainAllowRE",
                                        localDomain.Data());
      EnvPutString(NAME_REDIRDOMAINALLOW_RE, allowRE.Data());

      // List of domains where redirection is denied
      TString denyRE  = gEnv->GetValue("XNet.RedirDomainDenyRE",
                                       "<unknown>");
      EnvPutString(NAME_REDIRDOMAINDENY_RE, denyRE.Data());

      // List of domains where connection is allowed
      TString allowCO = gEnv->GetValue("XNet.ConnectDomainAllowRE",
                                       localDomain.Data());
      EnvPutString(NAME_CONNECTDOMAINALLOW_RE, allowCO.Data());

      // List of domains where connection is denied
      TString denyCO  = gEnv->GetValue("XNet.ConnectDomainDenyRE",
                                       "<unknown>");
      EnvPutString(NAME_CONNECTDOMAINDENY_RE, denyCO.Data());

      // Connect Timeout
      Int_t connTO = gEnv->GetValue("XNet.ConnectTimeout",
                                     DFLT_CONNECTTIMEOUT);
      EnvPutInt(NAME_CONNECTTIMEOUT, connTO);

      // Reconnect Timeout
      Int_t recoTO = gEnv->GetValue("XNet.ReconnectTimeout",
                                     DFLT_RECONNECTTIMEOUT);
      EnvPutInt(NAME_RECONNECTTIMEOUT, recoTO);

      // Request Timeout
      Int_t requTO = gEnv->GetValue("XNet.RequestTimeout",
                                     DFLT_REQUESTTIMEOUT);
      EnvPutInt(NAME_REQUESTTIMEOUT, requTO);

      // Max number of redirections
      Int_t maxRedir = gEnv->GetValue("XNet.MaxRedirectCount",
                                       DFLT_MAXREDIRECTCOUNT);
      EnvPutInt(NAME_MAXREDIRECTCOUNT, maxRedir);

      // Whether to use a separate thread for garbage collection
      Int_t garbCollTh = gEnv->GetValue("XNet.StartGarbageCollectorThread",
                                         DFLT_STARTGARBAGECOLLECTORTHREAD);
      EnvPutInt(NAME_STARTGARBAGECOLLECTORTHREAD, garbCollTh);

      // Whether to use a separate thread for reading
      Int_t goAsync = gEnv->GetValue("XNet.GoAsynchronous",
                                      DFLT_GOASYNC);
      EnvPutInt(NAME_GOASYNC, goAsync);

      // Read ahead size
      Int_t rAheadsiz = gEnv->GetValue("XNet.ReadAheadSize",
                                        DFLT_READAHEADSIZE);
      EnvPutInt(NAME_READAHEADSIZE, rAheadsiz);

      // Cache size (<= 0 disables cache)
      Int_t rCachesiz = gEnv->GetValue("XNet.ReadCacheSize",
                                        DFLT_READCACHESIZE);
      EnvPutInt(NAME_READCACHESIZE, rCachesiz);

      // Max number of retries on first connect
      Int_t maxRetries = gEnv->GetValue("XNet.TryConnect",
                                        DFLT_FIRSTCONNECTMAXCNT);
      EnvPutInt(NAME_FIRSTCONNECTMAXCNT, maxRetries);

      // Whether to activate automatic rootd backward-compatibility
      Bool_t fgRootdBC = gEnv->GetValue("XNet.RootdFallback",
                                         NAME_KEEPSOCKOPENIFNOTXRD);
      EnvPutInt(NAME_KEEPSOCKOPENIFNOTXRD, fgRootdBC);

      // For password-based authentication
      TString autolog = gEnv->GetValue("XSec.Pwd.AutoLogin","1");
      if (autolog.Length() > 0)
         gSystem->Setenv("XrdSecPWDAUTOLOG",autolog.Data());

      TString alogfile = gEnv->GetValue("XSec.Pwd.ALogFile","");
      if (alogfile.Length() > 0)
         gSystem->Setenv("XrdSecPWDALOGFILE",alogfile.Data());

      TString verisrv = gEnv->GetValue("XSec.Pwd.VerifySrv","1");
      if (verisrv.Length() > 0)
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
      if (deplen.Length() > 0)
         gSystem->Setenv("XrdSecGSIPROXYDEPLEN",deplen.Data());

      TString pxybits = gEnv->GetValue("XSec.GSI.ProxyKeyBits","");
      if (pxybits.Length() > 0)
         gSystem->Setenv("XrdSecGSIPROXYKEYBITS",pxybits.Data());

      TString crlcheck = gEnv->GetValue("XSec.GSI.CheckCRL","2");
      if (crlcheck.Length() > 0)
         gSystem->Setenv("XrdSecGSICRLCHECK",crlcheck.Data());

      // Print the tag, if required (only once)
      if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
         Info("TXNetFile","(C) 2005 SLAC TXNetFile (eXtended TNetFile) %s",
               gROOT->GetVersion());

      // Using ROOT mechanism to IGNORE SIGPIPE signal
      gSystem->IgnoreSignal(kSigPipe);

      // Only once
      fgInitDone = kTRUE;
   }

   // Create an instance
   CreateXClient(url, option, netopt);
}

//_____________________________________________________________________________
TXNetFile::~TXNetFile()
{
   // Destructor
   if (IsOpen())
      Close(0);

   if (fClient)
      delete fClient;
   fClient = 0;
}

//_____________________________________________________________________________
void TXNetFile::FormUrl(char *uut, TString &uus)
{
   // Form url for rootd socket

   // We need something to work on 
   if (!uut) return;

   // raw url
   TUrl uu(uut);

   // Clean up
   free(uut);

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
void TXNetFile::CreateXClient(const char *url, Option_t *option, Int_t netopt)
{
   // The real creation work is done here

   // Init members
   fSize = 0;
   fIsRootd = kFALSE;

   //
   // Setup a client instance
   fClient = new XrdClient(url);
   if (!fClient) {
      Error("CreateXClient","fatal error: new object creation failed -"
            " out of system resources.");
      gSystem->Abort();
      goto zombie;
   }

   //
   // Now try opening the file
   // Cycling through the different urls and handling of
   // redirections is done internally
   Open(option);

   //
   // Open file
   if (!fClient->IsOpen()) {
      if (gDebug > 1)
         Info("CreateXClient", "remote file could not be open");

      if (fgRootdBC) {
         Bool_t isRootd =
            (fClient->GetClientConn()->GetServerType() == XrdClientConn::kSTRootd);
         Int_t sd = fClient->GetClientConn()->GetOpenSockFD();
         if (isRootd && sd > -1) {
            //
            // Create a TSocket on the open connection
            TSocket *s = new TSocket(sd);

            s->SetOption(kNoBlock, 0);

            // Find out the remote protocol (send the client protocol first)
            UInt_t cproto = 0;
            Int_t len = sizeof(cproto);
            memcpy((char *)&cproto,
               Form(" %d", TSocket::GetClientProtocol()),len);
            Int_t ns = s->SendRaw(&cproto, len);
            if (ns != len) {
               Error("CreateXClient", "sending %d bytes to rootd server [%s:%d]",
                     len, fUrl.GetHost(), fUrl.GetPort());
               goto zombie;
            }
            
            // Get the remote protocol
            Int_t ibuf[2] = {0};
            len = sizeof(ibuf);
            Int_t nr = s->RecvRaw(ibuf, len);
            if (nr != len) {
               Error("CreateXClient", "reading %d bytes from rootd server [%s:%d]",
                     len, fUrl.GetHost(), fUrl.GetPort());
               goto zombie;
            }
            Int_t kind = net2host(ibuf[0]);
            Int_t rproto = 0;
            if (kind == kROOTD_PROTOCOL) {
               rproto = net2host(ibuf[1]);
            } else {
               kind = net2host(ibuf[1]);
               if (kind == kROOTD_PROTOCOL) {
                  len = sizeof(rproto);
                  nr = s->RecvRaw(&rproto, len);
                  if (nr != len) {
                     Error("CreateXClient",
                           "reading %d bytes from rootd server [%s:%d]",
                           len, fUrl.GetHost(), fUrl.GetPort());
                     goto zombie;
                  }
                  rproto = net2host(rproto);
               }
            }
            if (gDebug > 2)
               Info("CreateXClient",
                    "remote rootd: buf1: %d, buf2: %d rproto: %d",
                    net2host(ibuf[0]),net2host(ibuf[1]),rproto);
            // Finalize TSocket initialization
            s->SetRemoteProtocol(rproto);
            char *uut = StrDup((fClient->GetClientConn()
                                       ->GetCurrentUrl()).GetUrl().c_str());
            TString uu;
            FormUrl(uut,uu);
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
         } else {
            Error("CreateXClient", "some severe error occurred while opening"
                  " the remote file at %s - exit",url);
            goto zombie;
         }
      } else {
         Error("CreateXClient",
               "while opening the remote file at %s - exit",url);
         goto zombie;
      }
   }

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;

}

//_____________________________________________________________________________
void TXNetFile::Open(Option_t *option)
{
   // The real creation work is done here

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
   // Create and Recreate are correlated
   if (recreate) {
      openOpt |= kXR_delete;
   } else if (create) {
      openOpt |= kXR_new;
   }
   if (update)
      openOpt |= kXR_open_updt;
   if (read)
      openOpt |= kXR_open_read;

   //
   // Set open mode to rw-r-r
   kXR_unt16 openMode = kXR_or | kXR_gr | kXR_ur | kXR_uw;

   //
   // Open file
   if (!fClient->Open(openMode, openOpt)) {
      if (gDebug > 1)
         Info("Open", "remote file could not be open");
   } else {
      // Initialize the file
      Init(create);
   }

   return;
}

//_____________________________________________________________________________
Bool_t TXNetFile::ReadBuffer(char *buffer, Int_t BufferLength)
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
      return TNetFile::ReadBuffer(buffer, BufferLength);
   }

   if (!IsOpen()) {
      Error("ReadBuffer","The remote file is not open");
      return kTRUE;
   }

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->ReadBuffer(fOffset, buffer, BufferLength)) < 0) {
         Error("ReadBuffer", "error reading from cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::ReadBuffer(), reset it
         Seek(off + BufferLength);
         return result;
      }
   }

   //
   // Read for the remote xrootd
   Int_t nr = fClient->Read(buffer, fOffset, BufferLength);

   if ((result = (nr > 0))) {

      if (gDebug > 1)
         Info("ReadBuffer", " %d bytes of data read from offset"
                            " %Ld (%d requested)", nr, fOffset, BufferLength);

      fOffset += BufferLength;
      fBytesRead += BufferLength;
#ifdef WIN32
      SetFileBytesRead(GetFileBytesRead() + BufferLength);
#else
      fgBytesRead += BufferLength;
#endif
   }

   return result;
}

//_____________________________________________________________________________
Bool_t TXNetFile::WriteBuffer(const char *buffer, Int_t BufferLength)
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
      return TNetFile::WriteBuffer(buffer, BufferLength );
   }

   if (!IsOpen()) {
      Error("WriteBuffer","The remote file is not open");
      return kTRUE;
   }

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->WriteBuffer(fOffset, buffer, BufferLength)) < 0) {
         SetBit(kWriteError);
         Error("WriteBuffer", "error writing to cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::WriteBuffer(), reset it
         Seek(off + BufferLength);
         return result;
      }
   }

   //
   // Read for the remote xrootd
   Int_t nw = fClient->Write(buffer, fOffset, BufferLength);

   if ((result = (nw > 0))) {

      if (gDebug > 1)
         Info("WriteBuffer", " %d bytes of data wrote to offset"
                            " %Ld (%d requested)", nw, fOffset, BufferLength);

      fOffset += BufferLength;
      fBytesWrite += BufferLength;
#ifdef WIN32
      SetFileBytesWritten(GetFileBytesWritten() + BufferLength);
#else
      fgBytesWrite += BufferLength;
#endif
   }
   return result;
}

//_____________________________________________________________________________
Bool_t TXNetFile::IsOpen() const
{
   // Return kTRUE if the file is open, kFALSE otherwise

   if (fIsRootd) {
      if (gDebug > 1)
         Info("IsOpen","Calling TNetFile::IsOpen");
      return TNetFile::IsOpen();
   }

   return (fClient ? fClient->IsOpen() : 0);
}

//_____________________________________________________________________________
Int_t TXNetFile::ReOpen(const Option_t *Mode)
{
   // Re-open the file (see TNetFile::ReOpen() or TFile::ReOpen()
   // for more details)

   if (fIsRootd) {
      if (gDebug > 1)
         Info("ReOpen","Calling TNetFile::ReOpen");
      return TNetFile::ReOpen(Mode);
   }

   fSize = 0;

   return TFile::ReOpen(Mode);
}

//_____________________________________________________________________________
void TXNetFile::Close(const Option_t *opt)
{
   // Close the file (see TNetFile::Close() or TFile::Close()
   // for more details)

   if (fIsRootd) {
      if (gDebug > 1)
         Info("Close","Calling TNetFile::Close");
      TNetFile::Close(opt);
      return;
   }

   TFile::Close(opt);

   fSize = 0;
   fIsRootd = kFALSE;
}

//_____________________________________________________________________________
void TXNetFile::Flush()
{
   // Flushes un-written data

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
   // Override TNetFile::SysStat (see parent's method for more details)

   if (IsZombie()) {
      Error("SysStat", "SysStat is not possible because object is"
            " in 'zombie' state");
      *size = 0;
      return 1;
   }

   if (fIsRootd) {
      if (gDebug > 1)
         Info("SysStat","Calling TNetFile::SysStat");
      return TNetFile::SysStat(fd, id, size, flags, modtime);
   }

   if (!IsOpen()) {
      Error("SysStat","The remote file is not open");
      *size = 0;
      return 1;
   }

   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo().


   //
   // Flush via the remote xrootd
   fClient->Sync();
   struct XrdClientStatInfo stinfo;
   if (fClient->Stat(&stinfo)) {
      *id = (Long_t)(stinfo.id);
      *size = (Long64_t)(stinfo.size);
      *flags = (Long_t)(stinfo.flags);
      *modtime = (Long_t)(stinfo.modtime);
      if (gDebug > 1)
         Info("SysStat", "got stats = %ld %lld %ld %ld",
                         *id, *size, *flags, *modtime);
   } else {
      if (gDebug > 1)
         Info("SysStat", "could not stat remote file");
      *id = -1;
      return 1;
   }

   // We are done
   return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysClose(Int_t fd)
{
   // Override TNetFile::SysClose (see parent's method for more details)

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
   // Override TNetFile::SysOpen (see parent's method for more details)

   if (fIsRootd) {
      if (gDebug > 1)
         Info("SysOpen", "Calling TNetFile::SysOpen");
      return TNetFile::SysOpen(pathname, flags, mode);
   }

   // url is not needed because already stored
   // fOption is set in TFile::ReOpen
   Open(fOption.Data());

   // If not successful, flag it
   if (!IsOpen())
      return -1;

   // This means ok for net files
   return -2;
}

//_____________________________________________________________________________
Long64_t TXNetFile::Size(void)
{
   // Return file size

   Long64_t size;
   Long_t i, f, m;

   SysStat((Int_t)0, &i, &size, &f, &m);

   memcpy((void *)&fSize, (const void*)&size, sizeof(size));
   return fSize;
}
