// @(#)root/netx:$Name:  $:$Id: TXNetSystem.cxx,v 1.2 2005/12/12 12:54:27 rdm Exp $
// Author: Frank Winklmeier, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetSystem                                                          //
//                                                                      //
// Authors: Frank Winklmeier,  Fabrizio Furano                          //
//          INFN Padova, 2005                                           //
//                                                                      //
// TXNetSystem is an extension of TNetSystem able to deal with new      //
// xrootd servers. The class detects the nature of the server and       //
// redirects the calls to TNetSystem in case of a rootd server.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include "TEnv.h"
#include "TSocket.h"
#include "TUrl.h"
#include "TVirtualMutex.h"
#include "TXNetFile.h"
#include "TXNetSystem.h"

#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientEnv.hh"


ClassImp(TXNetSystem);

Bool_t TXNetSystem::fgInitDone = kFALSE;
Bool_t TXNetSystem::fgRootdBC = kTRUE;

//_____________________________________________________________________________
TXNetSystem::TXNetSystem(Bool_t owner) : TNetSystem(owner)
{
   // Create system management class without connecting to server.

   SetTitle("(x)rootd system administration");
   fClientAdmin = 0;
   fIsXRootd = kFALSE;
   fDir = "";
   fDirp = 0;
   fDirListValid = kFALSE;
}

//_____________________________________________________________________________
TXNetSystem::TXNetSystem(const char *url, Bool_t owner) : TNetSystem(owner)
{
   // Create system management class and connect to server specified by url.

   SetTitle("(x)rootd system administration");
   fIsRootd = kFALSE;
   fIsXRootd = kFALSE;
   fDir = "";
   fDirp = 0;
   fDirListValid = kFALSE;

   // The first timr do some global initialization
   if (!fgInitDone)
      InitXrdClient();

   // We need a dummy filename after the server url to connect
   TString dummy = url;
   dummy += "/dummy";

   fClientAdmin = new XrdClientAdmin(dummy);

   if (!fClientAdmin) {
      Error("TXNetSystem","fatal error: new object creation failed.");
      gSystem->Abort();
   }

   // Try to connect to the server
   if (fClientAdmin->Connect()) {
      fIsXRootd = kTRUE;
   } else {
      if (fgRootdBC) {
         Bool_t isRootd =
            (fClientAdmin->GetClientConn()->GetServerType() == XrdClientConn::kSTRootd);
         Int_t sd = fClientAdmin->GetClientConn()->GetOpenSockFD();
         if (isRootd && sd > -1) {
            //
            // Create a TSocket on the open connection
            TSocket *s = new TSocket(sd);

            // We will clean it by ourselves
            R__LOCKGUARD2(gROOTMutex);
            gROOT->GetListOfSockets()->Remove(s);

            s->SetOption(kNoBlock, 0);

            // Find out the remote protocol (send the client protocol first)
            Int_t rproto = TXNetFile::GetRootdProtocol(s);
            if (rproto < 0) {
               Error("TXNetSystem", "getting protocol of the rootd server");
               return;
            }
            // Finalize TSocket initialization
            s->SetRemoteProtocol(rproto);
            TUrl uut((fClientAdmin->GetClientConn()
                             ->GetCurrentUrl()).GetUrl().c_str());
            TString uu;
            TXNetFile::FormUrl(uut,uu);
            if (gDebug > 2)
               Info("TXNetSystem"," url: %s",uu.Data());

            s->SetUrl(uu.Data());
            s->SetService("rootd");
            s->SetServType(TSocket::kROOTD);
            //
            // Now we can check if we can create a TNetFile on the
            // open connection
            if (rproto > 13) {
               //
               // Remote support for reuse of open connection
               TNetSystem::Create(uu, s);
            } else {
               //
               // Open connection has been closed because could
               // not be reused; TNetSystem will open a new connection
               TNetSystem::Create(uu);
            }

            // Type of server
            fIsRootd = kTRUE;

         } else {
            Error("TXNetSystem", "some severe error occurred while opening"
                  " the connection at %s - exit", url);
            return;
         }
      } else {
         Error("TXNetSystem",
               "while opening the connection at %s - exit", url);
         return;
      }
   }

   return;
}

//_____________________________________________________________________________
TXNetSystem::~TXNetSystem()
{
   // Destructor

   if (fIsXRootd && fClientAdmin)
      delete fClientAdmin;
}


//_____________________________________________________________________________
void TXNetSystem::InitXrdClient()
{
   // One-time initialization of some communication variables for xrootd protocol

   // Set debug level
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", 0));

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
   Int_t goAsync = gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC);
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
   // (We override XrdClient default)
   fgRootdBC = gEnv->GetValue("XNet.RootdFallback", 1);
   EnvPutInt(NAME_KEEPSOCKOPENIFNOTXRD, fgRootdBC);

   // For password-based authentication
   TString autolog = gEnv->GetValue("XSec.Pwd.AutoLogin","1");
   if (autolog.Length() > 0)
      gSystem->Setenv("XrdSecPWDAUTOLOG",autolog.Data());

   // Old style netrc file
   TString netrc;
   netrc.Form("%s/.rootnetrc",gSystem->HomeDirectory());
   gSystem->Setenv("XrdSecNETRC", netrc.Data());

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

   // Using ROOT mechanism to IGNORE SIGPIPE signal
   gSystem->IgnoreSignal(kSigPipe);

   // Only once
   fgInitDone = kTRUE;

   // Print the tag, if required (only once)
   if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
     Info("TXNetFile","(C) 2005 SLAC TXNetSystem (eXtended TNetSystem) %s",
	  gROOT->GetVersion());
}

//_____________________________________________________________________________
void* TXNetSystem::OpenDirectory(const char* dir)
{
   // Open a directory. Returns a non-zero pointer (with no special
   // purpose) in case of success, 0 in case of error.

   if (fIsXRootd) {
      // Extract the directory name
      fDir = TUrl(dir).GetFile();
      fDirp = (void*)&fDir;     // serves as directory pointer

      vecString dirs;
      vecBool existDirs;
      XrdClientString s(fDir.Data());
      dirs.Push_back(s);
      // Check if the directory exists
      fClientAdmin->ExistDirs(dirs,existDirs);
      if (existDirs.GetSize()>0 && existDirs[0])
         return fDirp;
      else
         return 0;
   }

   if (gDebug > 1)
      Info("OpenDirectory", "calling TNetSystem::OpenDirectory");
   return TNetSystem::OpenDirectory(dir);       // for a rootd
}

//_____________________________________________________________________________
void TXNetSystem::FreeDirectory(void *dirp)
{
   // Free(Close) the directory referenced by dirp

   if (fIsXRootd) {
      if (dirp != fDirp) {
	 Error("FreeDirectory","invalid directory pointer (%p, %p)", dirp, fDirp);
	 return;
      }
      fDir = "";
      fDirp = 0;
      fDirListValid = kFALSE;
      return;
   }

   if (gDebug > 1)
      Info("FreeDirectory","calling TNetSystem::FreeDirectory");
   return TNetSystem::FreeDirectory(dirp);     // for a rootd
}

//_____________________________________________________________________________
Int_t TXNetSystem::MakeDirectory(const char* dir)
{
   // Create a directory. Return 0 on success, -1 otherwise.

   if (fIsXRootd) {
      // Extract the directory name
      TString edir = TUrl(dir).GetFile();

      // use default permissions 755 to create directory
      Bool_t ok = fClientAdmin->Mkdir(dir,7,5,5);
      return (ok ? 0 : -1);
   }

   if (gDebug > 1) Info("MakeDirectory","Calling TNetSystem::MakeDirectory");
   return TNetSystem::MakeDirectory(dir);     // for a rootd
}

//_____________________________________________________________________________
const char* TXNetSystem::GetDirEntry(void *dirp)
{
   // Get directory entry for directory referenced by dirp.
   // Returns 0 in case there are no more entries.

   if (fIsXRootd) {
      if (dirp != fDirp) {
	 Error("GetDirEntry","invalid directory pointer");
	 return 0;
      }

      // Only request new directory listing the first time called
      if (!fDirListValid) {
	 Bool_t ok = fClientAdmin->DirList(fDir,fDirList);
	 if (ok)
            fDirListValid = kTRUE;
	 else
            return 0;
      }

      // Return entries one by one with each call of method
      if (fDirList.GetSize()>0) return fDirList.Pop_back().c_str();
      else return 0;   // until all of them have been returned
   }

   if (gDebug > 1) Info("GetDirEntry","Calling TNetSystem::GetDirEntry");
   return TNetSystem::GetDirEntry(dirp);      // for a rootd
}

//_____________________________________________________________________________
Int_t TXNetSystem::GetPathInfo(const char* path, FileStat_t &buf)
{
   // Get info about a file. Info is returned in the form of a FileStat_t
   // structure (see TSystem.h).
   // The function returns 0 in case of success and 1 if the file could
   // not be stat'ed.
   // NOTICE: Not all information is available with an xrootd server.

   if (fIsXRootd) {

      Long_t id;
      Long64_t size;
      Long_t flags;
      Long_t modtime;

      // Extract the directory name
      TString edir = TUrl(path).GetFile();
      Bool_t ok = fClientAdmin->Stat(edir,id,size,flags,modtime);

      if (ok) {
	 buf.fDev = (id >> 24);
	 buf.fIno = (id && 0x00FFFFFF);
	 buf.fUid = -1;       // not all information available in xrootd
	 buf.fGid = -1;       // not available
	 buf.fSize = size;
	 buf.fMtime = modtime;

	 if (flags == 0) buf.fMode = kS_IFREG;
	 if (flags & 1) buf.fMode = (kS_IFREG|kS_IXUSR|kS_IXGRP|kS_IXOTH);
	 if (flags & 2) buf.fMode = kS_IFDIR;
	 if (flags & 4) buf.fMode = kS_IFSOCK;

	 buf.fIsLink = 0;     // not available
	 return 0;
      }
      else return 1;
   }

   if (gDebug > 1)
      Info("GetPathInfo","Calling TNetSystem::GetPathInfo");
   return TNetSystem::GetPathInfo(path,buf);       // for a rootd
}

//_____________________________________________________________________________
Bool_t TXNetSystem::ConsistentWith(const char *path, void *dirptr)
{
   // Check consistency of this helper with the one required
   // by 'path' or 'dirptr'.

   if (fIsXRootd) {
      return TSystem::ConsistentWith(path,dirptr);
   }

   if (gDebug > 1) Info("ConsistenWith","Calling TNetSystem::ConsistenWith");
   return TNetSystem::ConsistentWith(path,dirptr);    // for a rootd
}

//_____________________________________________________________________________
Bool_t TXNetSystem::AccessPathName(const char *path, EAccessMode mode)
{
   // Returns FALSE if one can access a file using the specified access mode.
   // NB: for the time being mode is ignored for XROOTD (just checks existence
   // of the file or directory).
   // Mode is the same as for the Unix access(2) function.
   // Attention, bizarre convention of return value!!

   if (fIsXRootd) {
      // Check only if the file or directory exists and
      FileStat_t buf;
      if (GetPathInfo(path, buf) == 0)
	 if (buf.fMode != kS_IFSOCK)
            return kFALSE;
      // The file could not be stated
      return kTRUE;
   }

   if (gDebug > 1)
      Info("AccessPathName", "calling TNetSystem::AccessPathName");
   return TNetSystem::AccessPathName(path,mode);    // for a rootd
}
