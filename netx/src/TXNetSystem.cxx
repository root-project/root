// @(#)root/netx:$Name:  $:$Id: TXNetSystem.cxx,v 1.9 2006/04/27 15:14:11 rdm Exp $
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
#include "TROOT.h"

#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientConn.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XProtocol/XProtocol.hh"


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

   // Set debug level
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XNet.Debug", -1));

   // The first time do some global initialization
   if (!fgInitDone)
      InitXrdClient();

   // Fill in user, host, port
   TNetSystem::InitRemoteEntity(url);

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

   // Init vars with default debug level -1, so we do not get warnings
   TXNetFile::SetEnv();

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
      XrdOucString s(fDir.Data());
      dirs.Push_back(s);
      // Check if the directory exists
      fClientAdmin->ExistDirs(dirs,existDirs);
      if (existDirs.GetSize()>0 && existDirs[0])
         return fDirp;
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
      // use default permissions 755 to create directory
      Bool_t ok = fClientAdmin->Mkdir(TUrl(dir).GetFile(),7,5,5);
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
      if (fDirList.GetSize() > 0)
         return fDirList.Pop_back().c_str();
      return 0;   // until all of them have been returned
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

      // Count offline files as inexistent 
      ok &= !(flags & kXR_offline);

      if (ok) {
         buf.fDev = (id >> 24);
         buf.fIno = (id && 0x00FFFFFF);
         buf.fUid = -1;       // not all information available in xrootd
         buf.fGid = -1;       // not available
         buf.fSize = size;
         buf.fMtime = modtime;

         if (flags == 0) buf.fMode = kS_IFREG;
         if (flags & kXR_xset) buf.fMode = (kS_IFREG|kS_IXUSR|kS_IXGRP|kS_IXOTH);
         if (flags & kXR_isDir) buf.fMode = kS_IFDIR;
         if (flags & kXR_other) buf.fMode = kS_IFSOCK;
         if (flags & kXR_readable) buf.fMode |= kS_IRUSR;
         if (flags & kXR_writable) buf.fMode |= kS_IWUSR;

         buf.fIsLink = 0;     // not available
         return 0;
      }
      return 1;
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

   if (gDebug > 1)
      Info("ConsistenWith","Calling TNetSystem::ConsistenWith");

   // Make sure we are checking the end-point user, host, port
   if (fClientAdmin && fClientAdmin->GetClientConn()) {
      XrdClientUrlInfo eurl = fClientAdmin->GetClientConn()->GetCurrentUrl();
      TNetSystem::InitRemoteEntity(eurl.GetUrl().c_str());
   }

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
