// @(#)root/netx:$Id$
// Author: Frank Winklmeier, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXNetSystem
#define ROOT_TXNetSystem

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

#include "TNetFile.h"

#include "Rtypes.h"

#include "TString.h"

#include "THashList.h"

class XrdClientAdmin;
class TCollection;
class TXNetSystemConnectGuard;

typedef /*XrdClientVector<XrdOucString>*/ void* VecStringVP_t;

class TXrdClientAdminWrapper : public TNamed {
public:
   TXrdClientAdminWrapper(const char *key, XrdClientAdmin *ca) : TNamed(key,""), fXCA(ca) { }
   virtual ~TXrdClientAdminWrapper();
   XrdClientAdmin *fXCA;
};

class TXNetSystem : public TNetSystem {

friend class TXNetSystemConnectGuard;

private:
   Bool_t          fIsRootd;      // Nature of remote file server
   Bool_t          fIsXRootd;     // Nature of remote file server
   TString         fDir;          // Current directory
   void           *fDirp;         // Directory pointer
   VecStringVP_t   fDirList;      // Buffer for directory content
   TString         fDirEntry;     // Last dir entry
   TString         fUrl;          // Initial url

   static Bool_t   fgInitDone;    // Avoid initializing more than once
   static Bool_t   fgRootdBC;     // Control rootd backward compatibility

   XrdClientAdmin *Connect(const char *url); // Connect to server
   void           *GetDirPtr() const { return fDirp; }
   void            InitXrdClient();

   static THashList fgAddrFQDN;   // Cross-table address<->FQDN
   static THashList fgAdminHash;  // List of existing XrdClientAdmin
   static XrdClientAdmin *GetClientAdmin(const char *url);
   static TString GetKey(const char *url);

public:
   TXNetSystem(Bool_t owner = kTRUE);
   TXNetSystem(const char *url, Bool_t owner = kTRUE);
   virtual ~TXNetSystem() { }

   Bool_t              AccessPathName(const char *path, EAccessMode mode);
   virtual Bool_t      ConsistentWith(const char *path, void *dirptr);
   virtual void        FreeDirectory(void *dirp);
   virtual const char *GetDirEntry(void *dirp);
   virtual Int_t       GetPathInfo(const char* path, FileStat_t &buf);
   virtual Bool_t      IsPathLocal(const char *path);
   virtual Int_t       Locate(const char* path, TString &endurl);
   virtual Int_t       MakeDirectory(const char* dir);
   virtual void       *OpenDirectory(const char* dir);
   virtual int         Unlink(const char *path);

   // TXNetSystem specific
   Bool_t              GetPathsInfo(const char *paths, UChar_t *info);
   Bool_t              IsOnline(const char *path);
   Bool_t              Prepare(const char *path, UChar_t opt = 8, UChar_t prio = 0);
   Int_t               Prepare(TCollection *paths,
                               UChar_t opt = 8, UChar_t prio = 0, TString *buf = 0);

   ClassDef(TXNetSystem,0)   // System management class for xrootd servers
};

//
// Simple guard class for connections
//
class TXNetSystemConnectGuard {

private:
   XrdClientAdmin *fClientAdmin;  // Handle to the client admin object

public:
   TXNetSystemConnectGuard(TXNetSystem *xn, const char *url);
   ~TXNetSystemConnectGuard();

   bool IsValid() const { return ((fClientAdmin) ? 1 : 0); }

   XrdClientAdmin *ClientAdmin() const { return fClientAdmin; }

   void NotifyLastError();
};

#endif
