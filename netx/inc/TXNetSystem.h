// @(#)root/netx:$Name:  $:$Id: TXNetSystem.h,v 1.3 2006/03/16 09:08:08 rdm Exp $
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

#ifndef ROOT_TNetSystem
#include "TNetFile.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "XrdOuc/XrdOucString.hh"
#include "XrdClient/XrdClientVector.hh"

class XrdClientAdmin;

typedef XrdClientVector<XrdOucString> vecString;
typedef XrdClientVector<bool>         vecBool;


class TXNetSystem : public TNetSystem {

private:
   XrdClientAdmin *fClientAdmin;  // Handle to the client admin object
   Bool_t          fIsRootd;      // Nature of remote file server
   Bool_t          fIsXRootd;     // Nature of remote file server
   TString         fDir;          // Current directory
   void           *fDirp;         // Directory pointer
   vecString       fDirList;      // Buffer for directory content
   Bool_t          fDirListValid; // fDirList content valid ?

   static Bool_t   fgInitDone;    // Avoid initializing more than once
   static Bool_t   fgRootdBC;     // Control rootd backward compatibility

   void           *GetDirPtr() const { return fDirp; }
   void            InitXrdClient();
   void            SaveEndPointUrl();

public:
   TXNetSystem(Bool_t owner = kTRUE);
   TXNetSystem(const char *url, Bool_t owner = kTRUE);
   virtual ~TXNetSystem();

   Bool_t              AccessPathName(const char *path, EAccessMode mode);
   virtual Bool_t      ConsistentWith(const char *path, void *dirptr);
   virtual void        FreeDirectory(void *dirp);
   virtual const char *GetDirEntry(void *dirp);
   virtual Int_t       GetPathInfo(const char* path, FileStat_t &buf);
   virtual Int_t       MakeDirectory(const char* dir);
   virtual void       *OpenDirectory(const char* dir);

   ClassDef(TXNetSystem,0)   // System management class for xrootd servers
};

#endif
