// @(#)root/netxng:$Id$
/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNetXNGSystem
#define ROOT_TNetXNGSystem

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TNetXNGSystem                                                              //
//                                                                            //
// Authors: Justin Salmon, Lukasz Janyst                                      //
//          CERN, 2013                                                        //
//                                                                            //
// Enables access to XRootD filesystem interface using the new client.        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TSystem.h"
#include "TCollection.h"
#include "TMutex.h"
#include "THashList.h"
#include <set>

#ifndef __CINT__
#include <XrdCl/XrdClXRootDResponses.hh>
#include <XrdCl/XrdClURL.hh>
#endif

namespace XrdCl {
   class FileSystem;
   class URL;
   class DirectoryList;
}

class TNetXNGSystem: public TSystem {

private:
   std::set<void *>   fDirPtrs;
   static THashList   fgAddrFQDN;  // Cache of addresses to FQDNs
   static TMutex      fgAddrMutex; // Serialise access to the FQDN list
#ifndef __CINT__
private:
   XrdCl::URL        *fUrl;        // URL of this TSystem
   XrdCl::FileSystem *fFileSystem; // Cached for convenience

#endif

public:
   TNetXNGSystem(Bool_t owner = kTRUE);
   TNetXNGSystem(const char *url, Bool_t owner = kTRUE);
   virtual ~TNetXNGSystem();

   virtual void*       OpenDirectory(const char* dir);
   virtual Int_t       MakeDirectory(const char* dir);
   virtual void        FreeDirectory(void *dirp);
   virtual const char* GetDirEntry(void *dirp);
   virtual Int_t       GetPathInfo(const char* path, FileStat_t &buf);
   virtual Bool_t      ConsistentWith(const char *path, void *dirptr);
   virtual int         Unlink(const char *path);
   virtual Bool_t      IsPathLocal(const char *path);
   virtual Int_t       Locate(const char* path, TString &endurl);
   virtual Int_t       Stage(const char* path, UChar_t priority);
   virtual Int_t       Stage(TCollection *files, UChar_t priority);

   ClassDef(TNetXNGSystem, 0 ) // ROOT class definition
};

#endif
