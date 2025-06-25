// @(#)root/net:$Id$
// Author: Adrien Devresse and Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDavixSystem
#define ROOT_TDavixSystem

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDavixSystem                                                         //
//                                                                      //
// A TSystem specialization for HTTP and WebDAV                         //
// It supports HTTP and HTTPS in a number of dialects and options       //
//  e.g. S3 is one of them                                              //
// Other caracteristics come from the full support of Davix,            //
//  e.g. full redirection support in any circumstance                   //
//                                                                      //
// Authors:     Adrien Devresse (CERN IT/SDC)                           //
//              Fabrizio Furano (CERN IT/SDC)                           //
//                                                                      //
// September 2013                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystem.h"

class TDavixFileInternal;

class TDavixSystem : public TSystem {
private:
   TDavixFileInternal* d_ptr;

public:
   TDavixSystem();
   TDavixSystem(const char *url);

   virtual ~TDavixSystem();

   void FreeDirectory(void *dirp) override;
   const char *GetDirEntry(void *dirp) override;
   Bool_t ConsistentWith(const char *path, void *dirptr) override;

   Int_t GetPathInfo(const char* path, FileStat_t &buf) override;
   Bool_t IsPathLocal(const char *path) override;
   virtual Int_t Locate(const char* path, TString &endurl);
   Int_t MakeDirectory(const char* dir) override;
   void *OpenDirectory(const char* dir) override;
   int Unlink(const char *path) override;

   ClassDefOverride(TDavixSystem, 0);
};

#endif
