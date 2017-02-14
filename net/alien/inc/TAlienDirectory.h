// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienDirectory
#define ROOT_TAlienDirectory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienDirectory                                                      //
//                                                                      //
// Class which creates Directory files for the AliEn middleware.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TBrowser.h"
#include "TNamed.h"
#include "TMap.h"


class TAlienDirectoryEntry : public TNamed {

private:
   TString fLfn;             // logical file name
   TMap    fBrowserObjects;  // objects shown in browser

public:
   TAlienDirectoryEntry(const char *lfn, const char *name) : TNamed(name,name) { fLfn = lfn; }
   virtual ~TAlienDirectoryEntry() { }
   Bool_t IsFolder() const { return kTRUE; }
   void Browse(TBrowser *b);

   ClassDef(TAlienDirectoryEntry,1)  // Creates Directory files entries for the AliEn middleware
};


class TAlienDirectory : public TNamed {

private:
   TList fEntries;          // directory entries
   TMap  fBrowserObjects;   // objects shown in browser

public:
   TAlienDirectory(const char *ldn, const char *name=0);
   virtual ~TAlienDirectory();
   void   Fill();
   Bool_t IsFolder() const { return kTRUE; }
   void   Browse(TBrowser *b);

   ClassDef(TAlienDirectory,1)  // Creates Directory files for the AliEn middleware
};

#endif
