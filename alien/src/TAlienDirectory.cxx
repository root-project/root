// @(#)root/alien:$Name:  $:$Id: TAlienDirectory.cxx,v 1.1 2005/05/20 11:13:30 rdm Exp $
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienDirectory                                                      //
//                                                                      //
// Class which creates Directory files for the AliEn middleware         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienDirectory.h"
#include "TAlien.h"
#include "TGridResult.h"
#include "TSystemFile.h"
#include "TAlienFile.h"
#include "TSystem.h"
#include "TFile.h"


ClassImp(TAlienDirectoryEntry)

//______________________________________________________________________________
void TAlienDirectoryEntry::Browse(TBrowser* b)
{
   if (b) {
      TAlienFile* newfile  = (new TAlienFile(fLfn.Data()));
      b->Add(newfile);
   }
}


ClassImp(TAlienDirectory)

//______________________________________________________________________________
TAlienDirectory::TAlienDirectory(const char *ldn, const char *name)
{
   if (!gGrid->Cd(ldn)) {
      MakeZombie();
      return;
   }

   if (!name) {
      SetName(gSystem->BaseName(ldn));
   } else {
      SetName(name);
   }

   SetTitle(ldn);

   TGridResult *dirlist = gGrid->Ls(ldn,"-la");
   if (dirlist) {
      dirlist->Sort();
      Int_t i =0;
      while (dirlist->GetFileName(i)) {
         if (!strcmp(".",dirlist->GetFileName(i))) {
            i++;
            continue;
         }
         if (!strcmp("..",dirlist->GetFileName(i))) {
            i++;
            continue;
         }

         if (dirlist->GetKey(i,"permissions")[0] == 'd') {
            fEntries.Add(new TAlienDirectory(dirlist->GetFileNamePath(i)));
         } else {
            fEntries.Add(new TAlienDirectoryEntry(dirlist->GetFileNamePath(i),dirlist->GetFileName(i)));
         }
         i++;
      }
      delete dirlist;
   }
}

//______________________________________________________________________________
void TAlienDirectory::Browse(TBrowser *b)
{
   // Browser interface to ob status.

   if (b) {
      TIterator *iter = fEntries.MakeIterator();
      TObject *obj = 0;
      while ((obj = iter->Next()) != 0) {
         b->Add(obj);
      }
      delete iter;
   }
}

//______________________________________________________________________________
TAlienDirectory::~TAlienDirectory()
{
   fEntries.Clear();
}
