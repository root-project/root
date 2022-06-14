// @(#)root/alien:$Id$
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
#include "TObjString.h"


ClassImp(TAlienDirectoryEntry);

////////////////////////////////////////////////////////////////////////////////
/// Browse an Alien directory.

void TAlienDirectoryEntry::Browse(TBrowser* b)
{
   if (b) {
      TString alienname = "alien://";
      alienname += fLfn;
      if (!fBrowserObjects.FindObject(alienname)) {
         TFile *newfile = TFile::Open(alienname.Data());
         b->Add(newfile);
         fBrowserObjects.Add(new TObjString(alienname.Data()), (TObject*) newfile);
      }
   }
}


ClassImp(TAlienDirectory);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

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
};

////////////////////////////////////////////////////////////////////////////////
/// Fill directory entry list.

void TAlienDirectory::Fill()
{
   if (!gGrid->Cd(GetTitle())) {
      MakeZombie();
      return;
   }

   fEntries.Clear();
   TGridResult *dirlist = gGrid->Ls(GetTitle(), "-la");
   if (dirlist) {
      dirlist->Sort();
      Int_t i = 0;
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
            fEntries.Add(new TAlienDirectoryEntry(dirlist->GetFileNamePath(i), dirlist->GetFileName(i)));
         }
         i++;
      }
      delete dirlist;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Browser interface to ob status.

void TAlienDirectory::Browse(TBrowser *b)
{
   if (b) {
      Fill();
      TIter next(&fEntries);
      TObject *obj = 0;
      TObject *bobj = 0;
      while ((obj = next())) {
         if (!(bobj = fBrowserObjects.FindObject(obj->GetName()))) {
            b->Add(obj, obj->GetName());
            fBrowserObjects.Add(new TObjString(obj->GetName()), (TObject*) obj);
         } else {
            b->Add(bobj, bobj->GetName());
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TAlienDirectory::~TAlienDirectory()
{
   fEntries.Clear();
}
