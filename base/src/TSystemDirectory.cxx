// @(#)root/base:$Name:  $:$Id: TSystemDirectory.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Christian Bormann  13/10/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSystemDirectory                                                     //
//                                                                      //
// Describes an Operating System directory for the browser.             //
//                                                                      //
// Author: Christian Bormann  30/09/97                                  //
//         http://www.ikf.physik.uni-frankfurt.de/~bormann/             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystemDirectory.h"
#include "TSystem.h"
#include "TBrowser.h"
#include "TOrdCollection.h"



ClassImp(TSystemDirectory)

//______________________________________________________________________________
TSystemDirectory::TSystemDirectory()
{
   // Create a system directory object.

   fDirsInBrowser  = 0;
   fFilesInBrowser = 0;
}

//______________________________________________________________________________
TSystemDirectory::TSystemDirectory(const char *dirname, const char *path) :
   TSystemFile(dirname, path)
{
   // Create a system directory object.

   fDirsInBrowser  = 0;
   fFilesInBrowser = 0;
}

//______________________________________________________________________________
TSystemDirectory::~TSystemDirectory()
{
   // Delete system directory object.

   delete fDirsInBrowser;
   delete fFilesInBrowser;
}

//______________________________________________________________________________
void TSystemDirectory::SetDirectory(const char *name)
{
   // Create a system directory object.

   SetName(name);
   SetTitle(name);
}

//______________________________________________________________________________
Bool_t TSystemDirectory::IsDirectory(const char* name)
{
   // Check if name is a directory.

   Long_t id, size, flags, modtime;
   const char *dirfile = GetTitle();

   gSystem->ChangeDirectory(dirfile);
   flags = id = size = modtime = 0;
   gSystem->GetPathInfo(name, &id, &size, &flags, &modtime);
   Int_t isdir = (Int_t)flags & 2;

   // this is a directory
   if (isdir) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TSystemDirectory::Browse(TBrowser *b)
{
   // Browse OS system directories.

   // Collections to keep track of all browser objects that have been generated.
   // It's main goal is to prevent the contineous allocations of new
   // objects with the same names during browsing.
   if (!fDirsInBrowser)  fDirsInBrowser  = new TOrdCollection;
   if (!fFilesInBrowser) fFilesInBrowser = new TOrdCollection(10);

   const char *name = GetTitle();
   TSystemFile *sfile;
   TSystemDirectory *sdir;
   const char *file;

   gSystem->ChangeDirectory(name);

#ifdef WIN32
//      gSystem->Exec("explorer .");
#endif

   void *dir = gSystem->OpenDirectory(name);

   if (dir) {
      while ((file = gSystem->GetDirEntry(dir))) {
         if (!strcmp(file,".") || !strcmp(file,"..")) continue;
         if (IsDirectory(file)) {
            TString sdirname(name);
            sdirname = sdirname + "/";
            sdirname = sdirname + file;

            if ((sdir = FindDirObj(sdirname.Data())) == 0) {
               sdir = new TSystemDirectory(file, sdirname.Data());
               fDirsInBrowser->Add(sdir);
            }
            b->Add(sdir, file);
         } else {
            if ((sfile = FindFileObj(file, gSystem->WorkingDirectory())) == 0) {
               sfile = new TSystemFile(file, gSystem->WorkingDirectory());
               fFilesInBrowser->Add(sfile);
            }
            b->Add(sfile, file);
         }
      }
      gSystem->FreeDirectory(dir);
      return;
   }
}

//______________________________________________________________________________
TSystemDirectory *TSystemDirectory::FindDirObj(const char *name)
{
   // Method that returns system directory object if it
   // exists in list, 0 otherwise.

   int size = fDirsInBrowser->GetSize();
   for (int i = 0; i < size; i++) {
      TSystemDirectory *obj = (TSystemDirectory *) fDirsInBrowser->At(i);
      if (!strcmp(name, obj->GetTitle()))
         return obj;
   }
   return 0;
}

//______________________________________________________________________________
TSystemFile *TSystemDirectory::FindFileObj(const char *name, const char *dir)
{
   // Method that returns system file object if it exists in
   // list, 0 otherwise.

   int size = fFilesInBrowser->GetSize();
   for (int i = 0; i < size; i++) {
      TSystemFile *obj = (TSystemFile *) fFilesInBrowser->At(i);
      if (!strcmp(name, obj->GetName()) && !strcmp(dir, obj->GetTitle()))
         return obj;
   }
   return 0;
}
