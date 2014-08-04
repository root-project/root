// @(#)root/base:$Id$
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
#include "TList.h"


ClassImp(TSystemDirectory);

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
TSystemDirectory::TSystemDirectory(const TSystemDirectory& sd) :
  TSystemFile(sd),
  fDirsInBrowser(sd.fDirsInBrowser),
  fFilesInBrowser(sd.fFilesInBrowser)
{
   //copy constructor
}

//______________________________________________________________________________
TSystemDirectory& TSystemDirectory::operator=(const TSystemDirectory& sd)
{
   //assignment operator
   if(this!=&sd) {
      TSystemFile::operator=(sd);
      fDirsInBrowser=sd.fDirsInBrowser;
      fFilesInBrowser=sd.fFilesInBrowser;
   }
   return *this;
}

//______________________________________________________________________________
TSystemDirectory::~TSystemDirectory()
{
   // Delete system directory object.

   delete fDirsInBrowser;
   delete fFilesInBrowser;
}

//______________________________________________________________________________
TList *TSystemDirectory::GetListOfFiles() const
{
   // Returns a TList of TSystemFile objects representing the contents
   // of the directory. It's the responsibility of the user to delete
   // the list (the list owns the contained objects).
   // Returns 0 in case of errors.

   void *dir = gSystem->OpenDirectory(GetTitle());
   if (!dir) return 0;

   const char *file = 0;
   TList *contents  = new TList;
   contents->SetOwner();
   while ((file = gSystem->GetDirEntry(dir))) {
      if (IsItDirectory(file)) {
         TString sdirpath;
         if (file[0] == '.' && file[1] == '\0')
            sdirpath = GetTitle();
         else if (file[0] == '.' && file[1] == '.' && file[2] == '.')
            sdirpath = gSystem->DirName(GetTitle());
         else {
            sdirpath = GetTitle();
            if (!sdirpath.EndsWith("/"))
               sdirpath += "/";
            sdirpath += file;
         }
         contents->Add(new TSystemDirectory(file, sdirpath.Data()));
      } else
         contents->Add(new TSystemFile(file, GetTitle()));
   }
   gSystem->FreeDirectory(dir);
   return contents;
}

//______________________________________________________________________________
void TSystemDirectory::SetDirectory(const char *name)
{
   // Create a system directory object.

   SetName(name);
   SetTitle(name);
}

//______________________________________________________________________________
Bool_t TSystemDirectory::IsItDirectory(const char *name) const
{
   // Check if name is a directory.

   Long64_t size;
   Long_t id, flags, modtime;
   const char *dirfile = GetTitle();
   TString savDir = gSystem->WorkingDirectory();

   gSystem->ChangeDirectory(dirfile);
   flags = id = size = modtime = 0;
   gSystem->GetPathInfo(name, &id, &size, &flags, &modtime);
   Int_t isdir = (Int_t)flags & 2;

   gSystem->ChangeDirectory(savDir);
   return isdir ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TSystemDirectory::Browse(TBrowser *b)
{
   // Browse OS system directories.

   // Collections to keep track of all browser objects that have been
   // generated. It's main goal is to prevent the contineous
   // allocations of new objects with the same names during browsing.
   if (!fDirsInBrowser)  fDirsInBrowser  = new TOrdCollection;
   if (!fFilesInBrowser) fFilesInBrowser = new TOrdCollection(10);

   const char *name = GetTitle();
   TSystemFile *sfile;
   TSystemDirectory *sdir;
   const char *file;

   gSystem->ChangeDirectory(name);

   if (GetName()[0] == '.' && GetName()[1] == '.')
      SetName(gSystem->BaseName(name));

   void *dir = gSystem->OpenDirectory(name);

   if (!dir)
      return;

   while ((file = gSystem->GetDirEntry(dir))) {
      if (b->TestBit(TBrowser::kNoHidden) && file[0] == '.' && file[1] != '.' )
         continue;
      if (IsItDirectory(file)) {
         TString sdirpath;
         if (!strcmp(file, "."))
            sdirpath = name;
         else if (!strcmp(file, ".."))
            sdirpath = gSystem->DirName(name);
         else {
            sdirpath =  name;
            if (!sdirpath.EndsWith("/"))
               sdirpath += "/";
            sdirpath += file;
         }
         if (!(sdir = FindDirObj(sdirpath.Data()))) {
            sdir = new TSystemDirectory(file, sdirpath.Data());
            fDirsInBrowser->Add(sdir);
         }
         b->Add(sdir, file);
      } else {
         if (!(sfile = FindFileObj(file, gSystem->WorkingDirectory()))) {
            sfile = new TSystemFile(file, gSystem->WorkingDirectory());
            fFilesInBrowser->Add(sfile);
         }
         b->Add(sfile, file);
      }
   }
   gSystem->FreeDirectory(dir);
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
