// @(#)root/base:$Id$
// Author: Christian Bormann  13/10/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSystemDirectory
\ingroup Base

Describes an Operating System directory for the browser.
*/

#include "TSystemDirectory.h"
#include "TSystem.h"
#include "TBrowser.h"
#include "TOrdCollection.h"
#include "TList.h"


ClassImp(TSystemDirectory);

////////////////////////////////////////////////////////////////////////////////
/// Create a system directory object.

TSystemDirectory::TSystemDirectory()
{
   fDirsInBrowser  = 0;
   fFilesInBrowser = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a system directory object.

TSystemDirectory::TSystemDirectory(const char *dirname, const char *path) :
   TSystemFile(dirname, path)
{
   fDirsInBrowser  = 0;
   fFilesInBrowser = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TSystemDirectory::TSystemDirectory(const TSystemDirectory& sd) :
  TSystemFile(sd),
  fDirsInBrowser(sd.fDirsInBrowser),
  fFilesInBrowser(sd.fFilesInBrowser)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TSystemDirectory& TSystemDirectory::operator=(const TSystemDirectory& sd)
{
   if(this!=&sd) {
      TSystemFile::operator=(sd);
      fDirsInBrowser=sd.fDirsInBrowser;
      fFilesInBrowser=sd.fFilesInBrowser;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete system directory object.

TSystemDirectory::~TSystemDirectory()
{
   delete fDirsInBrowser;
   delete fFilesInBrowser;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a TList of TSystemFile objects representing the contents
/// of the directory. It's the responsibility of the user to delete
/// the list (the list owns the contained objects).
/// Returns 0 in case of errors.

TList *TSystemDirectory::GetListOfFiles() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Create a system directory object.

void TSystemDirectory::SetDirectory(const char *name)
{
   SetName(name);
   SetTitle(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if name is a directory.

Bool_t TSystemDirectory::IsItDirectory(const char *name) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Browse OS system directories.

void TSystemDirectory::Browse(TBrowser *b)
{
   // Collections to keep track of all browser objects that have been
   // generated. It's main goal is to prevent the continuous
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

////////////////////////////////////////////////////////////////////////////////
/// Method that returns system directory object if it
/// exists in list, 0 otherwise.

TSystemDirectory *TSystemDirectory::FindDirObj(const char *name)
{
   int size = fDirsInBrowser->GetSize();
   for (int i = 0; i < size; i++) {
      TSystemDirectory *obj = (TSystemDirectory *) fDirsInBrowser->At(i);
      if (!strcmp(name, obj->GetTitle()))
         return obj;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Method that returns system file object if it exists in
/// list, 0 otherwise.

TSystemFile *TSystemDirectory::FindFileObj(const char *name, const char *dir)
{
   int size = fFilesInBrowser->GetSize();
   for (int i = 0; i < size; i++) {
      TSystemFile *obj = (TSystemFile *) fFilesInBrowser->At(i);
      if (!strcmp(name, obj->GetName()) && !strcmp(dir, obj->GetTitle()))
         return obj;
   }
   return 0;
}
