// @(#)root/base:$Id$
// Author: Rene Brun   26/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSystemFile                                                          //
//                                                                      //
// A TSystemFile describes an operating system file.                    //
// The information is used by the browser (see TBrowser).               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystemFile.h"
#include "TBrowser.h"
#include "TSystem.h"
#include "TEnv.h"


ClassImp(TSystemFile)

//______________________________________________________________________________
TSystemFile::TSystemFile() : TNamed()
{
   // TSystemFile default constructor

}

//______________________________________________________________________________
TSystemFile::TSystemFile(const char *filename, const char *dirname)
   : TNamed(filename, dirname)
{
   // TSystemFile normal constructor

   SetBit(kCanDelete);
}

//______________________________________________________________________________
TSystemFile::~TSystemFile()
{
   // Delete TSystemFile object.
}

//______________________________________________________________________________
Bool_t TSystemFile::IsDirectory(const char *dir) const
{
   // Check if object is a directory.

   Long64_t size;
   Long_t id, flags, modtime;

   flags = id = size = modtime = 0;
   gSystem->GetPathInfo(!dir ? fName.Data() : dir, &id, &size, &flags, &modtime);
   Int_t isdir = (Int_t)flags & 2;

   return isdir ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TSystemFile::Browse(TBrowser *b)
{
   // Execute default action for this system file (action is specified
   // in the $HOME/.root.mimes or $ROOTSYS/etc/root.mimes file.

   if (b)
      b->ExecuteDefaultAction(this);
}

//______________________________________________________________________________
void TSystemFile::Edit()
{
   // Invoke text editor on this file

#ifndef _WIN32
   const char *ed = gEnv->GetValue("Editor", "vi");
   Int_t nch = strlen(ed)+strlen(GetName()) + 50;
   Char_t *cmd = new Char_t[nch];
   if (!strcmp(ed, "vi"))
      snprintf(cmd,nch, "xterm -e vi %s &", GetName());
   else
      snprintf(cmd,nch, "%s %s &", ed, GetName());
#else
   const char *ed = gEnv->GetValue("Editor", "notepad");
   Int_t nch = strlen(ed)+strlen(GetName()) + 50;
   Char_t *cmd = new Char_t[nch];
   snprintf(cmd,nch, "start %s %s", ed, GetName());
#endif
   gSystem->Exec(cmd);

   delete [] cmd;
}

//______________________________________________________________________________
void TSystemFile::Copy(const char *to)
{
   // copy this file

   TString name = to;

   if (IsDirectory(to)) {
      if (name.EndsWith("/")) name.Chop();
      char *s = gSystem->ConcatFileName(name, fName);
      name = s;
      delete [] s;
   }

   Int_t status = gSystem->CopyFile(fName, name, kFALSE);

   if (status == -2) {
      Warning("Copy", "File %s already exists", name.Data());
   } else if (status == -1) {
      Warning("Copy", "Failed to move file %s", name.Data());
   }
}

//______________________________________________________________________________
void TSystemFile::Move(const char *to)
{
   // move this file

   if (!to) {
      Warning("Move", "No file/dir name specified");
      return;
   }

   TString name = to;

   if (IsDirectory(to)) {
      if (name.EndsWith("/")) name.Chop();
      char *s = gSystem->ConcatFileName(name, fName);
      name = s;
      delete [] s;
   }
   Int_t status = gSystem->CopyFile(fName, name, kFALSE);

   if (!status) {
      gSystem->Unlink(fName);
   } else if (status == -2) {
      Warning("Move", "File %s already exists", name.Data());
   } else if (status == -1) {
      Warning("Move", "Failed to move file %s", name.Data());
   }
}

//______________________________________________________________________________
void TSystemFile::Delete()
{
   // delete this file

   gSystem->Unlink(fName);
}

//______________________________________________________________________________
void TSystemFile::Rename(const char *name)
{
   // rename this file

   gSystem->Rename(fName, name);
}

//______________________________________________________________________________
void TSystemFile::Inspect() const
{
   // inspect this file
}

//______________________________________________________________________________
void TSystemFile::Dump() const
{
   // dump this file
}

