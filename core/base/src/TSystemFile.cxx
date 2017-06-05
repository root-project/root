// @(#)root/base:$Id$
// Author: Rene Brun   26/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSystemFile
\ingroup Base

A TSystemFile describes an operating system file.
The information is used by the browser (see TBrowser).
*/

#include "TSystemFile.h"
#include "TBrowser.h"
#include "TSystem.h"
#include "TEnv.h"


ClassImp(TSystemFile);

////////////////////////////////////////////////////////////////////////////////
/// TSystemFile default constructor

TSystemFile::TSystemFile() : TNamed()
{
}

////////////////////////////////////////////////////////////////////////////////
/// TSystemFile normal constructor

TSystemFile::TSystemFile(const char *filename, const char *dirname)
   : TNamed(filename, dirname)
{
   SetBit(kCanDelete);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TSystemFile object.

TSystemFile::~TSystemFile()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Check if object is a directory.

Bool_t TSystemFile::IsDirectory(const char *dir) const
{
   Long64_t size;
   Long_t id, flags, modtime;

   flags = id = size = modtime = 0;
   gSystem->GetPathInfo(!dir ? fName.Data() : dir, &id, &size, &flags, &modtime);
   Int_t isdir = (Int_t)flags & 2;

   return isdir ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute default action for this system file (action is specified
/// in the $HOME/.root.mimes or $ROOTSYS/etc/root.mimes file.

void TSystemFile::Browse(TBrowser *b)
{
   if (b)
      b->ExecuteDefaultAction(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke text editor on this file

void TSystemFile::Edit()
{
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

////////////////////////////////////////////////////////////////////////////////
/// copy this file

void TSystemFile::Copy(const char *to)
{
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

////////////////////////////////////////////////////////////////////////////////
/// move this file

void TSystemFile::Move(const char *to)
{
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

////////////////////////////////////////////////////////////////////////////////
/// delete this file

void TSystemFile::Delete()
{
   gSystem->Unlink(fName);
}

////////////////////////////////////////////////////////////////////////////////
/// rename this file

void TSystemFile::Rename(const char *name)
{
   gSystem->Rename(fName, name);
}

////////////////////////////////////////////////////////////////////////////////
/// inspect this file

void TSystemFile::Inspect() const
{
}

////////////////////////////////////////////////////////////////////////////////
/// dump this file

void TSystemFile::Dump() const
{
}

