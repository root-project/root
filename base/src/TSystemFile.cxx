// @(#)root/base:$Name:  $:$Id: TSystemFile.cxx,v 1.2 2001/03/08 20:16:28 rdm Exp $
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
void TSystemFile::Browse(TBrowser *b)
{
   // Execute default action for this system file (action is specified
   // in the $HOME/.root.mimes or $ROOTSYS/etc/root.mimes file.

#ifndef WIN32
   if (b)
      b->ExecuteDefaultAction(this);
#else
#ifndef GDK_WIN32
   Edit();  // Temporary unless the "default action" will be done
#else
   if (b)
      b->ExecuteDefaultAction(this);
#endif
#endif
}

//______________________________________________________________________________
void TSystemFile::Edit()
{
   // Invoke text editor on this file

#ifndef _WIN32
   const char *ed = gEnv->GetValue("Editor", "vi");
   Char_t *cmd = new Char_t[strlen(ed)+strlen(GetName()) + 50];
   if (!strcmp(ed, "vi"))
      sprintf(cmd, "xterm -e vi %s &", GetName());
   else
      sprintf(cmd, "%s %s &", ed, GetName());
#else
   const char *ed = gEnv->GetValue("Editor", "notepad");
   Char_t *cmd = new Char_t[strlen(ed)+strlen(GetName()) + 50];
   sprintf(cmd, "start %s %s", ed, GetName());
#endif
   gSystem->Exec(cmd);

   delete [] cmd;
}
