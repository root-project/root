// Author: Rene Brun   02/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream.h>

#include "Strlen.h"
#include "TFolder.h"
#include "TBrowser.h"
#include "TROOT.h"
#include "TError.h"
#include "TRegexp.h"


//______________________________________________________________________________
//Begin_Html
/*
<img src="gif/TFolder_classtree.gif">
*/
//End_Html

ClassImp(TFolder)

//______________________________________________________________________________
//

//______________________________________________________________________________
TFolder::TFolder() : TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*-*Directory default constructor-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
   fFolders = 0;
}

//______________________________________________________________________________
TFolder::TFolder(const char *name, const char *title)
           : TNamed(name, title)
{
//*-*-*-*-*-*-*-*-*-*-*-* Create a new Folder *-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                     ===================
//  A new folder with name,title is created.
//  To add this folder to another folder, use the Append function.
//
//  Note that the folder name cannot contain slashes.
//
   if (strchr(name,'/')) {
      ::Error("TFolder::TFolder","folder name cannot contain a slash", name);
      return;
   }
   if (strlen(GetName()) == 0) {
      ::Error("TFolder::TFolder","folder name cannot be \"\"");
      return;
   }

   fFolders = new TList();
}

//______________________________________________________________________________
TFolder::TFolder(const TFolder &folder)
{
   ((TFolder&)folder).Copy(*this);
}

//______________________________________________________________________________
TFolder::~TFolder()
{
//*-*-*-*-*-*-*-*-*-*-*-*Directory destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ====================

   if (gROOT == 0) return; //when called by TROOT destructor

   TCollection::StartGarbageCollection();

   if (fFolders) {
      fFolders->Delete();
      SafeDelete(fFolders);
   }

   TCollection::EmptyGarbageCollection();

   if (gDebug)
      cerr << "TFolder dtor called for "<< GetName() << endl;
}

//______________________________________________________________________________
void TFolder::Append(TObject *obj)
{
   // Append object to this folder.

   if (obj == 0 || fFolders == 0) return;
   fFolders->Add(obj);
}

//______________________________________________________________________________
void TFolder::Browse(TBrowser *b)
{

   if (fFolders) fFolders->Browse(b);
}

//______________________________________________________________________________
void TFolder::Clear(Option_t *)
{
//*-*-*-*Delete all objects from a Directory list-*-*-*-*-*
//*-*    =======================================

   if (fFolders) fFolders->Clear();

}

//______________________________________________________________________________
TObject *TFolder::Get(const char *name)
{
// search object identified by name in the tree of folders inside
// this folder.
// name may contain "/"

   TObject *idcur = fFolders->FindObject(name);
   return idcur;
}

//______________________________________________________________________________
const char *TFolder::GetPath() const
{
   // Returns the full path of the folder. E.g. file://root/dir1/dir2.
   // The returned path will be re-used by the next call to GetPath().

   static char *path = 0;
   const int kMAXDEPTH = 128;
   const TFolder *d[kMAXDEPTH];
   const TFolder *cur = this;
   int depth = 0, len = 0;

   d[depth++] = cur;
   len = strlen(cur->GetName()) + 1;  // +1 for the /

//   while (cur->fMother && depth < kMAXDEPTH) {
//      cur = (TFolder *)cur->fMother;
//      d[depth++] = cur;
//      len += strlen(cur->GetName()) + 1;
//   }

   if (path) delete [] path;
   path = new char[len+2];

   for (int i = depth-1; i >= 0; i--) {
      if (i == depth-1) {    // file or TROOT name
         strcpy(path, d[i]->GetName());
         strcat(path, ":");
         if (i == 0) strcat(path, "/");
      } else {
         strcat(path, "/");
         strcat(path, d[i]->GetName());
      }
   }

   return path;
}

//______________________________________________________________________________
void TFolder::ls(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*List Folder contents*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =======================
//  Indentation is used to identify the folder tree
//
//  The <regexp> will be used to match the name of the objects.
//
   IndentLevel();
   cout <<ClassName()<<"*\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TObject::IncreaseDirLevel();

   TString opta = option;
   TString opt  = opta.Strip(TString::kBoth);
   TString reg = "*";
   reg = opt;

   TRegexp re(reg, kTRUE);

   TObject *obj;
   TIter nextobj(fFolders);
   while ((obj = (TObject *) nextobj())) {
      TString s = obj->GetName();
      if (s.Index(re) == kNPOS) continue;
      obj->ls(option);
   }
   TObject::DecreaseDirLevel();
}

//______________________________________________________________________________
void TFolder::Print(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Print all objects in the folder *-*-*-*-*-*-*-*
//*-*                    ===============================
//

   fFolders->ForEach(TObject,Print)(option);
}

//______________________________________________________________________________
void TFolder::RecursiveRemove(TObject *obj)
{
//*-*-*-*-*-*-*-*Recursively remove object from a Folder*-*-*-*-*-*-*-*
//*-*            =========================================

   fFolders->RecursiveRemove(obj);
}
