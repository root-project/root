// @(#)root/base:$Name:  $:$Id: TFolder.cxx,v 1.0 2000/09/05 09:21:22 brun Exp $
// Author: Rene Brun   02/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//______________________________________________________________________________
//
// A TFolder object is a collection of objects and folders.
// Folders have a name and a title and are identified in the folder hierarchy
// by a "Unix-like" naming mechanism. The root of all folders is //root.
// New folders can be dynamically added or removed to/from a folder.
// The folder hierarchy can be visualized via the TBrowser.
//
// A TFolder is created by invoking the TFolder constructor. It is placed
// inside an existing folder via the TFolder::Add method.
// One can search for a folder or an object in a folder using the FindObject
// method. FindObject analyzes the string passed as its argument and searches
// in the hierarchy until it finds an object or folder matching the name.
//
// Standard Root objects are automatically added to the folder hierarchy.
// For example, the following folders exist:
//   //root/files      with the list of currently connected Root files
//   //root/classes    with the list of active classes
//   //root/geometries with active geometries
//   //root/canvases   with the list of active canvases
//   //root/styles     with the list of graphics styles
//   //root/colors     with the list of active colors
//
// For example, if a file "myFile.root" is added to the list of files, one can
// retrieve a pointer to the corresponding TFile object with a statement like:
//   TFile *myFile = (TFile*)gROOT->FindObject("//root/files/myFile.root");
// The above statement can be abbreviated to:
//   TFile *myFile = (TFile*)gROOT->FindObject("/files/myFile.root");
// or even to:
//   TFile *myFile = (TFile*)gROOT->FindObject("myFile.root");
// In this last case, the TROOT::FindObject function will scan the folder hierarchy
// starting at //root and will return the first object named "myFile.root".
//
// Because a string-based search mechanism is expensive, it is recommended
// to save the pointer to the object as a class member or local variable
// if this pointer is used frequently or inside loops.
//
//Begin_Html
/*
<img src="gif/folder.gif">
*/
//End_Html

#include <iostream.h>

#include "Strlen.h"
#include "TFolder.h"
#include "TBrowser.h"
#include "TROOT.h"
#include "TError.h"
#include "TRegexp.h"

ClassImp(TFolder)

//______________________________________________________________________________
TFolder::TFolder() : TNamed()
{
// default constructor used by the Input functions
//
   fFolders = 0;
}

//______________________________________________________________________________
TFolder::TFolder(const char *name, const char *title, TCollection *collection)
           : TNamed(name, title)
{
// Create a new Folder
//
//  A new folder with name,title is created.
//  To add this folder to another folder, use the Add function.
//  if (collection is non NULL, the pointer fFolders is set to the existing
//  collection, otherwise a default collection (Tlist) is created
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

   if (collection) fFolders = collection;
   else            fFolders = new TList();
}

//______________________________________________________________________________
TFolder::TFolder(const TFolder &folder)
{
   ((TFolder&)folder).Copy(*this);
}

//______________________________________________________________________________
TFolder::~TFolder()
{
// folder destructor. Remove all objects from its lists and delete
// all its sub folders

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
void TFolder::Add(TObject *obj)
{
   // Add object to this folder. obj must be a TObject or a TFolder

   if (obj == 0 || fFolders == 0) return;
   fFolders->Add(obj);
}

//______________________________________________________________________________
TFolder *TFolder::AddFolder(const char *name, const char *title, TCollection *collection)
{
// Create a new folder and add it to the list of folders of this folder
// return a pointer to the created folder

   TFolder *folder = new TFolder(name,title,collection);
   fFolders->Add(folder);
   return folder;
}

//______________________________________________________________________________
void TFolder::Browse(TBrowser *b)
{
// Browse this folder
   
   if (fFolders) fFolders->Browse(b);
}

//______________________________________________________________________________
void TFolder::Clear(Option_t *)
{
// Delete all objects from a folder list

   if (fFolders) fFolders->Clear();

}

//______________________________________________________________________________
TObject *TFolder::FindObject(TObject *) const
{
// find object in an folder
   
   Error("FindObject","Not yet implemented");
   return 0;
}

//______________________________________________________________________________
TObject *TFolder::FindObject(const char *name) const
{
// search object identified by name in the tree of folders inside
// this folder.
// name may be of the forms:
//   A, specify a full pathname starting at the top ROOT folder
//     /xxx/yyy/name
//
//   B, Specify a pathname relative to this folder
//     xxx/yyy/name
//     yyy/name
//     name
   
   if (name == 0) return 0;
   if (name[0] == '/') return gROOT->GetRootFolder()->FindObject(name+1);
   char cname[1024];
   strcpy(cname,name);
   TObject *obj;
   char *slash = strchr(cname,'/');
   if (slash) {
      *slash = 0;
      obj = fFolders->FindObject(cname);
      if (!obj) return 0;
      return obj->FindObject(slash+1);
   } else {
      return fFolders->FindObject(name);
   }
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
// List folder contents
//  Indentation is used to identify the folder tree
//
//  The <regexp> will be used to match the name of the objects.
//
   TROOT::IndentLevel();
   cout <<ClassName()<<"*\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TROOT::IncreaseDirLevel();

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
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
void TFolder::Print(Option_t *option)
{
// Invoke the Print function of all objects/folders in this folder

   fFolders->ForEach(TObject,Print)(option);
}

//______________________________________________________________________________
void TFolder::RecursiveRemove(TObject *obj)
{
// Recursively remove object from a Folder

   fFolders->RecursiveRemove(obj);
}

//______________________________________________________________________________
void TFolder::SetCollection(TCollection *collection)
{
// Set the collection of folders pointing to collection
// if fFolders points to an existing collection, the previous collection
// is deleted.

   if (fFolders) fFolders->Delete();
   delete fFolders;
   fFolders = collection;
}
