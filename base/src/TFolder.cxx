// @(#)root/base:$Name:  $:$Id: TFolder.cxx,v 1.4 2000/09/06 09:29:20 brun Exp $
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
// The Root folders hierarchy can be seen as a whiteboard where objects
// are posted. Other classes/tasks can access these objects by specifying
// only a string pathname. This whiteboard facility greatly improves the
// modularity of an application, minimizing the class relationship problem
// that penalizes large applications.
//
// If we assume a sets of tasks T1, T2, T3, T4, etc (could be TTask objects),
// each task may be one or more classes with strong relationships in the
// form of data member pointers. This is an efficient way of communication.
// However, one has interest to minimize direct coupling between tasks
// in the form of direct pointers. One better uses the naming and search
// service provided by the Root folders hierarchy. This makes the tasks
// loosely coupled and also greatly facilitates I/O operations.
// In a client/server environment, this mechanism facilitates the access
// to any kind of object in //root stores running on different processes.
//
// A TFolder is created by invoking the TFolder constructor. It is placed
// inside an existing folder via the TFolder::AddFolder method.
// One can search for a folder or an object in a folder using the FindObject
// method. FindObject analyzes the string passed as its argument and searches
// in the hierarchy until it finds an object or folder matching the name.
//
// When a folder is deleted, its reference from the parent folder and
// possible other folders is deleted.
//
// If a folder has been declared the owner of its objects/folders via
// TFolder::SetOwner, then the contained objects are deleted when the
// folder is deleted. By default, a folder does not own its contained objects.
//
// Standard Root objects are automatically added to the folder hierarchy.
// For example, the following folders exist:
//   //root/Files      with the list of currently connected Root files
//   //root/Classes    with the list of active classes
//   //root/Geometries with active geometries
//   //root/Canvases   with the list of active canvases
//   //root/Styles     with the list of graphics styles
//   //root/Colors     with the list of active colors
//
// For example, if a file "myFile.root" is added to the list of files, one can
// retrieve a pointer to the corresponding TFile object with a statement like:
//   TFile *myFile = (TFile*)gROOT->FindObject("//root/Files/myFile.root");
// The above statement can be abbreviated to:
//   TFile *myFile = (TFile*)gROOT->FindObject("/Files/myFile.root");
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
   fIsOwner = kFALSE;
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
// Note that a folder can be added to several folders
//
// if (collection is non NULL, the pointer fFolders is set to the existing
// collection, otherwise a default collection (Tlist) is created
// Note that the folder name cannot contain slashes.
   
   if (strchr(name,'/')) {
      ::Error("TFolder::TFolder","folder name cannot contain a slash", name);
      return 0;
   }
   if (strlen(GetName()) == 0) {
      ::Error("TFolder::TFolder","folder name cannot be \"\"");
      return 0;
   }
   TFolder *folder = new TFolder();
   folder->SetName(name);
   folder->SetTitle(title);
   if (!fFolders) fFolders = new TList(); //only true when gROOT creates its 1st folder
   fFolders->Add(folder);

   if (collection) folder->fFolders = collection;
   else            folder->fFolders = new TList();
   return folder;
}

//______________________________________________________________________________
void TFolder::Browse(TBrowser *b)
{
// Browse this folder
   
   if (fFolders) fFolders->Browse(b);
}

//______________________________________________________________________________
void TFolder::Clear(Option_t *option)
{
// Delete all objects from a folder list

   if (fFolders) fFolders->Clear(option);

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
//     //root/xxx/yyy/name
//
//   B, specify a pathname starting with a single slash. //root is assumed
//     /xxx/yyy/name
//
//   C, Specify a pathname relative to this folder
//     xxx/yyy/name
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
TObject *TFolder::FindObjectAny(const char *name) const
{
// return a pointer to the first object with name starting at this folder
   
   TIter next(fFolders);
   TObject *obj;
   TFolder *folder;
   while ((obj=next())) {
      if (!obj->InheritsFrom(TFolder::Class())) continue;
      folder = (TFolder*)obj;
      return folder->FindObjectAny(name);
   }
   return 0;
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
//   if option contains "dump",  the Dump function of contained objects is called
//   if option contains "print", the Print function of contained objects is called
//   By default the ls function of contained objects is called.
//  Indentation is used to identify the folder tree
//
//  The <regexp> will be used to match the name of the objects.
//
   TROOT::IndentLevel();
   cout <<ClassName()<<"*\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TROOT::IncreaseDirLevel();

   TString opta = option;
   TString opt  = opta.Strip(TString::kBoth);
   opt.ToLower();
   TString reg  = opt;
   Bool_t dump  = opt.Contains("dump");
   Bool_t print = opt.Contains("print");
   TRegexp re(reg, kTRUE);

   TObject *obj;
   TIter nextobj(fFolders);
   while ((obj = (TObject *) nextobj())) {
      TString s = obj->GetName();
      if (s.Index(re) == kNPOS) continue;
      if (dump)      obj->Dump();
      else if(print) obj->Print(option);
      else           obj->ls(option);
   }
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
void TFolder::RecursiveRemove(TObject *obj)
{
// Recursively remove object from a Folder

   fFolders->RecursiveRemove(obj);
}

//______________________________________________________________________________
void TFolder::Remove(TObject *obj)
{
// Remove object from this folder. obj must be a TObject or a TFolder

   if (obj == 0 || fFolders == 0) return;
   fFolders->Remove(obj);
}
