// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TFileSet.h"
#include "TBrowser.h"
#include "TSystem.h"

#ifndef WIN32
#include <errno.h>
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileSet                                                             //
//                                                                      //
// TFileSet class is a class to convert the                             //
//      "native file system structure"                                  //
// into an instance of the TDataSet class                               //
//                                                                      //
//  Example:                                                            //
//    How to convert your home directory into the OO dataset            //
//                                                                      //
//  root [0] TString home = "$HOME";                                    //
//  root [1] TFileSet set(home);                                        //
//  root [2] TBrowser b("MyHome",&set);                                 //
//  root [3] set.ls("*");                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TFileSet)

//______________________________________________________________________________
TFileSet::TFileSet()
         : TDataSet()
{
   //to be documented
}

//______________________________________________________________________________
TFileSet::TFileSet(const TString &dirname,const Char_t *setname,Bool_t expand, Int_t maxDepth)
           : TDataSet()
{
  //
  // Creates TFileSet
  // Convert the "opearting system" file system tree into the memory resided TFileSet
  //
  //  Parameters:
  //  -----------
  //  dirname  - the name of the "native file system" directory
  //             to convert into TFileSet
  //  setname  - the name of this TFileSet (it is the "base name"
  //                                 of the "dirname" by default)
  //  expand   - flag whether the "dirname" must be "expanded
  //             (kTRUE by default)
  //  maxDeep  - the max number of the levels of the directory to read in
  //             (=10 by default)
  //  Note: If the "dirname" points to non-existent object, for examoe it is dead-link
  //  ----  the object is marked as "Zombie" and this flag is propagated upwards

   if (!maxDepth) return;

   Long64_t size;
   Long_t id, flags, modtime;
   TString dirbuf = dirname;

   if (expand) gSystem->ExpandPathName(dirbuf);
   const char *name= dirbuf;
   if (gSystem->GetPathInfo(name, &id, &size, &flags, &modtime)==0) {

      if (!setname) {
         setname = strrchr(name,'/');
         if (setname) setname++;
      }
      if (setname) SetName(setname);
      else SetName(name);

      // Check if "dirname" is a directory.
      void *dir = 0;
      if (flags & 2 ) {
         dir = gSystem->OpenDirectory(name);
         if (!dir) {
#ifndef WIN32
            perror("can not be open due error\n");
            Error("TFileSet", "directory: %s",name);
#endif
         }
      }
      if (dir) {   // this is a directory
         SetTitle("directory");
         while ( (name = gSystem->GetDirEntry(dir)) ) {
            // skip some "special" names
            if (!name[0] || strcmp(name,"..")==0 || strcmp(name,".")==0) continue;
            Char_t *file = gSystem->ConcatFileName(dirbuf,name);
            TString nextdir = file;
            delete [] file;
            TFileSet *fs = new TFileSet(nextdir,name,kFALSE,maxDepth-1);
            if (fs->IsZombie())  {
               // propagate "Zombie flag upwards
               MakeZombie();
            }
            Add(fs);
         }
         gSystem->FreeDirectory(dir);
      } else
         SetTitle("file");
   } else {
      // Set Zombie flag
      MakeZombie();
      SetTitle("Zombie");
   }
}

//______________________________________________________________________________
TFileSet::~TFileSet()
{
   //to be documented
}

//______________________________________________________________________________
Bool_t TFileSet::IsEmpty() const
{
   //to be documented
   return  strcmp(GetTitle(),"file")!=0 ? kTRUE : kFALSE ;
}

//______________________________________________________________________________
Long_t TFileSet::HasData() const
{
   // This implementation is done in the TDataSet::Purge() method in mind
   // Since this method returns non-zero for files the last are NOT "purged"
   // by TDataSet::Purge()
   //
   return strcmp(GetTitle(),"file")==0 ? 1 : 0;

   //  this must be like this:
   //  return !IsFolder() ;
   //  Alas TObject::IsFolder() isn't defined as "const" (IT IS in 2.25/03)
}

//______________________________________________________________________________
Bool_t TFileSet::IsFolder() const
{
   // If the title of this TFileSet is "file" it is NOT folder
   // see: TFileSet(TString &dirname,const Char_t *setname,Bool_t expand)
   //
   return strcmp(GetTitle(),"file")!=0;
}
