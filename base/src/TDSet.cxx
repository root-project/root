// @(#)root/base:$Name:  $:$Id: TDSet.cxx,v 1.10 2002/06/16 01:40:36 rdm Exp $
// Author: Fons Rademakers   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDSet                                                                //
//                                                                      //
// This class implements a data set to be used for PROOF processing.    //
// The TDSet defines the class of which objects will be processed,      //
// the directory in the file where the objects of that type can be      //
// found and the list of files to be processed. The files can be        //
// specified as logical file names (LFN's) or as physical file names    //
// (PFN's). In case of LFN's the resolution to PFN's will be done       //
// according to the currently active GRID interface.                    //
// Examples:                                                            //
//   TDSet treeset("TTree", "AOD");                                     //
//   treeset.Add("lfn:/alien.cern.ch/alice/prod2002/file1");            //
//   ...                                                                //
//   treeset.AddFriend(friendset);                                      //
//                                                                      //
// or                                                                   //
//                                                                      //
//   TDSet objset("MyEvent", "*", "/events");                           //
//   objset.Add("root://cms.cern.ch/user/prod2002/hprod_1.root");       //
//   ...                                                                //
//   objset.Add(set2003);                                               //
//                                                                      //
// Validity of file names will only be checked at processing time       //
// (typically on the PROOF master server), not at creation time.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDSet.h"
#include "TList.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "Riostream.h"


ClassImp(TDSetElement)
ClassImp(TDSet)

//______________________________________________________________________________
TDSetElement::TDSetElement(const TDSet *set, const char *file,
                           const char *objname, const char *dir,
                           Long64_t first, Long64_t num)
{
   // Create a TDSet element.

   fSet      = set;
   fFileName = file;
   fFirst    = first;
   fNum      = num;

   if (objname)
      fObjName = objname;
   if (dir)
      fDirectory = dir;
}

//______________________________________________________________________________
const char *TDSetElement::GetObjName() const
{
   // Return object name.

   if (fSet && fObjName.IsNull())
      return fSet->GetObjName();
   return fObjName;
}


//______________________________________________________________________________
const char *TDSetElement::GetDirectory() const
{
   // Return directory where to look for object.

   if (fSet && fDirectory.IsNull())
      return fSet->GetDirectory();
   return fDirectory;
}


//______________________________________________________________________________
void TDSetElement::Print(Option_t *option) const
{
   // Print a TDSetElement.

   cout << IsA()->GetName()
      << " file '" << fFileName
      << "' dir '" << fDirectory
      << "' obj '" << fObjName
      << "' first=" << fFirst << " num=" << fNum << endl;
}


//______________________________________________________________________________
TDSet::TDSet()
{
   // Default ctor.

   fElements = new TList;
   fElements->IsOwner();
   fIsTree   = kFALSE;
   fIterator = 0;
   fCurrent  = 0;
}

//______________________________________________________________________________
TDSet::TDSet(const char *type, const char *objname, const char *dir)
{
   // Create a TDSet object. The "type" defines the class of which objects
   // will be processed. The optional "objname" argument specifies the
   // name of the objects of the specified class (the name is mandatory
   // if the type inherits from a TTree). If the "objname" is not given all
   // objects of the class found in the specified directory are processed.
   // The "dir" argument specifies in which directory the objects are
   // to be found, the top level directory ("/") is the default.
   // Directories can be specified using wildcards, e.g. "*" or "/*"
   // means to look in all top level directories, "/dir/*" in all
   // directories under "/dir", and "/*/*" to look in all directories
   // two levels deep.

   fElements = new TList;
   fElements->IsOwner();
   fIterator = 0;
   fCurrent  = 0;

   if (!type || !*type) {
      Error("TDSet", "type name must be specified");
      return;
   }

   TClass *c;
   if (!(c = gROOT->GetClass(type)))
      Warning("TDSet", "type %s not yet known", type);

   fName   = type;
   fIsTree = kFALSE;

   if (c && c->InheritsFrom("TTree"))
      fIsTree = kTRUE;

   if (objname)
      fObjName = objname;

   if (dir)
      fTitle = dir;
}

//______________________________________________________________________________
TDSet::~TDSet()
{
   // Cleanup.

   delete fElements;
   delete fIterator;
}

//______________________________________________________________________________
Int_t TDSet::Process(const char *selector, Long64_t nentries,
                     Long64_t first, TEventList *evl)
{
   // Process TDSet on currently active PROOF session.
   // Returns -1 in case of error, 0 otherwise.

   if (!IsValid() || !fElements->GetSize()) {
      Error("Process", "not a correctly initialized TDSet");
      return -1;
   }

   if (TClassTable::GetDict("TProof")) {
      if (gROOT->ProcessLineFast("TProof::IsActive()")) {
         return (Int_t) gROOT->ProcessLineFast(
            Form("TProof::This()->Process((TDSet *)0x%lx, (const char*) 0x%lx, %ld, %ld, (TEventList *)0x%lx);",
            (Long_t)this, (Long_t)selector, nentries, first, (Long_t)evl));
      } else {
         Error("Process", "no active PROOF session");
         return -1;
      }
   }

   Error("Process", "PROOF session not started");
   return -1;
}

//______________________________________________________________________________
void TDSet::Print(const Option_t *option) const
{
   // Print TDSet basic or full data.

   cout <<"OBJ: " << IsA()->GetName() << "\ttype " << GetName() << "\t"
      << fObjName << "\tin " << GetTitle() << endl;

   if (option && *option) {
      TIter next(GetListOfElements());
      TObject *obj;
      while ((obj = next())) {
         obj->Print(option);
      }
   }
}

//______________________________________________________________________________
void TDSet::SetObjName(const char *objname)
{
   // Set/change object name.

   if (objname)
      fObjName = objname;
}

//______________________________________________________________________________
void TDSet::SetDirectory(const char *dir)
{
   // Set/change directory.

   if (dir)
      fTitle = dir;
}

//______________________________________________________________________________
void TDSet::Add(const char *file, const char *objname, const char *dir,
                Long64_t first, Long64_t num)
{
   // Add file to list of files to be analyzed. Optionally with the
   // objname and dir arguments the default, TDSet wide, objname and
   // dir can be overridden.

   if (!file || !*file) {
      Error("Add", "file name must be specified");
      return;
   }

   fElements->Add(new TDSetElement(this, file, objname, dir, first, num));
}

//______________________________________________________________________________
void TDSet::Add(TDSet *set)
{
   // Add specified data set to the this set.

   if (!set)
      return;

   if (set->fName != fName) {
      Error("Add", "cannot add a set with a different type");
      return;
   }

   TDSetElement *el;
   TIter next(set->fElements);
   while ((el = (TDSetElement*) next()))
      Add(el->GetFileName(), el->GetObjName(), el->GetDirectory());
}

//______________________________________________________________________________
void TDSet::AddFriend(TDSet *friendset)
{
   // Add friend dataset to this set. Only possible if the TDSet type is
   // a TTree or derived class.

   if (!friendset)
      return;

   if (!fIsTree) {
      Error("AddFriend", "a friend set can only be added to a TTree TDSet");
      return;
   }

   // to be implemented
}


//______________________________________________________________________________
void TDSet::Reset()
{
   // Reset or initialize access to the elements.

   if (!fIterator) {
      fIterator = new TIter(fElements);
   } else {
      fIterator->Reset();
   }
}

//______________________________________________________________________________
TDSetElement *TDSet::Next()
{
   // Returns next TDSetElement.

   if (!fIterator) {
      fIterator = new TIter(fElements);
   }

   fCurrent = (TDSetElement *) fIterator->Next();
   return fCurrent;
}
