// @(#)root/base:$Name:  $:$Id: TDSet.h,v 1.1 2002/01/18 14:24:09 rdm Exp $
// Author: Fons Rademakers   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDSet
#define ROOT_TDSet


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
//   TDSet objset("MyEvent", "", "/events");                            //
//   objset.Add("root://cms.cern.ch/user/prod2002/hprod_1.root");       //
//   ...                                                                //
//   objset.Add(set2003);                                               //
//                                                                      //
// Validity of file names will only be checked at processing time       //
// (typically on the PROOF master server), not at creation time.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TList;


class TDSetElement : public TObject {
private:
   TString   fFileName;   // physical or logical file name
   TString   fObjName;    // name of objects to be analyzed in this file
   TString   fDirectory;  // directory in file where to look for objects

public:
   TDSetElement() { }
   TDSetElement(const char *file, const char *objname = 0,
                const char *dir = 0);
   virtual ~TDSetElement() { }

   const char *GetFileName() const { return fFileName; }
   const char *GetObjName() const { return fObjName; }
   const char *GetDirectory() const { return fDirectory; }

   ClassDef(TDSetElement,1)  // A TDSet element
};



class TDSet : public TNamed {

private:
   TString  fObjName;     // name of objects to be analyzed (e.g. TTree name)
   TList   *fElements;    //-> list of TDSetElements
   Bool_t   fIsTree;      // true if type is a TTree (or TTree derived)

public:
   TDSet();
   TDSet(const char *type, const char *objname = "", const char *dir = "/");
   virtual ~TDSet();

   void        SetObjName(const char *objname);
   void        SetDirectory(const char *dir);

   const char *GetType() const { return fName; }
   const char *GetObjName() const { return fObjName; }
   const char *GetDirectory() const { return fTitle; }

   void        Add(const char *file, const char *objname = 0,
                   const char *dir = 0);
   void        Add(TDSet *set);
   void        AddFriend(TDSet *friendset);

   Bool_t      IsTree() const { return fIsTree; }
   Bool_t      IsValid() const { return !fName.IsNull(); }
   TList      *GetElementList() const { return fElements; }

   ClassDef(TDSet,1)  // Data set for remote processing (PROOF)
};

#endif
