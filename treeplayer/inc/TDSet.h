// @(#)root/treeplayer:$Name:$:$Id:$
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
//   TDSet treeset("TTree:AOD");                                        //
//   treeset.Add("lfn:/alien.cern.ch/alice/prod2002/file1");            //
//   ...                                                                //
//   treeset.AddFriend(friendset);                                      //
//                                                                      //
// or                                                                   //
//                                                                      //
//   TDSet objset("MyEvent", "/events");                                //
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


class TDSet : public TNamed {

private:
   TString  fObjName;
   TList   *fFiles;
   Bool_t   fIsTree;

public:
   TDSet();
   TDSet(const char *type, const char *dir = "/");
   virtual ~TDSet();

   void        SetType(const char *type);
   void        SetObjName(const char *objname);
   void        SetDirectory(const char *dir) { SetTitle(dir); }

   const char *GetType() const { return GetName(); }
   const char *GetObjName() const { return fObjName; }
   const char *GetDirectory() const { return GetTitle(); }

   void        Add(const char *file, const char *namedir);
   void        Add(TDSet *set);
   void        AddFriend(TDSet *friendset);

   Bool_t      IsTree() const { return fIsTree; }
   TList      *GetFileList() const { return fFiles; }

   ClassDef(TDSet,1)  // Data set for remote processing (PROOF)
};

#endif
