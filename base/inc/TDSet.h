// @(#)root/base:$Name:  $:$Id: TDSet.h,v 1.8 2002/06/11 15:47:35 rdm Exp $
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
//   TDSet objset("MyEvent", "*", "/events");                           //
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

// typedef can be removed as soon as Long64_t becomes real 64 bit type
typedef Long_t  Long64_t;

class TList;
class TDSet;
class TEventList;


class TDSetElement : public TObject {
private:
   TString      fFileName;   // physical or logical file name
   TString      fObjName;    // name of objects to be analyzed in this file
   TString      fDirectory;  // directory in file where to look for objects
   Long64_t     fFirst;      // first entry to process
   Long64_t     fNum;        // number of entries to process
   const TDSet *fSet;        // set to which element belongs

public:
   TDSetElement() { fSet = 0; }
   TDSetElement(const TDSet *set, const char *file, const char *objname = 0,
                const char *dir = 0, Long64_t first = 0, Long64_t num = -1);
   virtual ~TDSetElement() { }

   const char *GetFileName() const { return fFileName; }
   Long64_t    GetFirst() const { return fFirst; }
   void        SetFirst(Long64_t first) { fFirst = first; }
   Long64_t    GetNum() const { return fNum ; }
   void        SetNum(Long64_t num) { fNum = num; }
   const char *GetObjName() const;
   const char *GetDirectory() const;
   void        Print(Option_t *option="") const;

   ClassDef(TDSetElement,1)  // A TDSet element
};



class TDSet : public TNamed {

private:
   TString        fObjName;   // name of objects to be analyzed (e.g. TTree name)
   TList         *fElements;  //-> list of TDSetElements
   Bool_t         fIsTree;    // true if type is a TTree (or TTree derived)
   TIter         *fIterator;  //! iterator on fElements

protected:
   TDSetElement  *fCurrent;   //! current element

public:
   TDSet();
   TDSet(const char *type, const char *objname = "*", const char *dir = "/");
   virtual ~TDSet();

   Int_t                 Process(const char *selector, Long64_t nentries = -1,
                                 Long64_t first = 0, TEventList *evl = 0);

   void                  Print(Option_t *option="") const;
   void                  SetObjName(const char *objname);
   void                  SetDirectory(const char *dir);

   const char           *GetType() const { return fName; }
   const char           *GetObjName() const { return fObjName; }
   const char           *GetDirectory() const { return fTitle; }

   virtual void          Add(const char *file, const char *objname = 0,
                             const char *dir = 0, Long64_t first = 0,
                             Long64_t num = -1);
   virtual void          Add(TDSet *set);
   virtual void          AddFriend(TDSet *friendset);

   virtual Bool_t        IsTree() const { return fIsTree; }
   virtual Bool_t        IsValid() const { return !fName.IsNull(); }
   virtual TList        *GetListOfElements() const { return fElements; }

   virtual void           Reset();
   virtual TDSetElement  *Next();
   TDSetElement          *Current() const { return fCurrent; };

   ClassDef(TDSet,1)  // Data set for remote processing (PROOF)
};


#endif
