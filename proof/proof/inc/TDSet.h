// @(#)root/proof:$Id$
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

class TChain;
class TCollection;
class TCut;
class TDSet;
class TEventList;
class TEntryList;
class TFileInfo;
class THashList;
class TIter;
class TList;
class TProof;
class TProofChain;
class TTree;

// For backward compatibility (handle correctly requests from old clients)
#include <set>
#include <list>

class TDSetElement : public TNamed {
public:
   typedef  std::list<std::pair<TDSetElement*, TString> > FriendsList_t;
   // TDSetElement status bits
   enum EStatusBits {
      kHasBeenLookedUp = BIT(15),
      kWriteV3         = BIT(16),
      kEmpty           = BIT(17),
      kCorrupted       = BIT(18),
      kNewRun          = BIT(19),
      kNewPacket       = BIT(20)
   };

private:
   TString          fDirectory;  // directory in file where to look for objects
   Long64_t         fFirst;      // first entry to process
   Long64_t         fNum;        // number of entries to process
   TString          fMsd;        // mass storage domain name
   Long64_t         fTDSetOffset;// the global offset in the TDSet of the first
                                 // entry in this element
   TObject         *fEntryList;  // entry (or event) list to be used in processing
   Bool_t           fValid;      // whether or not the input values are valid
   Long64_t         fEntries;    // total number of possible entries in file
   TList           *fFriends;    // friend elements

   TString          fDataSet;    // Name of the dataset of which this element is part
   TList           *fAssocObjList;  // List of objects associated to this element
                                   // (e.g. TObjString describing associated files)

   Bool_t           HasBeenLookedUp() const { return TestBit(kHasBeenLookedUp); }

   TDSetElement& operator=(const TDSetElement &); // Not implemented

public:
   TDSetElement();
   TDSetElement(const char *file, const char *objname = 0,
                const char *dir = 0, Long64_t first = 0, Long64_t num = -1,
                const char *msd = 0, const char *dataset = 0);
   TDSetElement(const TDSetElement& elem);
   virtual ~TDSetElement();

   virtual TList   *GetListOfFriends() const { return fFriends; }
   virtual void     AddFriend(TDSetElement *friendElement, const char *alias);
   virtual void     DeleteFriends();
   const char      *GetFileName() const { return GetName(); }
   Long64_t         GetFirst() const { return fFirst; }
   void             SetFirst(Long64_t first) { fFirst = first; }
   Long64_t         GetNum() const { return fNum; }
   Long64_t         GetEntries(Bool_t istree = kTRUE, Bool_t openfile = kTRUE);
   void             SetEntries(Long64_t ent) { fEntries = ent; }
   const char      *GetMsd() const { return fMsd; }
   void             SetNum(Long64_t num) { fNum = num; }
   Bool_t           GetValid() const { return fValid; }
   const char      *GetObjName() const { return GetTitle(); }
   const char      *GetDirectory() const;
   const char      *GetDataSet() const { return fDataSet; }
   void             SetDataSet(const char *dataset) { fDataSet = dataset; }
   void             AddAssocObj(TObject *assocobj);
   TList           *GetListOfAssocObjs() const { return fAssocObjList; }
   TObject         *GetAssocObj(Long64_t i, Bool_t isentry = kFALSE);
   void             Print(Option_t *options="") const;
   Long64_t         GetTDSetOffset() const { return fTDSetOffset; }
   void             SetTDSetOffset(Long64_t offset) { fTDSetOffset = offset; }
   void             SetEntryList(TObject *aList, Long64_t first = -1, Long64_t num = -1);
   TObject         *GetEntryList() const { return fEntryList; }
   void             Validate(Bool_t isTree);
   void             Validate(TDSetElement *elem);
   void             Invalidate() { fValid = kFALSE; }
   void             SetValid() { fValid = kTRUE; }
   Int_t            Compare(const TObject *obj) const;
   Bool_t           IsSortable() const { return kTRUE; }
   Int_t            Lookup(Bool_t force = kFALSE);
   void             SetLookedUp() { SetBit(kHasBeenLookedUp); }
   TFileInfo       *GetFileInfo(const char *type = "TTree");

   Int_t            MergeElement(TDSetElement *elem);

   ClassDef(TDSetElement,8)  // A TDSet element
};


class TDSet : public TNamed {

public:
   // TDSet status bits
   enum EStatusBits {
      kWriteV3         = BIT(16),
      kEmpty           = BIT(17),
      kValidityChecked = BIT(18),  // Set if elements validiy has been checked
      kSomeInvalid     = BIT(19),  // Set if at least one element is invalid
      kMultiDSet       = BIT(20)   // Set if fElements is a list of datasets
   };

private:
   Bool_t         fIsTree;      // true if type is a TTree (or TTree derived)
   TObject       *fEntryList;   //! entry (or event) list for processing
   TProofChain   *fProofChain;  //! for browsing purposes

   void           SplitEntryList(); //Split entry list between elements

   TDSet(const TDSet &);           // not implemented
   void operator=(const TDSet &);  // not implemented

protected:
   TString        fDir;         // name of the directory
   TString        fType;        // type of objects (e.g. TTree);
   TString        fObjName;     // name of objects to be analyzed (e.g. TTree name)
   THashList     *fElements;    //-> list of TDSetElements (or TDSets, if in multi mode)
   TIter         *fIterator;    //! iterator on fElements
   TDSetElement  *fCurrent;     //! current element
   TList         *fSrvMaps;     //! list for mapping server coordinates for files
   TIter         *fSrvMapsIter; //! iterator on fSrvMaps

public:
   TDSet();
   TDSet(const char *name, const char *objname = "*",
         const char *dir = "/", const char *type = 0);
   TDSet(const TChain &chain, Bool_t withfriends = kTRUE);
   virtual ~TDSet();

   virtual Bool_t        Add(const char *file, const char *objname = 0,
                             const char *dir = 0, Long64_t first = 0,
                             Long64_t num = -1, const char *msd = 0);
   virtual Bool_t        Add(TDSet *set);
   virtual Bool_t        Add(TCollection *fileinfo, const char *meta = 0,
                             Bool_t availableOnly = kFALSE, TCollection *badlist = 0);
   virtual Bool_t        Add(TFileInfo *fileinfo, const char *meta = 0);
   virtual void          AddFriend(TDSet *friendset, const char *alias);

   virtual Long64_t      Process(const char *selector, Option_t *option = "",
                                 Long64_t nentries = -1,
                                 Long64_t firstentry = 0,
                                 TObject *enl = 0); // *MENU*
   virtual Long64_t      Draw(const char *varexp, const char *selection,
                              Option_t *option = "", Long64_t nentries = -1,
                              Long64_t firstentry = 0); // *MENU*
   virtual Long64_t      Draw(const char *varexp, const TCut &selection,
                              Option_t *option = "", Long64_t nentries = -1,
                              Long64_t firstentry = 0); // *MENU*
   virtual void          Draw(Option_t *opt) { Draw(opt, "", "", 1000000000, 0); }

   Int_t                 ExportFileList(const char *filepath, Option_t *opt = "");

   void                  Print(Option_t *option="") const;

   void                  SetObjName(const char *objname);
   void                  SetDirectory(const char *dir);

   Bool_t                IsTree() const { return fIsTree; }
   Bool_t                IsValid() const { return !fType.IsNull(); }
   Bool_t                ElementsValid();
   const char           *GetType() const { return fType; }
   const char           *GetObjName() const { return fObjName; }
   const char           *GetDirectory() const { return fDir; }
   TList                *GetListOfElements() const { return (TList *)fElements; }
   Int_t                 GetNumOfFiles();

   Int_t                 Remove(TDSetElement *elem, Bool_t deleteElem = kTRUE);

   virtual void          Reset();
   virtual TDSetElement *Next(Long64_t totalEntries = -1);
   TDSetElement         *Current() const { return fCurrent; };

   static Long64_t       GetEntries(Bool_t isTree, const char *filename,
                                    const char *path, TString &objname);

   void                  AddInput(TObject *obj);
   void                  ClearInput();
   TObject              *GetOutput(const char *name);
   TList                *GetOutputList();
   virtual void          StartViewer(); // *MENU*

   virtual TTree        *GetTreeHeader(TProof *proof);
   virtual void          SetEntryList(TObject *aList);
   TObject              *GetEntryList() const { return fEntryList; }
   void                  Validate();
   void                  Validate(TDSet *dset);

   void                  Lookup(Bool_t removeMissing = kFALSE, TList **missingFiles = 0);
   void                  SetLookedUp();

   void                  SetSrvMaps(TList *srvmaps = 0);

   void                  SetWriteV3(Bool_t on = kTRUE);

   ClassDef(TDSet,8)  // Data set for remote processing (PROOF)
};

#endif
