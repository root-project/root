// @(#)root/tree:$Name:  $:$Id: Exp $
// Author: Philippe Canal 07/11/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeCloner
#define ROOT_TTreeCloner

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCloner                                                          //
//                                                                      //
// Class implementing or helping  the various TTree cloning method      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TBranch;
#include <vector>

#ifdef R__OLDHPACC
namespace std {
   using ::string;
   using ::vector;
}
#endif

class TTreeCloner {
   TTree     *fFromTree;
   TTree     *fToTree;
   Option_t  *fMethod;
   TObjArray  fFromBranches;
   TObjArray  fToBranches;

   UInt_t     fMaxBaskets;
   UInt_t    *fBasketBranchNum;  //[fMaxBaskets] Index of the branch(es) of the basket
   UInt_t    *fBasketNum;        //[fMaxBaskets] index of the basket within the branch

   Long64_t  *fBasketSeek;       //[fMaxBaskets] list of basket position to be read
   Int_t     *fBasketIndex;      //[fMaxBaskets] ordered list of basket indices to be written

   UShort_t   fPidOffset;        //Offset to be added to the copied key/basket

   UInt_t     fCloneMethod;      //Indicates which cloning method was selected

   enum ECloneMethod {
      kDefault            = 0,
      kSortBasketByBranch = 1,
      kSortBasketByOffset = 2
   };
   
public:
   TTreeCloner(TTree *from, TTree *to, Option_t *method);
   virtual ~TTreeCloner();

   void   CloseOutWriteBaskets();
   UInt_t CollectBranches(TBranch *from, TBranch *to);
   UInt_t CollectBranches(TObjArray *from, TObjArray *to);
   UInt_t CollectBranches();
   void   CollectBaskets();
   void   CopyMemoryBaskets();
   void   CopyStreamerInfos();
   void   CopyProcessIds();
   Bool_t Exec();
   void   SortBaskets();
   void   WriteBaskets();

   ClassDef(TTreeCloner,0); 
};

#endif
