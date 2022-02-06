// @(#)root/hbook:$Id$
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THbookTree
#define ROOT_THbookTree


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookTree                                                           //
//                                                                      //
// A wrapper class supporting Hbook ntuples (CWN and RWN).              //
// The normal TTree calls can be used, including TTree::Draw().         //
// Data read directly from the Hbook file via THbookFile.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"
#include "THbookFile.h"


class THbookTree : public TTree {

protected:
   Int_t       fID;         //Hbook identifier
   Int_t       fType;       //RWN (0) or CWN (1)
   char       *fX;          //storage area for RWN
   Bool_t      fInit;       //flag to know if branches computed
   THbookFile *fFile;       //pointer to Hbook file

public:
   THbookTree();
   THbookTree(const char *name, Int_t id);
   ~THbookTree() override;
   Int_t     GetEntry(Long64_t entry=0, Int_t getall=0) override;
   THbookFile       *GetHbookFile() {return fFile;}
   virtual Int_t     GetID() {return fID;}
   virtual Int_t     GetType() {return fType;}
           Float_t  *GetX() {return (Float_t*)fX;}
   virtual void      InitBranches(Long64_t entry);
           char     *MakeX(Int_t nvars) {fX = new char[nvars]; return fX;}
   void      Print(Option_t *option="") const override;
   Long64_t  SetEntries(Long64_t n=-1) override;
   virtual void      SetHbookFile(THbookFile *file) {fFile = file;}
   virtual void      SetType(Int_t atype) {fType = atype;}

   ClassDefOverride(THbookTree,1)  //A wrapper class supporting Hbook ntuples (CWN and RWN)
};

#endif
