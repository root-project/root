// @(#)root/tree:$Id$
// Author: Philippe Canal and al. 08/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBasketSQL                                                           //
//                                                                      //
// Implement TBasket for a SQL backend                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef TBASKETSQL_CXX
#define TBASKETSQL_CXX

#include "TBasket.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "TMath.h"
#include "TBasketSQL.h"
#include <Riostream.h>
#include <vector>
#include "TTreeSQL.h"
#include "TBufferSQL.h"

ClassImp(TBasketSQL)

namespace std {} using namespace std;

//_________________________________________________________________________
TBasketSQL::TBasketSQL() : TBasket()
{
  // Default constructor
}

//_________________________________________________________________________
TBasketSQL::TBasketSQL(const char *name, const char *title, TBranch *branch, 
                         TSQLResult ** rs, TString *insert_query, 
                         vector<Int_t> *vc, TSQLRow **r) :
  fResultPtr(rs),fRowPtr(r)
{ 
   // Regular constructor

   SetName(name);
   SetTitle(title);
   fClassName   = "TBasketSQL";
   fBufferSize  = branch->GetBasketSize();
   fNevBufSize  = branch->GetEntryOffsetLen();
   fNevBuf      = 0;
   fEntryOffset = 0;  //Must be set to 0 before calling Sizeof
   fDisplacement= 0;  //Must be set to 0 before calling Sizeof
   fBuffer      = 0;  //Must be set to 0 before calling Sizeof
   fInsertQuery = insert_query;
   
   if (vc==0) {
      fBufferRef = 0;
   } else {
      fBufferRef = new TBufferSQL(TBuffer::kWrite, fBufferSize, vc, fInsertQuery, fRowPtr);
   }
   fHeaderOnly  = kTRUE;
   fLast        = 0; // Must initialize before calling Streamer()
   //Streamer(*fBufferRef);
   fBuffer      = 0;
   fBranch      = branch;
   fHeaderOnly  = kFALSE;
   branch->GetTree()->IncrementTotalBuffers(fBufferSize);
}

//_________________________________________________________________________
TBasketSQL::~TBasketSQL()
{
   // Destructor
}

//_________________________________________________________________________
void TBasketSQL::CreateBuffer(const char *name, TString title, 
                              vector<Int_t> *vc, 
                              TBranch *branch, TSQLResult ** rs)
{
   // Create a TSQLBuffer for this basket.

   fResultPtr = rs;
   SetName(name);
   SetTitle(title);
   fClassName   = "TBasketSQL";
   fBufferSize  = branch->GetBasketSize();
   fNevBufSize  = branch->GetEntryOffsetLen();
   fNevBuf      = 0;
   fEntryOffset = 0;  //Must be set to 0 before calling Sizeof
   fDisplacement= 0;  //Must be set to 0 before calling Sizeof
   fBuffer      = 0;  //Must be set to 0 before calling Sizeof

   if(vc==0) {
      fBufferRef = 0;
      Error("CreateBuffer","Need a vector of columns\n");
   } else {    
      fBufferRef   = new TBufferSQL(TBuffer::kWrite, fBufferSize, vc, fInsertQuery, fRowPtr);
   }
   fHeaderOnly  = kTRUE;
   fLast        = 0; 
   //Streamer(*fBufferRef);
   fBuffer      = 0;
   fBranch      = branch;
   fHeaderOnly  = kFALSE;
   branch->GetTree()->IncrementTotalBuffers(fBufferSize);
}

//_________________________________________________________________________
void TBasketSQL::PrepareBasket(Long64_t entry)
{
   // Prepare the basket for the next entry.

   ((TBufferSQL*)fBufferRef)->ResetOffset();
   ((TTreeSQL*)fBranch->GetTree())->PrepEntry(entry);
   fBufferRef->Reset();
}

//_________________________________________________________________________
Int_t TBasketSQL::ReadBasketBytes(Long64_t , TFile *)
{
   // See TBasket::ReadBasketBytes.  This is not implemented in TBasketSQL.

   Error("ReadBasketBytes","This member function should not be called!");
   return 0;
}

//_________________________________________________________________________
Int_t TBasketSQL::ReadBasketBuffers(Long64_t , Int_t, TFile *)
{	 
   // See TBasket::ReadBasketBuffers.  This is not implemented in TBasketSQL.

   Error("ReadBasketBuffers","This member function should not be called!");
   return 0;
}

//_________________________________________________________________________
void TBasketSQL::Reset()
{	 
   // See TBasket::Reset

   TBasket::Reset();
}


//_________________________________________________________________________
void TBasketSQL::Update(Int_t, Int_t) 
{
   // See TBasket::Update.

   ((TBufferSQL*)fBufferRef)->ResetOffset();
   fNevBuf++;
}


#endif
