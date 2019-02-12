// Author: Brian Bockelman 14/06/17

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBulkBranchRead
#define ROOT_TBulkBranchRead

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBulkBranchRead                                                      //
//                                                                      //
// TBulkBranchRead is used to check if a branch supports bulk API;      //
// it also includes a set of API that allows reading serialized data    //
// directly into user-defined buffer without going through ROOT-        //
// deserialization code path by default.                                //
//////////////////////////////////////////////////////////////////////////

#include "TBranch.h"

namespace ROOT {
namespace Experimental {
namespace Internal {

class TBulkBranchRead {

   friend class ::TBranch;

public:
   Int_t  GetEntriesFast(Long64_t evt, TBuffer& user_buf) { return fParent.GetEntriesFast(evt, user_buf); }
   Int_t  GetEntriesSerialized(Long64_t evt, TBuffer& user_buf) { return fParent.GetEntriesSerialized(evt, user_buf); }
   Int_t  GetEntriesSerialized(Long64_t evt, TBuffer& user_buf, TBuffer* count_buf) { return fParent.GetEntriesSerialized(evt, user_buf, count_buf); }
   Bool_t SupportsBulkRead() const { return fParent.SupportsBulkRead(); }

private:
   TBulkBranchRead(TBranch &parent)
      : fParent(parent)
   {}

   TBranch &fParent;
};

}  // Internal
}  // Experimental
}  // ROOT

#endif  // ROOT_TBulkBranchRead
