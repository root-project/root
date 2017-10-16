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

#include "TBranch.h"

namespace ROOT {
namespace Experimental {
namespace Internal {

class TBulkBranchRead {

   friend class ::TBranch;

public:
   Int_t  GetEntriesFast(Long64_t evt, TBuffer& user_buf, bool checkDeserializeType=true) {return fParent.GetEntriesFast(evt, user_buf, checkDeserializeType);}
   Int_t  GetEntriesSerialized(Long64_t evt, TBuffer& user_buf, bool checkDeserializeType=true) {return fParent.GetEntriesSerialized(evt, user_buf, checkDeserializeType);}
   Int_t  GetEntriesSerialized(Long64_t evt, TBuffer& user_buf, TBuffer* count_buf, bool checkDeserializeType=true) {return fParent.GetEntriesSerialized(evt, user_buf, count_buf, checkDeserializeType);}
   Bool_t SupportsBulkRead() const {return fParent.SupportsBulkRead();}

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
