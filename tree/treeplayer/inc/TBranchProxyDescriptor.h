// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxyDescriptor
#define ROOT_TBranchProxyDescriptor

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif


namespace ROOT {

   class TBranchProxyDescriptor : public TNamed {
      TString fDataName;
      TString fBranchName;
      Bool_t  fIsSplit;
      Bool_t  fBranchIsSkipped;
      Bool_t  fIsLeafList;      // true if the branch was constructed from a leaf list.

   public:
      TBranchProxyDescriptor(const char *dataname, const char *type,
                             const char *branchname, Bool_t split = true, Bool_t skipped = false, Bool_t isleaflist = false);
      const char *GetDataName();
      const char *GetTypeName();
      const char *GetBranchName();

      Bool_t IsEquivalent(const TBranchProxyDescriptor *other, Bool_t inClass = kFALSE);
      Bool_t IsSplit() const;

      void OutputDecl(FILE *hf, int offset, UInt_t maxVarname);
      void OutputInit(FILE *hf, int offset, UInt_t maxVarname,
                      const char *prefix);

      ClassDef(TBranchProxyDescriptor,0); // Describe the proxy for a branch
   };
}

#endif
