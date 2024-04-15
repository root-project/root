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

#include "TNamed.h"


namespace ROOT {
namespace Internal {

   class TBranchProxyDescriptor : public TNamed {
      TString fDataName;
      TString fBranchName;
      bool    fIsSplit;
      bool    fBranchIsSkipped;
      bool    fIsLeafList;      // true if the branch was constructed from a leaf list.

   public:
      TBranchProxyDescriptor(const char *dataname, const char *type,
                             const char *branchname, bool split = true, bool skipped = false, bool isleaflist = false);
      const char *GetDataName();
      const char *GetTypeName();
      const char *GetBranchName();

      bool IsEquivalent(const TBranchProxyDescriptor *other, bool inClass = false);
      bool IsSplit() const;

      void OutputDecl(FILE *hf, int offset, UInt_t maxVarname);
      void OutputInit(FILE *hf, int offset, UInt_t maxVarname,
                      const char *prefix);

      ClassDefOverride(TBranchProxyDescriptor,0); // Describe the proxy for a branch
   };
}
}

#endif
