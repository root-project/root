// @(#)root/treeplayer:$Name:  $:$Id: TFormLeafInfo.h,v 1.1 2004/06/17 17:37:10 brun Exp $
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TBRANCHPROXYDESCRIPTOR_H
#define TBRANCHPROXYDESCRIPTOR_H

#include "TNamed.h"

namespace ROOT {

   class TBranchProxyDescriptor : public TNamed {
      TString fBranchName;
      bool fIsSplit;
      
   public:
      TBranchProxyDescriptor(const char *dataname, const char *type, 
                             const char *branchname, bool split = true);
      const char *GetDataName();
      const char *GetTypeName();
      const char *GetBranchName();

#if !defined(_MSC_VER) || (_MSC_VER>1300)
      using TObject::IsEqual;
#else
      using TNamed::IsEqual;
#endif
      bool IsEqual(const TBranchProxyDescriptor *other);
      bool IsSplit() const;

      void OutputDecl(FILE *hf, int offset, UInt_t maxVarname);
      void OutputInit(FILE *hf, int offset, UInt_t maxVarname,
                      const char *prefix);

      ClassDef(TBranchProxyDescriptor,0);
   };
}

#endif
