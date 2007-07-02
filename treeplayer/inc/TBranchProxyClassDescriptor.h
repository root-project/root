// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxyClassDescriptor.h,v 1.8 2007/02/05 18:11:29 brun Exp $
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxyClassDescriptor
#define ROOT_TBranchProxyClassDescriptor

#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TTree;
class TStreamerInfo;

namespace ROOT {

   class TBranchProxyDescriptor;

   class TBranchProxyClassDescriptor : public TNamed {

   public:
      typedef enum { kOut=0, kClones, kInsideClones } EInClones; // for IsClones
   private:
      TList   fListOfSubProxies;
      TList   fListOfBaseProxies;
      UInt_t  fIsClones;   // 1 for the general case, 2 when this a split clases inside a TClonesArray.
      Bool_t  fIsLeafList; // true if the branch was constructed from a leaf list.

      UInt_t  fSplitLevel;

      TString        fRawSymbol;
      TString        fBranchName;
      TString        fSubBranchPrefix;
      TStreamerInfo *fInfo;     // TStreamerInfo describing this class

      UInt_t  fMaxDatamemberType;

      void NameToSymbol();

   public:

      TBranchProxyClassDescriptor(const char *type, TStreamerInfo *info, const char *branchname,
                                  UInt_t isclones, UInt_t splitlevel);
      TBranchProxyClassDescriptor(const char *branchname);

      TBranchProxyClassDescriptor(const char *type, TStreamerInfo *info, const char *branchname,
                                  const char *branchPrefix, UInt_t isclones,
                                  UInt_t splitlevel);
      TBranchProxyClassDescriptor(const TBranchProxyClassDescriptor &b) :TNamed(b){;}
      TBranchProxyClassDescriptor& operator=(const TBranchProxyClassDescriptor&) {return *this;}

      const char* GetBranchName() const;
      const char* GetSubBranchPrefix() const;

      const char* GetRawSymbol() const;
      
      TStreamerInfo *GetInfo() const { return fInfo; }

      UInt_t GetSplitLevel() const;

      virtual Bool_t IsEquivalent(const TBranchProxyClassDescriptor* other);

      void AddDescriptor(TBranchProxyDescriptor *desc, Bool_t isBase);
      Bool_t IsLoaded() const;
      Bool_t IsClones() const;
      UInt_t GetIsClones() const;

      void OutputDecl(FILE *hf, int offset, UInt_t /* maxVarname */);

      ClassDef(TBranchProxyClassDescriptor,0); // Class to cache the information we gathered about the branch and its content
   };

}

#endif
