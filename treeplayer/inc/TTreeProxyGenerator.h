// @(#)root/treeplayer:$Name:  $:$Id: TTreeProxyGenerator.h,v 1.1 2004/06/25 18:42:19 brun Exp $
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeProxyGenerator
#define ROOT_TTreeProxyGenerator

#ifndef ROOT_Tlist
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TBranch;
class TLeaf;
class TTree;
class TStreamerElement;

namespace ROOT {
   class TBranchProxy;
   class TBranchProxyDescriptor;
   class TBranchProxyClassDescriptor;

   class TTreeProxyGenerator
   {
   public:
      enum EContainer { kNone, kClones };
      UInt_t   fMaxDatamemberType;
      TString  fScript;
      TString  fCutScript;
      TString  fPrefix;
      TString  fHeaderFilename;
      UInt_t   fMaxUnrolling;
      TTree   *fTree;
      TList    fListOfHeaders;
      TList    fListOfClasses;
      TList    fListOfTopProxies;
      TList    fListOfForwards;
      TTreeProxyGenerator(TTree* tree, const char *script, const char *fileprefix, UInt_t maxUnrolling);
      TTreeProxyGenerator(TTree* tree, const char *script, const char *cutscript, const char *fileprefix, UInt_t maxUnrolling);

      TBranchProxyClassDescriptor* AddClass(TBranchProxyClassDescriptor *desc);
      void AddForward(TClass *cl);
      void AddForward(const char *classname);
      void AddHeader(TClass *cl);
      void AddHeader(const char *classname);
      void AddDescriptor(TBranchProxyDescriptor *desc);

      bool NeedToEmulate(TClass *cl, UInt_t level);

      UInt_t AnalyzeBranch(TBranch *branch, UInt_t level, TBranchProxyClassDescriptor *desc);
      UInt_t AnalyzeOldBranch(TBranch *branch, UInt_t level, TBranchProxyClassDescriptor *desc);
      UInt_t AnalyzeOldLeaf(TLeaf *leaf, UInt_t level, TBranchProxyClassDescriptor *topdesc);
      void   AnalyzeElement(TBranch *branch, TStreamerElement *element, UInt_t level, TBranchProxyClassDescriptor *desc, const char* path);
      void   AnalyzeTree();
      void   WriteProxy();

      const char *GetFilename() { return fHeaderFilename; }
   };

}

using ROOT::TTreeProxyGenerator;

#endif
