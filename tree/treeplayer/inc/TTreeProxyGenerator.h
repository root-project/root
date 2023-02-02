// @(#)root/treeplayer:$Id$
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

#include "TTreeGeneratorBase.h"

class TBranch;
class TBranchElement;
class TLeaf;
class TStreamerElement;

namespace ROOT {
namespace Internal {
   class TFriendProxyDescriptor;
   class TBranchProxyDescriptor;
   class TBranchProxyClassDescriptor;

   class TTreeProxyGenerator : public TTreeGeneratorBase
   {
   public:
      enum EContainer { kNone, kClones, kSTL };
      enum EOption { kNoOption, kNoHist };
      UInt_t   fMaxDatamemberType;
      TString  fScript;
      TString  fCutScript;
      TString  fPrefix;
      TString  fHeaderFileName;
      UInt_t   fOptions;
      UInt_t   fMaxUnrolling;
      TList    fListOfClasses;
      TList    fListOfFriends;
      TList    fListOfPragmas;
      TList    fListOfTopProxies;
      TList   *fCurrentListOfTopProxies; //!
      TList    fListOfForwards;
      TTreeProxyGenerator(TTree* tree, const char *script, const char *fileprefix,
                          const char *option, UInt_t maxUnrolling);
      TTreeProxyGenerator(TTree* tree, const char *script, const char *cutscript,
                          const char *fileprefix, const char *option, UInt_t maxUnrolling);

      TBranchProxyClassDescriptor* AddClass(TBranchProxyClassDescriptor *desc);
      void AddDescriptor(TBranchProxyDescriptor *desc);
      void AddForward(TClass *cl);
      void AddForward(const char *classname);
      void AddFriend(TFriendProxyDescriptor *desc);
      void AddMissingClassAsEnum(const char *clname, Bool_t isscope);
      void AddPragma(const char *pragma_text);
      void CheckForMissingClass(const char *clname);

      Bool_t NeedToEmulate(TClass *cl, UInt_t level);

      void   ParseOptions();

      UInt_t AnalyzeBranches(UInt_t level, TBranchProxyClassDescriptor *topdesc, TBranchElement *branch, TVirtualStreamerInfo *info = nullptr);
      UInt_t AnalyzeBranches(UInt_t level, TBranchProxyClassDescriptor *topdesc, TIter &branches, TVirtualStreamerInfo *info);
      UInt_t AnalyzeOldBranch(TBranch *branch, UInt_t level, TBranchProxyClassDescriptor *desc);
      UInt_t AnalyzeOldLeaf(TLeaf *leaf, UInt_t level, TBranchProxyClassDescriptor *topdesc);
      void   AnalyzeElement(TBranch *branch, TStreamerElement *element, UInt_t level, TBranchProxyClassDescriptor *desc, const char* path);
      void   AnalyzeTree(TTree *tree);
      void   WriteProxy();

      const char *GetFileName() { return fHeaderFileName; }
   };

}
}

#endif
