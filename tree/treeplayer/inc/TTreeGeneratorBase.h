// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 13/08/2015

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeGeneratorBase
#define ROOT_TTreeGeneratorBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeGeneratorBase                                                   //
//                                                                      //
// Base class for code generators like TTreeProxyGenerator and          //
// TTreeReaderGenerator                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TString.h"

class TBranch;
class TBranchElement;
class TClass;
class TStreamerElement;
class TTree;
class TVirtualStreamerInfo;

namespace ROOT {
namespace Internal {
   class TTreeGeneratorBase {
      public:
         TList    fListOfHeaders;     ///< List of included headers
         TTree   *fTree;              ///< Pointer to the tree
         TString  fOptionStr;         ///< User options as a string

         TTreeGeneratorBase(TTree *tree, const char *option);

         void    AddHeader(TClass *cl);
         void    AddHeader(const char *classname);
         TString GetContainedClassName(TBranchElement *branch, TStreamerElement *element, Bool_t ispointer);
         TVirtualStreamerInfo *GetBaseClass(TStreamerElement *element);
         TVirtualStreamerInfo *GetStreamerInfo(TBranch *branch, TIter current, TClass *cl);
   };
}
}

#endif
