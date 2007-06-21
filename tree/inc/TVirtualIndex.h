// @(#)root/tree:$Name:  $:$Id: TVirtualIndex.h,v 1.4 2005/11/11 22:16:04 pcanal Exp $
// Author: Rene Brun   05/07/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualIndex
#define ROOT_TVirtualIndex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualIndex                                                        //
//                                                                      //
// Abstract interface for Tree Index                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TTree;

class TVirtualIndex : public TNamed {

protected:
   TTree         *fTree;          //! pointer to Tree
  
public:
   TVirtualIndex();
   virtual               ~TVirtualIndex();
   virtual void           Append(const TVirtualIndex *,Bool_t delaySort = kFALSE) = 0;
   virtual Int_t          GetEntryNumberFriend(const TTree * /*T*/) = 0;
   virtual Long64_t       GetEntryNumberWithIndex(Int_t major, Int_t minor) const = 0;
   virtual Long64_t       GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const = 0;
   virtual const char    *GetMajorName()    const = 0;
   virtual const char    *GetMinorName()    const = 0;
   virtual Long64_t       GetN()            const = 0;
   virtual TTree         *GetTree()         const {return fTree;}
   virtual void           UpdateFormulaLeaves() = 0;
   virtual void           SetTree(const TTree *T) = 0;
   
   ClassDef(TVirtualIndex,1);  //Abstract interface for Tree Index
};

#endif

