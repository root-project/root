// @(#)root/hbook:$Name:  $:$Id: THbookBranch.h,v 1.2 2002/02/20 16:57:30 brun Exp $
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THbookBranch
#define ROOT_THbookBranch


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookBranch                                                         //
//                                                                      //
// A branch for a THbookTree                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

class THbookBranch : public TBranch {

protected:
   TString      fBlockName;   //Hbook block name

public:
   THbookBranch() {;}
   THbookBranch(const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress=-1);
   virtual ~THbookBranch();
   virtual void     Browse(TBrowser *b);
   virtual Int_t    GetEntry(Int_t entry=0, Int_t getall=0);
        const char *GetBlockName() const {return fBlockName.Data();}
   virtual void     SetAddress(void *addobj);
           void     SetBlockName(const char *name) {fBlockName=name;}
   virtual void     SetEntries(Int_t n) {fEntries=n;}

   ClassDef(THbookBranch,1)  //A branch for a THbookTree
};

#endif
