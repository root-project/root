// @(#)root/treeplayer:$Name:  $:$Id: TSelector.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelector
#define ROOT_TSelector


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelector                                                            //
//                                                                      //
// A utility class for Trees selections.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TTree;

class TSelector : public TObject {

public:
   TSelector();
   virtual          ~TSelector() {;}
   virtual void     Analyze() = 0;
   virtual void     Begin() = 0;
   virtual void     Finish() = 0;
   virtual Bool_t   Select(Int_t entry) = 0;

   ClassDef(TSelector,0)  //A utility class for Trees selections.
};

#endif

