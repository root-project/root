// @(#)root/treeplayer:$Name:  $:$Id: TSelector.h,v 1.2 2000/07/10 06:12:15 brun Exp $
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
   virtual            ~TSelector();
   virtual void        Begin(TTree *) {;}
   virtual void        ExecuteBegin(TTree *tree);
   virtual Bool_t      ExecuteProcessCut(Int_t entry);
   virtual void        ExecuteProcessFill(Int_t entry);
   virtual void        ExecuteTerminate();
   virtual void        Terminate() {;}
   static  TSelector  *GetSelector(const char *filename);
   virtual Bool_t      ProcessCut(Int_t entry) {return kTRUE;}
   virtual void        ProcessFill(Int_t entry) {;}

   ClassDef(TSelector,0)  //A utility class for Trees selections.
};

#endif

