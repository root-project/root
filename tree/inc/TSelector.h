// @(#)root/treeplayer:$Name:  $:$Id: TSelector.h,v 1.1 2000/07/06 16:53:36 brun Exp $
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

class G__CallFunc;
class G__ClassInfo;

class TSelector : public TObject {

protected:
   G__CallFunc   *fFuncBegin;    //!
   G__CallFunc   *fFuncFinish;   //!
   G__CallFunc   *fFuncSelect;   //!
   G__CallFunc   *fFuncAnal;     //!
   TSelector     *fIntSelector;  //Pointer to interpreted selector (if interpreted)
   Bool_t         fIsCompiled;   //true if selector has been compiled
   
public:
   TSelector();
   virtual            ~TSelector();
   virtual void        Analyze(Int_t entry) {;}
   virtual void        Begin() {;}
   virtual void        Build(TSelector *iselector, G__ClassInfo *cl);
   virtual void        ExecuteAnalyze(Int_t entry);
   virtual void        ExecuteBegin();
   virtual void        ExecuteFinish();
   virtual Bool_t      ExecuteSelect(Int_t entry);
   virtual void        Finish() {;}
   static  TSelector  *GetSelector(const char *filename);
   virtual Bool_t      IsCompiled() {return fIsCompiled;}
   virtual Bool_t      Select(Int_t entry) {return kTRUE;}

   ClassDef(TSelector,0)  //A utility class for Trees selections.
};

#endif

