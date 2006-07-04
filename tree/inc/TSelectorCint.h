// @(#)root/tree:$Name:  $:$Id: TSelectorCint.h,v 1.12 2005/02/21 09:41:39 rdm Exp $
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelectorCint
#define ROOT_TSelectorCint


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorCint                                                        //
//                                                                      //
// A utility class for Trees selections.  (via interpreter)             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TSelector
#include "TSelector.h"
#endif

class G__CallFunc;
class G__ClassInfo;

class TSelectorCint : public TSelector {

private:
   void SetFuncProto(G__CallFunc *cf, G__ClassInfo* cl, const char* fname,
                     const char* argtype, Bool_t required = kTRUE);

protected:
   G__ClassInfo  *fClass;        //!
   G__CallFunc   *fFuncVersion;  //!
   G__CallFunc   *fFuncInit;     //!
   G__CallFunc   *fFuncBegin;    //!
   G__CallFunc   *fFuncSlBegin;  //!
   G__CallFunc   *fFuncNotif;    //!
   G__CallFunc   *fFuncSlTerm;   //!
   G__CallFunc   *fFuncTerm;     //!
   G__CallFunc   *fFuncCut;      //!
   G__CallFunc   *fFuncFill;     //!
   G__CallFunc   *fFuncProc;     //!
   G__CallFunc   *fFuncOption;   //!
   G__CallFunc   *fFuncObj;      //!
   G__CallFunc   *fFuncInp;      //!
   G__CallFunc   *fFuncOut;      //!
   G__CallFunc   *fFuncGetAbort; //!
   G__CallFunc   *fFuncGetStat;  //!
   TSelector     *fIntSelector;  //Pointer to interpreted selector (if interpreted)

public:
   TSelectorCint();
   virtual            ~TSelectorCint();
   virtual void        Build(TSelector *iselector, G__ClassInfo *cl);
   virtual int         Version() const;
   virtual void        Init(TTree *);
   virtual void        Begin(TTree *tree);
   virtual void        SlaveBegin(TTree *);
   virtual Bool_t      Notify();
   virtual Bool_t      ProcessCut(Long64_t entry);
   virtual void        ProcessFill(Long64_t entry);
   virtual Bool_t      Process(Long64_t entry);
   virtual void        SetOption(const char *option);
   virtual void        SetObject(TObject *obj);
   virtual void        SetInputList(TList *input);
   virtual TList      *GetOutputList() const;
   virtual void        SlaveTerminate();
   virtual void        Terminate();
   virtual EAbort      GetAbort() const;
   virtual Int_t       GetStatus() const;

   ClassDef(TSelectorCint,0)  //A utility class for tree and object processing (interpreted version)
};

#endif

