// @(#)root/tree:$Id$
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

namespace Cint {
   class G__CallFunc;
   class G__ClassInfo;
}

class TSelectorCint : public TSelector {

private:
   void SetFuncProto(Cint::G__CallFunc *cf, Cint::G__ClassInfo* cl, const char* fname,
                     const char* argtype, Bool_t required = kTRUE);

protected:
   Cint::G__ClassInfo  *fClass;        //!
   Cint::G__CallFunc   *fFuncVersion;  //!
   Cint::G__CallFunc   *fFuncInit;     //!
   Cint::G__CallFunc   *fFuncBegin;    //!
   Cint::G__CallFunc   *fFuncSlBegin;  //!
   Cint::G__CallFunc   *fFuncNotif;    //!
   Cint::G__CallFunc   *fFuncSlTerm;   //!
   Cint::G__CallFunc   *fFuncTerm;     //!
   Cint::G__CallFunc   *fFuncCut;      //!
   Cint::G__CallFunc   *fFuncFill;     //!
   Cint::G__CallFunc   *fFuncProc;     //!
   Cint::G__CallFunc   *fFuncOption;   //!
   Cint::G__CallFunc   *fFuncObj;      //!
   Cint::G__CallFunc   *fFuncInp;      //!
   Cint::G__CallFunc   *fFuncOut;      //!
   Cint::G__CallFunc   *fFuncGetAbort; //!
   Cint::G__CallFunc   *fFuncGetStat;  //!
   TSelector     *fIntSelector;  //Pointer to interpreted selector (if interpreted)
   Bool_t        fIsOwner;      //True if fIntSelector shoudl be deleted when the this object is deleted.

public:
   TSelectorCint();
   virtual            ~TSelectorCint();
   virtual void        Build(TSelector *iselector, Cint::G__ClassInfo *cl, Bool_t isowner = kTRUE);
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
   virtual Long64_t    GetStatus() const;

   ClassDef(TSelectorCint,0)  //A utility class for tree and object processing (interpreted version)
};

#endif

