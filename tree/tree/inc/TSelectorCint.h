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
#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif

class TSelectorCint : public TSelector {

private:
   void SetFuncProto(CallFunc_t *cf, ClassInfo_t *cl, const char* fname,
                     const char* argtype, Bool_t required = kTRUE);

protected:
   ClassInfo_t  *fClass;        //!
   CallFunc_t   *fFuncVersion;  //!
   CallFunc_t   *fFuncInit;     //!
   CallFunc_t   *fFuncBegin;    //!
   CallFunc_t   *fFuncSlBegin;  //!
   CallFunc_t   *fFuncNotif;    //!
   CallFunc_t   *fFuncSlTerm;   //!
   CallFunc_t   *fFuncTerm;     //!
   CallFunc_t   *fFuncCut;      //!
   CallFunc_t   *fFuncFill;     //!
   CallFunc_t   *fFuncProc;     //!
   CallFunc_t   *fFuncOption;   //!
   CallFunc_t   *fFuncObj;      //!
   CallFunc_t   *fFuncInp;      //!
   CallFunc_t   *fFuncOut;      //!
   CallFunc_t   *fFuncAbort;    //!
   CallFunc_t   *fFuncGetAbort; //!
   CallFunc_t  *fFuncResetAbort;//!
   CallFunc_t   *fFuncGetStat;  //!
   TSelector    *fIntSelector;  //Pointer to interpreted selector (if interpreted)
   Bool_t        fIsOwner;      //True if fIntSelector shoudl be deleted when the this object is deleted.

public:
   TSelectorCint();
   virtual            ~TSelectorCint();
   virtual void        Build(TSelector *iselector, ClassInfo_t *cl, Bool_t isowner = kTRUE);
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
   virtual void        Abort(const char *why, EAbort what = kAbortProcess);
   virtual EAbort      GetAbort() const;
   virtual void        ResetAbort();
   virtual Long64_t    GetStatus() const;
   virtual TClass     *GetInterpretedClass() const;
   virtual TSelector  *GetInterpretedSelector() const { return fIntSelector; }

   ClassDef(TSelectorCint,0)  //A utility class for tree and object processing (interpreted version)
};

#endif

