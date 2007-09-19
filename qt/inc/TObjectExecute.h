// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjectExecute
#define ROOT_TObjectExecute

#include "TQtRConfig.h"

#include "TObject.h"
#include "TFunction.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TROOT.h"

//______________________________________________________________________________
class TObjectExecute :  public TObject {
private:
  TObject *fObject;
  TMethod *fMethod;
  TFunction *fFunction;
  TObjArray *fParams;
protected:

public:
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  TObjectExecute():fObject(0),fMethod(0),fFunction(0),fParams(0)
  {}
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  void Execute(TObject *o,TMethod *m,TObjArray *p=0)
  {
    if (o && m) {
      fFunction=0;fObject=o;fMethod=m;fParams=p;
      o->Execute(m,p);
    }
  }
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  void  Execute(const char *method,  const char *params, Int_t *error=0)
  {TObject::Execute(method,params,error);}
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  void  Execute(TMethod *method, TObjArray *params, Int_t *error=0)
  {TObject::Execute(method,params,error);}

  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  void Execute(TObject *o,TFunction *f,TObjArray *p=0)
  {
    if (o && f) {
      fMethod=0;fObject=o;fFunction=f;fParams=p;
//      o->Execute(f,p);
    }
  }
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  void Execute(TFunction *f,TObjArray *p=0)
  {
    if (f) {
      fMethod=0;fObject=0;fFunction=f;fParams=p;
      TString args;
      TIter next(p);
      TObjString *s;
      while ((s = (TObjString*) next())) {
        if (!args.IsNull()) args += ",";
        args += s->String();
      }
      char *cmd = Form("%s(%s);", f->GetName(), args.Data());
      gROOT->ProcessLine(cmd);
    }
  }
  //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  virtual ~TObjectExecute(){}
};

#endif
