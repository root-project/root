// @(#)root/treeplayer:$Name$:$Id$
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


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TTree;

class TSelector : public TNamed {

protected:
   TTree      *fTree;         //Pointer to current TTree

public:
   TSelector();
   TSelector(const char *name, const char *title="");
   virtual          ~TSelector();
   virtual void     BeginFile();
   virtual void     EndFile();
   virtual void     Execute(TTree *tree, Int_t event);
   void             Execute(const char *method,  const char *params);
   void             Execute(TMethod *method, TObjArray *params);
   virtual void     Finish(Option_t *option="");
   virtual void     Init(TTree *tree, Option_t *option="");
   virtual void     Start(Option_t *option="");

   ClassDef(TSelector,0)  //A utility class for Trees selections.
};

inline void TSelector::Execute(const char *method, const char *params)
   { TObject::Execute(method, params); }
inline void TSelector::Execute(TMethod *method, TObjArray *params)
   { TObject::Execute(method, params); }

#endif

