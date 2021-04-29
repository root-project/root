// @(#)root/tree:$Id$
// Author: Maarten Ballintijn   13/02/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelectorScalar
#define ROOT_TSelectorScalar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorScalar                                                      //
//                                                                      //
// Named scalar type, based on Long64_t, streamable, storable and       //
// mergeable. Ideally to be used in tree selectors in the PROOF         //
// environment due to its merge functionality which allows a single     //
// merged value to be returned to the user.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TParameter.h"

#include "Rtypes.h"


class TCollection;

class TSelectorScalar : public TParameter<Long64_t> {

public:
   TSelectorScalar(const char *name = "", Long64_t val = 0)
             : TParameter<Long64_t>(name, val) { }
   ~TSelectorScalar() { }

   void     Inc(Long_t n = 1);
   Int_t    Merge(TCollection *list);

   ClassDef(TSelectorScalar,1)  // Mergeable scalar
};


#endif
