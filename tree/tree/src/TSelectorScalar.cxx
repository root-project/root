// @(#)root/tree:$Id$
// Author: Maarten Ballintijn   13/02/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSelectorScalar
\ingroup tree

Named scalar type, based on Long64_t, streamable, storable and
mergeable. Ideally to be used in tree selectors in the PROOF
environment due to its merge functionality which allows a single
merged value to be returned to the user.
*/

#include "TSelectorScalar.h"
#include "TCollection.h"

ClassImp(TSelectorScalar);

////////////////////////////////////////////////////////////////////////////////
/// Increment scalar value by n.

void TSelectorScalar::Inc(Long_t n)
{
   SetVal(GetVal() + n);
}

////////////////////////////////////////////////////////////////////////////////
/// Merge scalars with scalars in the list. The scalar values are added.
/// Returns the number of scalars that were in the list.

Int_t TSelectorScalar::Merge(TCollection *list)
{
   TIter next(list);
   Int_t n = 0;

   while (TObject *obj = next()) {
      TSelectorScalar *c = dynamic_cast<TSelectorScalar*>(obj);
      if (c) {
         Inc(c->GetVal());
         n++;
      }
   }

   return n;
}
