// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileSet
#define ROOT_TFileSet

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileSet                                                             //
//                                                                      //
// TFileSet class is a class to convert the                             //
//      "native file system structure"                                  //
// into an instance of the TDataSet class                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDataSet.h"
#include "TString.h"

class TFileSet : public TDataSet {
public:
   TFileSet();
   TFileSet(const TString &dirname, const Char_t *filename=0,Bool_t expand=kTRUE,Int_t maxDepth=10);
   virtual ~TFileSet();
   virtual Long_t HasData() const;
   virtual Bool_t IsEmpty() const;
   virtual Bool_t IsFolder() const;
   ClassDef(TFileSet,1)  // TDataSet class to read the native file system directory structure in
};

#endif
