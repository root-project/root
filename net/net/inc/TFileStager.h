// @(#)root/net:$Id$
// Author: A. Peters, G. Ganis   7/2/2007

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileStager
#define ROOT_TFileStager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileStager                                                          //
//                                                                      //
// Abstract base class defining an interface to a stager.               //
//                                                                      //
// To open a connection to a stager use the static method               //
// Open("<stager>"), where <stager> contains a keyword allowing to load //
// the relevant plug-in, e.g.                                           //
//           TFileStager::Open("root://lxb6064.cern.ch")                //
// will load TXNetFileStager and initialize it for the redirector at    //
// lxb6046.cern.ch .                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TString.h"

class TCollection;
class TFileCollection;

class TFileStager : public TNamed {

public:
   TFileStager(const char *stager) : TNamed(stager, stager) { }
   virtual ~TFileStager() { }

   virtual TList          *GetStaged(TCollection *pathlist);
   virtual Bool_t          IsStaged(const char *);
   virtual Int_t           Locate(const char *u, TString &f);
   virtual Int_t           LocateCollection(TFileCollection *fc, Bool_t addDummyUrl = kFALSE);
   virtual Bool_t          Matches(const char *s) { return ((s && (fName == s)) ? kTRUE : kFALSE); }
   virtual Bool_t          Stage(const char *, Option_t * = 0) { return kTRUE; }
   virtual Bool_t          Stage(TCollection *pathlist, Option_t *opt = 0);

   virtual Bool_t          IsValid() const { return kTRUE; }

   // Extract the path name from supported object types
   static TString          GetPathName(TObject *o);

   //--- Load desired plugin
   static TFileStager *Open(const char *stager);

   ClassDefOverride(TFileStager,0)  // ABC defining interface to a stager
};

#endif
