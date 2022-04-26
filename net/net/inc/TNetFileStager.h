// @(#)root/netx:$Id$
// Author: G. Ganis Feb 2011

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNetFileStager
#define ROOT_TNetFileStager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNetFileStager                                                       //
//                                                                      //
// TFileStager implementation for a 'rootd' backend.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileStager.h"

class TCollection;
class TNetSystem;

class TNetFileStager : public TFileStager {

private:
   TString        fPrefix; // prefix to prepend to requests
   TNetSystem   *fSystem; // instance of the admin interface

   static void    GetPrefix(const char *url, TString &pfx);

public:
   TNetFileStager(const char *stager = "");
   virtual ~TNetFileStager();

   Bool_t  IsStaged(const char *path) override;
   Int_t   Locate(const char *path, TString &endpath) override;
   Bool_t  Matches(const char *s) override;

   Bool_t  IsValid() const override { return fSystem ? kTRUE : kFALSE; }

   void    Print(Option_t *option = "") const override;

   ClassDefOverride(TNetFileStager,0)  // Implementation for a 'rootd' backend
};

#endif
