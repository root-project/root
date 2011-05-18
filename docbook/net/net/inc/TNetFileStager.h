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

#ifndef ROOT_TFileStager
#include "TFileStager.h"
#endif

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

   Bool_t  IsStaged(const char *path);
   Int_t   Locate(const char *path, TString &endpath);
   Bool_t  Matches(const char *s);
   
   Bool_t  IsValid() const { return (fSystem ? kTRUE : kFALSE); }

   void    Print(Option_t *option = "") const;

   ClassDef(TNetFileStager,0)  // Implementation for a 'rootd' backend
};

#endif
