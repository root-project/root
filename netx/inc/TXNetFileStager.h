// @(#)root/net:$Name:  $:$Id: TXNetFileStager.h,v 1.2 2007/02/20 12:53:09 rdm Exp $
// Author: A. Peters, G. Ganis   7/2/2007

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXNetFileStager
#define ROOT_TXNetFileStager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetFileStager                                                      //
//                                                                      //
// Interface to the 'XRD' staging capabilities.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFileStager
#include "TFileStager.h"
#endif

class TXNetSystem;

class TXNetFileStager : public TFileStager {

private:
   TString        fPrefix; // prefix to prepend to requests
   TXNetSystem   *fSystem; // instance of the admin interface

   static void    GetPrefix(const char *url, TString &pfx);

public:
   TXNetFileStager(const char *stager = "");
   virtual ~TXNetFileStager();

   Bool_t  IsStaged(const char *path);
   Int_t   Locate(const char *path, TString &endpath);
   Bool_t  Matches(const char *s);
   Bool_t  Stage(const char *path, Option_t *opt = 0);
   Bool_t  Stage(TList *pathlist, Option_t *opt = 0)
              { return TFileStager::Stage(pathlist, opt); }

   Bool_t  IsValid() const { return (fSystem ? kTRUE : kFALSE); }

   void    Print(Option_t *option = "") const;

   ClassDef(TXNetFileStager,0)  // Interface to a 'XRD' staging
};

#endif
