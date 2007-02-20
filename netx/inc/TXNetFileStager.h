// @(#)root/net:$Name:  $:$Id: TXNetFileStager.h,v 1.1 2007/02/14 18:25:22 rdm Exp $
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
#ifndef ROOT_TString
#include "TString.h"
#endif

class TXNetSystem;


class TXNetFileStager : public TFileStager {

private:
   TString        fPrefix; // prefix to prepend to requests
   TXNetSystem   *fSystem; // instance of the admin interface

   void           SetPrefix(const char *url);

public:
   TXNetFileStager(const char *stager = "");
   virtual ~TXNetFileStager();

   Bool_t  IsStaged(const char *path);
   Bool_t  Stage(const char *path, Option_t *opt = 0);
   Bool_t  Stage(TList *pathlist, Option_t *opt = 0)
              { return TFileStager::Stage(pathlist, opt); }

   Bool_t  IsValid() const { return (fSystem ? kTRUE : kFALSE); }

   void    Print(Option_t *option = "") const;

   ClassDef(TXNetFileStager,0)  // Interface to a 'XRD' staging
};

#endif
