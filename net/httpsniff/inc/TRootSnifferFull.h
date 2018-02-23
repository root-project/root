// $Id$
// Author: Sergey Linev   23/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootSnifferFull
#define ROOT_TRootSnifferFull

#include "TRootSniffer.h"

class TRootSnifferFull : public TRootSniffer {

public:
   TRootSnifferFull(const char *name, const char *objpath = "Objects");
   virtual ~TRootSnifferFull();

   virtual Bool_t ProduceImage(Int_t kind, const char *path, const char *options, void *&ptr, Long_t &length);

   virtual Bool_t ProduceXml(const char *path, const char *options, TString &res);

   virtual Bool_t ProduceExe(const char *path, const char *options, Int_t reskind, TString *ret_str,
                             void **ret_ptr = nullptr, Long_t *ret_length = nullptr);

   ClassDef(TRootSnifferFull, 0) // Sniffer of ROOT objects
};

#endif
