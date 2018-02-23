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

   ClassDef(TRootSnifferFull, 0) // Sniffer of ROOT objects
};

#endif
