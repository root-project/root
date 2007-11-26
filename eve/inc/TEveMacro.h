// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveMacro
#define ROOT_TEveMacro

#include "TEveUtil.h"

#include "TMacro.h"

class TEveMacro : public TMacro
{
protected:

public:
   TEveMacro();
   TEveMacro(const TEveMacro&);
   TEveMacro(const char* name);
   virtual ~TEveMacro() {}

   virtual Long_t Exec(const char* params = "0", Int_t* error = 0);

   void ResetRoot();

   ClassDef(TEveMacro, 1); // TMacro wrapper (attempting to fix issues with different macro loading and execution schemes).
};

#endif
