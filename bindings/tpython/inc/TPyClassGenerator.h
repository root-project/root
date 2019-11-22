// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL
//
// /*************************************************************************
//  * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
//  * All rights reserved.                                                  *
//  *                                                                       *
//  * For the licensing terms see $ROOTSYS/LICENSE.                         *
//  * For the list of contributors see $ROOTSYS/README/CREDITS.             *
//  *************************************************************************/

#ifndef ROOT_TPyClassGenerator
#define ROOT_TPyClassGenerator

// ROOT
#include "TClassGenerator.h"

class TPyClassGenerator : public TClassGenerator {
public:
   virtual TClass *GetClass(const char *name, Bool_t load);
   virtual TClass *GetClass(const std::type_info &typeinfo, Bool_t load);
   virtual TClass *GetClass(const char *name, Bool_t load, Bool_t silent);
   virtual TClass *GetClass(const std::type_info &typeinfo, Bool_t load, Bool_t silent);
};

#endif // !ROOT_TPyClassGenerator
