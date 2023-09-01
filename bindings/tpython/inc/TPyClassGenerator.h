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
   TClass *GetClass(const char *name, Bool_t load) override;
   TClass *GetClass(const std::type_info &typeinfo, Bool_t load) override;
   TClass *GetClass(const char *name, Bool_t load, Bool_t silent) override;
   TClass *GetClass(const std::type_info &typeinfo, Bool_t load, Bool_t silent) override;
};

#endif // !ROOT_TPyClassGenerator
