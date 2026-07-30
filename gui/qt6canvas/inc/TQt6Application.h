// Author: Sergey Linev   30/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TQt6Application
#define ROOT_TQt6Application


#include "TApplicationImp.h"

namespace ROOT {
namespace Experimental {

class TQt6Application : public TApplicationImp {
public:
   TQt6Application() = delete;
   TQt6Application(const char *appClassName, Int_t *argc, char **argv);
   ~TQt6Application() override;

   void    Show() override {}
   void    Hide() override {}
   void    Iconify() override {}
   Bool_t  IsCmdThread() override { return kTRUE; }
   void    Init() override {}
   void    Open() override {}
   void    Raise() override {}
   void    Lower() override {}

   ClassDefOverride(TQt6Application,0)  // ROOT Qt6 GUI application environment
};

} // namespace Experimental
} // namespace ROOT


#endif
