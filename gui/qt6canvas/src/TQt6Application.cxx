// Author: Sergey Linev   30/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TQt6Application
    \ingroup qt6canvas

Provides Qt6-specific methods for TApplication.
Main purpose - runs Qt event loop
*/


#include "TQt6Application.h"


using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Create Qt6 application environment.

TQt6Application::TQt6Application(const char *appClassName,
                                 Int_t *argc, char **argv) : TApplicationImp(appClassName, argc, argv)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Delete Qt6 application environment.

TQt6Application::~TQt6Application()
{
}
