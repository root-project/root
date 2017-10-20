// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpWSHandler.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpWSHandler                                                       //
//                                                                      //
// Abstract class for processing websocket requests                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(THttpWSHandler);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

THttpWSHandler::THttpWSHandler(const char *name, const char *title) :
   TNamed(name, title)
{
}

THttpWSHandler::~THttpWSHandler()
{
}
