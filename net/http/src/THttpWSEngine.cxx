// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpWSEngine.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpWSEngine                                                        //
//                                                                      //
// Internal instance used to exchange WS functionality between          //
// THttpServer and THttpWSHandler. Normally should not be used directly //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(THttpWSEngine);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

THttpWSEngine::THttpWSEngine(const char *name, const char *title)
   : TNamed(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

THttpWSEngine::~THttpWSEngine()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Envelope for sending string via the websocket

void THttpWSEngine::SendCharStar(const char *str)
{
   if (str) Send(str, strlen(str));
}

