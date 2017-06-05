// $Id$
// Author: Sergey Linev   21/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpEngine.h"

#include <string.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpEngine                                                          //
//                                                                      //
// Abstract class for implementing http protocol for THttpServer        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(THttpEngine);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

THttpEngine::THttpEngine(const char *name, const char *title)
   : TNamed(name, title), fServer(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

THttpEngine::~THttpEngine()
{
   fServer = 0;
}

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

