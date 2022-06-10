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

#include "THttpCallArg.h"

/** \class THttpWSEngine

Internal instance used to exchange WS functionality between
THttpServer and THttpWSHandler.

Normally should not be used directly

*/


////////////////////////////////////////////////////////////////////////////////
/// Envelope for sending string via the websocket

void THttpWSEngine::SendCharStar(const char *str)
{
   if (str)
      Send(str, strlen(str));
}

////////////////////////////////////////////////////////////////////////////////
/// Method should be invoked before processing data coming from websocket
/// If method returns kTRUE, data is processed internally and
/// not dedicated for further usage

Bool_t THttpWSEngine::PreProcess(std::shared_ptr<THttpCallArg> &)
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Method invoked after user process data received via websocket

void THttpWSEngine::PostProcess(std::shared_ptr<THttpCallArg> &)
{
}
