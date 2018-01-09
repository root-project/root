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

////////////////////////////////////////////////////////////////////////////////
/// Envelope for sending string via the websocket

void THttpWSEngine::SendCharStar(const char *str)
{
   if (str)
      Send(str, strlen(str));
}

////////////////////////////////////////////////////////////////////////////////
/// Method should be invoked before processing data coming from websocket
/// If method returns kTRUE, this is data is processed internally and
/// not dedicated for further usage

Bool_t THttpWSEngine::PreviewData(THttpCallArg &)
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Method invoked after user process data received via websocket
/// Normally request is no longer usable after that

void THttpWSEngine::PostProcess(THttpCallArg &)
{
}
