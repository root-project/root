// @(#)root/proof:$Id$
// Author: Maarten Ballintijn  12/03/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDSetProxy                                                           //
//                                                                      //
// TDSet proxy for use on slaves.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDSetProxy.h"
#include "TProofServ.h"


ClassImp(TDSetProxy);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TDSetProxy::TDSetProxy()
{
   fServ = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TDSetProxy::TDSetProxy(const char *type, const char *objname, const char *dir)
   : TDSet(type,objname,dir)
{
   fServ = 0;
   fCurrent = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the reference TProofServ instance

void TDSetProxy::SetProofServ(TProofServ *serv)
{
   fServ = serv;
   fCurrent = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset this instance

void TDSetProxy::Reset()
{
   delete fCurrent; fCurrent = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the next packet

TDSetElement *TDSetProxy::Next(Long64_t totalEntries)
{
   fCurrent = fServ->GetNextPacket(totalEntries);

   // Check log file length (before processing the next packet, so we have the
   // chance to keep the latest logs)
   fServ->TruncateLogFile();

   return fCurrent;
}
