// @(#)root/proof:$Name:$:$Id:$
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


ClassImp(TDSetProxy)


//______________________________________________________________________________
TDSetProxy::TDSetProxy()
{
   fServ = 0;
}


//______________________________________________________________________________
TDSetProxy::TDSetProxy(const char *type, const char *objname, const char *dir)
   : TDSet(type,objname,dir)
{
   fServ = 0;
   fCurrent = 0;
}


//______________________________________________________________________________
void TDSetProxy::SetProofServ(TProofServ *serv)
{
   fServ = serv;
   fCurrent = 0;
}


//______________________________________________________________________________
void TDSetProxy::Reset()
{
   delete fCurrent; fCurrent = 0;
}


//______________________________________________________________________________
TDSetElement *TDSetProxy::Next()
{
   fCurrent = fServ->GetNextPacket();

   return fCurrent;
}
