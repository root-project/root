// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    26/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClProxy
#define ROOT_TClProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClProxy                                                             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TXmlRpc;


class TClProxy : public TObject {
protected:
   TXmlRpc    *fRpc;       //RPC data

public:
   TClProxy(const Char_t *service, TXmlRpc *rpc);
   virtual ~TClProxy() { }

   void     Print(Option_t *option="") const;
   Bool_t   RpcFailed(const Char_t *member, const Char_t *what);

   ClassDef(TClProxy,0);  // Clarens Proxy base
};

#endif
