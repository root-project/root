// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    25/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEcho
#define ROOT_TEcho

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEcho                                                                //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClProxy
#include "TClProxy.h"
#endif


class TString;
class TXmlRpc;


class TEcho : public TClProxy {
public:
   TEcho(TXmlRpc *rpc);
   virtual ~TEcho() { }

   Bool_t   Echo(const Char_t *in, TString &out);
   Bool_t   Hostname(TString &name, TString &ip);

   void     Benchmark(Int_t iterations);

   ClassDef(TEcho,0);  // Echo proxy
};

#endif
