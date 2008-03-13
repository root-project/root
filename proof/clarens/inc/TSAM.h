// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    07/11/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSAM
#define ROOT_TSAM

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSAM                                                                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TClProxy
#include "TClProxy.h"
#endif


class TList;
class TString;
class TXmlRpc;


class TSAM : public TClProxy {
public:
   TSAM(TXmlRpc *rpc);
   virtual ~TSAM() { }

   Bool_t   GetVersion(TString &version);

   Bool_t   GetDatasets(TList *&datasets);
   Bool_t   GetDSetLocations(const Char_t *dataset, TList *&lmUrls);
   Bool_t   GetDSetFiles(const Char_t *dataset, const Char_t *lmUrl, TList *&files);
   Bool_t   GetDSetSize(const Char_t *dataset, Long64_t &size);

   ClassDef(TSAM,0);  // PEAC SAM proxy
};

#endif
