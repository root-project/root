// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGM
#define ROOT_TGM

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGM                                                                  //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClProxy
#include "TClProxy.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TList;
class TUrl;
class TXmlRpc;


class TGM : public TClProxy {
public:
   TGM(TXmlRpc *rpc);
   virtual ~TGM() { }

   Bool_t   GetVersion(TString &version);
   Bool_t   CreateSession(const Char_t *dataset,
                          TString &sessionid,
                          TList   *&list,
                          TUrl    &proofUrl);
   Bool_t   DestroySession(const Char_t *sessionid);

   struct TFileParams : public TObject {
      TString  fFileName;
      TString  fObjClass;
      TString  fObjName;
      TString  fDir;
      Long64_t fFirst;
      Long64_t fNum;

      TFileParams(const Char_t *file, const Char_t *cl, const Char_t *nm,
                  const Char_t *dir, Int_t first,  Int_t num);

      void     Print(Option_t *option="") const;

      ClassDef(TGM::TFileParams,0);  // PEAC File description
   };

   ClassDef(TGM,0);  // PEAC Global Manager proxy
};

#endif
