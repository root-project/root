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

#ifndef ROOT_TClarens
#define ROOT_TClarens

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClarens                                                             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TGM;
class TLM;
class TSAM;
class TEcho;
class THashList;
class TClSession;


class TClarens : public TObject {
private:
   Long_t      fTimeout;    //timeout on xmlrpc calls
   THashList  *fSessions;   //lsit of clarens sessions by URL

   TClarens();

   TClSession *Connect(const Char_t *url);

public:
   virtual ~TClarens();

   void           SetTimeout(Long_t msec) {fTimeout = msec;}
   Int_t          GetTimeout() const {return fTimeout;}

   TEcho         *CreateEcho(const Char_t *echoUrl);
   TGM           *CreateGM(const Char_t *gmUrl);
   TLM           *CreateLM(const Char_t *lmUrl);
   TSAM          *CreateSAM(const Char_t *samUrl);

   static void    Init();

   ClassDef(TClarens,0);  // Clarens main interface
};


R__EXTERN TClarens *gClarens;

#endif
