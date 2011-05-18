// @(#)root/peac:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPEAC
#define ROOT_TPEAC

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPEAC                                                                //
//                                                                      //
// Setup of a PROOF session using PEAC                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TGM;
class TDSet;
class TProof;


class TPEAC : public TObject {

private:
   TGM           *fGM;        //global manager object
   TString        fSessionID; //session id gotten from clarens
   TString        fDataSet;   //dataset used to create session
   TProof        *fProof;     //proof session started in StartSession

   TPEAC();

public:
   virtual ~TPEAC();

   static void    Init();
   TDSet         *StartSession(const Char_t *dataset);
   void           EndSession();
   void           EndSessionCallback();

   ClassDef(TPEAC,0)  // Manage PROOF sessions using PEAC
};


R__EXTERN TPEAC *gPEAC;

#endif
