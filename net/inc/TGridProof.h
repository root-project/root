// @(#)root/net:$Name:  $:$Id: TGridProof.h,v 1.1 2003/11/13 15:15:11 rdm Exp $
// Author: Andreas Peters  05/09/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridProof
#define ROOT_TGridProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridProof                                                           //
//                                                                      //
// Abstract base class defining interface to a GRID PROOF Services.     //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TGrid.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TVirtualProof
#include "TVirtualProof.h"
#endif


class TGridProof : public TObject {

protected:
   TVirtualProof       *fProof;           // PROOF pointer
   Grid_ProofSession_t *fProofSession;	  // the PROOF session pointer
   TString              fProofMasterUrl;  // the URL of the PROOF master
   TString              fProofLogin;      // the Login to the PROOF master
   TString              fProofPasswd;     // the Password for the PROOF master
   TString              fProofConfigFile; // the Config File for the PROOF master

public:
   TGridProof() : fProof(0), fProofSession(0) { }
   virtual ~TGridProof();

   void ps(Int_t sessionId) const
   {
      if (gGrid)
         gGrid->ListProofSessions(sessionId);
   }
   void ps() const { ps(0); }
   void ls(const Option_t * = "") const
   {
      if (gGrid)
         gGrid->ListProofDaemons();
   }
   Bool_t kill(Int_t sessionId) const
   {
      if (gGrid)
         return gGrid->KillProofSession(sessionId);
      return kFALSE;
   }
   virtual Bool_t Request(TDSet *dset) = 0;
   virtual void   Connect() = 0;

   virtual const char *GetSiteBlob() const { return 0; }   // AliEn specific
   virtual void        PrintSiteBlob() { }                 // AliEn specific
   virtual Int_t       DSetMsnExtraction(TDSet *)
   {
      return 0;
   }      // provide GridProof with the dataset to extract the required daemons

   ClassDef(TGridProof,0)  // ABC defining interface to GRID proof services
};

#endif
