// @(#)root/net:$Name:  $:$Id: TAlienProof.h,v 1.0 2003/09/05 10:00:00 peters Exp $
// Author: Andreas Peters  05/09/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienProof
#define ROOT_TAlienProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienProof                                                          //
//                                                                      //
// Class defining the interface to AliEn GRID PROOF Services.           //
// Objects of this class are created by TAlien methods.                 //
//                                                                      //
// Related classes are TAlien.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TDSet
#include "TDSet.h"
#endif

#ifndef ROOT_TGridProof
#include "TGridProof.h"
#endif


class TAlienProof : public TGridProof {

private:
   enum { kALIENPROOFMAXSITES = 1024 };

   Int_t        fNsites;                      // size of the fSites/fNtimes arrays
   std::string *fSites[kALIENPROOFMAXSITES];  // ugly string array with the site names
   std::string *fNtimes[kALIENPROOFMAXSITES]; // ugly string array with the number of daemons per site
   TString      fProofSiteBlob;               // ugly string with the assigned PROOF daemons

   void DSetProofServiceAnswer(TDSet *dse);  // private AliEn function to interprete the answer of the Proof service
   void DSetRepack(TDSet *dse);              // private AliEn function to set the assigned daemons in the dse Msn list

public:
   TAlienProof();
   virtual ~TAlienProof();

   Bool_t      Request(TDSet *dset);
   void        Connect();
   const char *GetSiteBlob() const { return fProofSiteBlob; }
   void        PrintSiteBlob() { printf("SiteBlob: %s\n", fProofSiteBlob.Data()); }
   Int_t       DSetMsnExtraction(TDSet *dset);  // provide GridProof with the dataset to extract the required daemons

   ClassDef(TAlienProof,0)  // AliEn Proof Service Class
};

#endif
