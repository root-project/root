// @(#)root/proof:$Name:  $:$Id: TProofLimitsFinder.cxx,v 1.3 2005/03/10 17:57:04 rdm Exp $
// Author: Maarten Ballintijn   19/04/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLimitsFinder                                                   //
//                                                                      //
// Class to find nice axis limits and synchonize them between slaves    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofLimitsFinder.h"
#include "TProofServ.h"
#include "TSocket.h"
#include "TH1.h"
#include "TMessage.h"
#include "TProofDebug.h"
#include "TError.h"

ClassImp(TProofLimitsFinder)

//______________________________________________________________________________
TProofLimitsFinder::TProofLimitsFinder()
{
}

//______________________________________________________________________________
TProofLimitsFinder::~TProofLimitsFinder()
{
}


//______________________________________________________________________________
void TProofLimitsFinder::AutoBinFunc(TString& key,
                                     Double_t& xmin, Double_t& xmax,
                                     Double_t& ymin, Double_t& ymax,
                                     Double_t& zmin, Double_t& zmax)
{
   if ( gProofServ == 0 ) return;

   TSocket *s = gProofServ->GetSocket();
   TMessage mess(kPROOF_AUTOBIN);

   PDB(kGlobal, 2) {
      TProofLimitsFinder f;
      f.Info("AutoBinFunc", Form("Sending %f, %f, %f, %f, %f, %f", xmin, xmax, ymin, ymax, zmin, zmax));
   }
   mess << key << xmin << xmax << ymin << ymax << zmin << zmax;

   s->Send(mess);

   TMessage *answ;
   if (s->Recv(answ) <= 0)
      return;

   (*answ) >> key >> xmin >> xmax >> ymin >> ymax >> zmin >> zmax;

   delete answ;
}

//______________________________________________________________________________
Int_t TProofLimitsFinder::FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax)
{
   // Find good limits

   Double_t dummy = 0;

   TString key = h->GetName();
   AutoBinFunc(key, xmin, xmax, dummy, dummy, dummy, dummy);

   return THLimitsFinder::FindGoodLimits( h, xmin, xmax);
}


//______________________________________________________________________________
Int_t TProofLimitsFinder::FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax)
{
   // Find good limits

   Double_t dummy = 0;

   TString key = h->GetName();
   AutoBinFunc(key, xmin, xmax, ymin, ymax, dummy, dummy);

   return THLimitsFinder::FindGoodLimits( h, xmin, xmax, ymin, ymax);
}


//______________________________________________________________________________
Int_t TProofLimitsFinder::FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax, Axis_t zmin, Axis_t zmax)
{
   // Find good limits

   TString key = h->GetName();
   AutoBinFunc(key, xmin, xmax, ymin, ymax, zmin, zmax);

   return THLimitsFinder::FindGoodLimits( h, xmin, xmax, ymin, ymax, zmin, zmax);
}
