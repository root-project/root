// @(#)root/tree:$Name$:$Id$
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TChainElement describes a component of a TChain.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"
#include "TChainElement.h"

ClassImp(TChainElement)

//______________________________________________________________________________
TChainElement::TChainElement(): TNamed()
{
//*-*-*-*-*-*Default constructor for a chain element*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =======================================

   fNPackets   = 0;
   fPackets    = 0;
   fEntries    = 0;
   fPacketSize = 100;
   fBaddress   = 0;
   fStatus     = -1;
}

//______________________________________________________________________________
TChainElement::TChainElement(const char *name, const char *title)
    :TNamed(name,title)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a chain element*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ======================
//

   fNPackets   = 0;
   fPackets    = 0;
   fEntries    = 0;
   fPacketSize = 100;
   fBaddress   = 0;
   fStatus     = -1;
}

//______________________________________________________________________________
TChainElement::~TChainElement()
{
//*-*-*-*-*-*Default destructor for a chain element*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ======================================

   delete [] fPackets;
}

//_______________________________________________________________________
void TChainElement::CreatePackets()
{
//*-*-*-*-*-*-*-*-*Initialize the packet descriptor string*-*-*-*-*-*-*-*-*-*
//*-*              =======================================

   fNPackets = 1 + fEntries/fPacketSize;
   delete [] fPackets;
   fPackets = new char[fNPackets+1];
   for (Int_t i=0;i<fNPackets;i++) fPackets[i] = ' ';
   fPackets[fNPackets] = 0;

}

//_______________________________________________________________________
void TChainElement::SetPacketSize(Int_t size)
{
//*-*-*-*-*-*-*-*-*Set number of entries per packet for parallel root*-*-*-*-*
//*-*              =================================================

   fPacketSize = size;
}
