// @(#):$Id$
// Author: Andrei Gheata   07/02/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoStateInfo.h"
#include "TGeoNode.h"
#include "TGeoPolygon.h"
#include "TGeoManager.h"

//_____________________________________________________________________________
TGeoStateInfo::TGeoStateInfo()
              :fNode(0),
               fAsmCurrent(0),
               fAsmNext(0),
               fDivCurrent(0),
               fDivNext(0),
               fDivTrans(),
               fDivRot(),
               fDivCombi(),
               fVoxNcandidates(0),
               fVoxCurrent(0),
               fVoxCheckList(0),
               fVoxBits1(0),
               fBoolSelected(0),
               fXtruSeg(0),
               fXtruIz(0),
               fXtruXc(0),
               fXtruYc(0),
               fXtruPoly(0)
{
// Constructor
   Int_t maxDaughters = TGeoManager::GetMaxDaughters();
   Int_t maxXtruVert  = TGeoManager::GetMaxXtruVert();
   fVoxCheckList = new Int_t[maxDaughters];
   fVoxBits1 = new UChar_t[2 + ((maxDaughters-1)>>3)];
   fXtruXc = new Double_t[maxXtruVert];
   fXtruYc = new Double_t[maxXtruVert];
}

//_____________________________________________________________________________
TGeoStateInfo::~TGeoStateInfo()
{
// Destructor
   delete [] fVoxCheckList;
   delete [] fVoxBits1;
   delete [] fXtruXc;
   delete [] fXtruYc;
}

//_____________________________________________________________________________
TGeoStateInfo::TGeoStateInfo(const TGeoStateInfo &other)
              :fNode(other.fNode),
               fAsmCurrent(other.fAsmCurrent),
               fAsmNext(other.fAsmNext),
               fDivCurrent(other.fDivCurrent),
               fDivNext(other.fDivNext),
               fDivTrans(other.fDivTrans),
               fDivRot(other.fDivRot),
               fDivCombi(other.fDivCombi),
               fVoxNcandidates(other.fVoxNcandidates),
               fVoxCurrent(other.fVoxCurrent),
               fVoxCheckList(0),
               fVoxBits1(0),
               fBoolSelected(other.fBoolSelected),
               fXtruSeg(other.fXtruSeg),
               fXtruIz(other.fXtruIz),
               fXtruXc(0),
               fXtruYc(0),
               fXtruPoly(other.fXtruPoly)
{
// Copy constructor.
   Int_t maxDaughters = TGeoManager::GetMaxDaughters();
   Int_t maxXtruVert  = TGeoManager::GetMaxXtruVert();
   fVoxCheckList = new Int_t[maxDaughters];
   fVoxBits1 = new UChar_t[1 + ((maxDaughters-1)>>3)];
   fXtruXc = new Double_t[maxXtruVert];
   fXtruYc = new Double_t[maxXtruVert];
}

//_____________________________________________________________________________
TGeoStateInfo &TGeoStateInfo::operator=(const TGeoStateInfo &other)
{
// Assignment
   if (this==&other) return *this;
   fNode = other.fNode;
   fAsmCurrent = other.fAsmCurrent;
   fAsmNext = other.fAsmNext;
   fDivCurrent = other.fDivCurrent;
   fDivNext = other.fDivNext;
   fDivTrans = other.fDivTrans;
   fDivRot = other.fDivRot;
   fDivCombi = other.fDivCombi;
   fVoxNcandidates = other.fVoxNcandidates;
   fVoxCurrent = other.fVoxCurrent;
   fBoolSelected = other.fBoolSelected;
   fXtruSeg = other.fXtruSeg;
   fXtruIz = other.fXtruIz;
   fXtruPoly = other.fXtruPoly;
   return *this;
}
