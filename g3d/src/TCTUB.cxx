// @(#)root/g3d:$Name:  $:$Id: TCTUB.cxx,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Rene Brun   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TCTUB.h"
#include "TClass.h"

ClassImp(TCTUB)

//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/ctub.gif"> </P> End_Html
//                                                                        //
// 'CTUB' is a cut  tube with 11 parameters.  The  first 5 parameters     //
//        are the same  as for the TUBS.  The  remaining 6 parameters     //
//        are the director  cosines of the surfaces  cutting the tube     //
//        respectively at the low and high Z values.                      //
//                                                                        //
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - rmin       inside radius
//     - rmax       outside radius
//     - dz         half length in z
//     - phi1       starting angle of the segment
//     - phi2       ending angle of the segment
//     - coslx      x dir cosinus at low z face
//     - cosly      y dir cosinus at low z face
//     - coslz      z dir cosinus at low z face
//     - coshx      x dir cosinus at high z face
//     - coshy      y dir cosinus at high z face
//     - coshz      z dir cosinus at high z face


//______________________________________________________________________________
TCTUB::TCTUB()
{
//*-*-*-*-*-*-*-*-*-*-*-*CTUB shape default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================


}


//______________________________________________________________________________
TCTUB::TCTUB(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2,
             Float_t coslx, Float_t cosly, Float_t coslz,
             Float_t coshx, Float_t coshy, Float_t coshz)
      : TTUBS(name,title,material,rmin,rmax,dz,phi1,phi2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CTUB shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fCosLow[0]  = coslx;
    fCosLow[1]  = cosly;
    fCosLow[2]  = coslz;
    fCosHigh[0] = coshx;
    fCosHigh[1] = coshy;
    fCosHigh[2] = coshz;
    TMath::Normalize(fCosLow);
    TMath::Normalize(fCosHigh);
}

//______________________________________________________________________________
TCTUB::TCTUB(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2,
             Float_t *lowNormal, Float_t *highNormal)
      : TTUBS(name,title,material,rmin,rmax,dz,phi1,phi2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CTUB shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================
    memcpy(fCosLow, lowNormal, sizeof(fCosLow) );
    memcpy(fCosHigh,highNormal,sizeof(fCosHigh));
    TMath::Normalize(fCosLow);
    TMath::Normalize(fCosHigh);
}

//______________________________________________________________________________
TCTUB::~TCTUB()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CTUB shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

}

//______________________________________________________________________________
static Float_t Product(const Float_t *x, const Float_t *y)
{
 Double_t s = 0;
 for (int i= 0 ; i <2 ; i++ ) s += x[i]*y[i];
 return Float_t(s);
}
//______________________________________________________________________________
void TCTUB::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create TUBS points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    Float_t dz;
    Int_t j, n;

    n = GetNumberOfDivisions()+1;

    dz   = TTUBE::fDz;

    if (buff) {
        Int_t indx = 0;
//*-* We've to checxk whether the table does exist and create it
//*-* since fCoTab/fSiTab are not saved with any TShape::Streamer function

        if (!fCoTab)   MakeTableOfCoSin();

        for (j = 0; j < n; j++) {
            buff[indx+6*n] = buff[indx] = fRmin * fCoTab[j];
            indx++;
            buff[indx+6*n] = buff[indx] = fAspectRatio*fRmin * fSiTab[j];
            indx++;
            buff[indx+6*n] = dz;
            buff[indx+6*n]-= Product(&buff[indx+6*n-2],fCosHigh)/fCosHigh[2];
            buff[indx]     =-dz;
            buff[indx]    -= Product(&buff[indx-2],fCosLow)/fCosLow[2];
            indx++;
        }
        for (j = 0; j < n; j++) {
            buff[indx+6*n] = buff[indx] = fRmax * fCoTab[j];
            indx++;
            buff[indx+6*n] = buff[indx] = fAspectRatio*fRmax * fSiTab[j];
            indx++;
            buff[indx+6*n] = dz;
            buff[indx+6*n]-= Product(&buff[indx+6*n-2],fCosHigh)/fCosHigh[2];
            buff[indx]     =-dz;
            buff[indx]    -= Product(&buff[indx-2],fCosLow)/fCosLow[2];
            indx++;
        }
    }
}


//______________________________________________________________________________
void TCTUB::Streamer(TBuffer &R__b)
{
   // Stream an object of class TCTUB.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TCTUB::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TTUBS::Streamer(R__b);
      R__b.ReadStaticArray(fCosLow);
      R__b.ReadStaticArray(fCosHigh);
      R__b.CheckByteCount(R__s, R__c, TCTUB::IsA());
      //====end of old versions
      
   } else {
      TCTUB::Class()->WriteBuffer(R__b,this);
   }
}
