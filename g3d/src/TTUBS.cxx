// @(#)root/g3d:$Name:  $:$Id: TTUBS.cxx,v 1.2 2002/11/11 11:21:16 brun Exp $
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTUBS.h"
#include "TNode.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TGeometry.h"

ClassImp(TTUBS)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/tubs.gif"> </P> End_Html
// TUBS is a segment of a tube. It has 8 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - rmin       inside radius
//     - rmax       outside radius
//     - dz         half length in z
//     - phi1       starting angle of the segment
//     - phi2       ending angle of the segment
//
//
// NOTE: phi1 should be smaller than phi2. If this is not the case,
//       the system adds 360 degrees to phi2.


//______________________________________________________________________________
TTUBS::TTUBS()
{
   // TUBS shape default constructor
}


//______________________________________________________________________________
TTUBS::TTUBS(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2)
      : TTUBE(name,title,material,rmin,rmax,dz)
{
   // TUBS shape normal constructor

   fPhi1 = phi1;
   fPhi2 = phi2;
   MakeTableOfCoSin();
}


//______________________________________________________________________________
TTUBS::TTUBS(const char *name, const char *title, const char *material, Float_t rmax, Float_t dz,
               Float_t phi1, Float_t phi2)
      : TTUBE(name,title,material,rmax,dz)
{
   // TUBS shape "simplified" constructor

   fPhi1 = phi1;
   fPhi2 = phi2;
   MakeTableOfCoSin();
}
//______________________________________________________________________________
void TTUBS::MakeTableOfCoSin()
{
   const Double_t PI  = TMath::ATan(1) * 4.0;
   const Double_t TWOPI  =2*PI;
   const Double_t ragrad  = PI/180.0;

   Int_t j;
   Int_t n = GetNumberOfDivisions () + 1;

   if (fCoTab)
      delete [] fCoTab; // Delete the old tab if any
      fCoTab = new Double_t [n];
   if (!fCoTab ) return;

   if (fSiTab)
      delete [] fSiTab; // Delete the old tab if any
   fSiTab = new Double_t [n];
   if (!fSiTab ) return;

   Double_t phi1    = Double_t(fPhi1  * ragrad);
   Double_t phi2    = Double_t(fPhi2  * ragrad);

   if (phi1 > phi2 ) phi2 += TWOPI;

   Double_t range = phi2- phi1;

   Double_t angstep = range/(n-1);

   Double_t ph = phi1;
   for (j = 0; j < n; j++) {
      ph = phi1 + j*angstep;
      fCoTab[j] = TMath::Cos(ph);
      fSiTab[j] = TMath::Sin(ph);
   }
}


//______________________________________________________________________________
TTUBS::~TTUBS()
{
   // TUBS shape default destructor
}


//______________________________________________________________________________
Int_t TTUBS::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a TUBE
   //
   // Compute the closest distance of approach from point px,py to each
   // computed outline point of the TUBE.

   Int_t n = GetNumberOfDivisions()+1;
   Int_t numPoints = n*4;
   return ShapeDistancetoPrimitive(numPoints,px,py);
}


//______________________________________________________________________________
void TTUBS::Paint(Option_t *option)
{
   // Paint this 3-D shape with its current attributes

   Int_t i, j;
   const Int_t n = GetNumberOfDivisions()+1;
   Int_t NbPnts = 4*n;
   Int_t NbSegs = 2*NbPnts;
   Int_t NbPols = NbPnts-2;

   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 3*NbSegs, 6*NbPols);
   if (!buff) return;

   buff->fType = TBuffer3D::kTUBS;
   buff->fId   = this;

   // Fill gPad->fBuffer3D. Points coordinates are in Master space
   buff->fNbPnts = NbPnts;
   buff->fNbSegs = NbSegs;
   buff->fNbPols = NbPols;
   // In case of option "size" it is not necessary to fill the buffer
   if (buff->fOption == TBuffer3D::kSIZE) {
      buff->Paint(option);
      return;
   }

   SetPoints(buff->fPnts);

   TransformPoints(buff);

   // Basic colors: 0, 1, ... 7
   Int_t c = ((GetLineColor() % 8) - 1) * 4;
   if (c < 0) c = 0;

   memset(buff->fSegs, 0, buff->fNbSegs*3*sizeof(Int_t));
   for (i = 0; i < 4; i++) {
      for (j = 1; j < n; j++) {
         buff->fSegs[(i*n+j-1)*3  ] = c;
         buff->fSegs[(i*n+j-1)*3+1] = i*n+j-1;
         buff->fSegs[(i*n+j-1)*3+2] = i*n+j;
      }
   }
   for (i = 4; i < 6; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c+1;
         buff->fSegs[(i*n+j)*3+1] = (i-4)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (i-2)*n+j;
      }
   }
   for (i = 6; i < 8; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c;
         buff->fSegs[(i*n+j)*3+1] = 2*(i-6)*n+j;
         buff->fSegs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
      }
   }

   Int_t indx = 0;
   memset(buff->fPols, 0, buff->fNbPols*6*sizeof(Int_t));
   i = 0;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (4+i)*n+j+1;
      buff->fPols[indx++] = (2+i)*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = i*n+j;
   }
   i = 1;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = i*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = (2+i)*n+j;
      buff->fPols[indx++] = (4+i)*n+j+1;
   }
   i = 2;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c+i;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (i-2)*2*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = ((i-2)*2+1)*n+j;
      buff->fPols[indx++] = (4+i)*n+j+1;
   }
   i = 3;
   for (j = 0; j < n-1; j++) {
      buff->fPols[indx++] = c+i;
      buff->fPols[indx++] = 4;
      buff->fPols[indx++] = (4+i)*n+j+1;
      buff->fPols[indx++] = ((i-2)*2+1)*n+j;
      buff->fPols[indx++] = (4+i)*n+j;
      buff->fPols[indx++] = (i-2)*2*n+j;
   }
   buff->fPols[indx++] = c+2;
   buff->fPols[indx++] = 4;
   buff->fPols[indx++] = 6*n;
   buff->fPols[indx++] = 4*n;
   buff->fPols[indx++] = 7*n;
   buff->fPols[indx++] = 5*n;
   buff->fPols[indx++] = c+2;
   buff->fPols[indx++] = 4;
   buff->fPols[indx++] = 6*n-1;
   buff->fPols[indx++] = 8*n-1;
   buff->fPols[indx++] = 5*n-1;
   buff->fPols[indx++] = 7*n-1;


   // Paint gPad->fBuffer3D
   buff->Paint(option);
}


//______________________________________________________________________________
void TTUBS::SetPoints(Double_t *buff)
{
   // Create TUBS points

   Int_t j, n;
   Int_t indx = 0;
   Float_t dz = TTUBE::fDz;

   n = GetNumberOfDivisions()+1;

   if (buff) {
      if (!fCoTab)   MakeTableOfCoSin();
      for (j = 0; j < n; j++) {
         buff[indx+6*n] = buff[indx] = fRmin * fCoTab[j];
         indx++;
         buff[indx+6*n] = buff[indx] = fAspectRatio*fRmin * fSiTab[j];
         indx++;
         buff[indx+6*n] = dz;
         buff[indx]     =-dz;
         indx++;
      }
      for (j = 0; j < n; j++) {
         buff[indx+6*n] = buff[indx] = fRmax * fCoTab[j];
         indx++;
         buff[indx+6*n] = buff[indx] = fAspectRatio*fRmax * fSiTab[j];
         indx++;
         buff[indx+6*n]= dz;
         buff[indx]    =-dz;
         indx++;
      }
   }
}


//______________________________________________________________________________
void TTUBS::Sizeof3D() const
{
   // Return total X3D needed by TNode::ls (when called with option "x")

   Int_t n = GetNumberOfDivisions()+1;

   gSize3D.numPoints += n*4;
   gSize3D.numSegs   += n*8;
   gSize3D.numPolys  += n*4-2;  
}
