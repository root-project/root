// @(#)root/g3d:$Name:  $:$Id: TTUBE.cxx,v 1.3 2002/11/11 11:21:16 brun Exp $
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTUBE.h"
#include "TNode.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TGeometry.h"

ClassImp(TTUBE)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/tube.gif"> </P> End_Html
// TUBE is a tube. It has 6 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - rmin       inside radius
//     - rmax       outside radius
//     - dz         half length in z


//______________________________________________________________________________
TTUBE::TTUBE()
{
   // TUBE shape default constructor

   fCoTab = 0;
   fSiTab = 0;
   fAspectRatio =1;
}


//______________________________________________________________________________
TTUBE::TTUBE(const char *name, const char *title, const char *material, Float_t rmin, Float_t rmax, Float_t dz,Float_t aspect)
      : TShape(name, title,material)
{
   // TUBE shape normal constructor

   fRmin  = rmin;
   fRmax  = rmax;

   fDz   = dz;
   fNdiv = 0;

   fCoTab = 0;
   fSiTab = 0;

   fAspectRatio = aspect;

   MakeTableOfCoSin();
}


//______________________________________________________________________________
TTUBE::TTUBE(const char *name, const char *title, const char *material, Float_t rmax, Float_t dz)
      : TShape(name, title,material)
{
   // TUBE shape "simplified" constructor

   fRmin  = 0;
   fRmax  = rmax;

   fDz   = dz;
   fNdiv = 0;

   fCoTab = 0;
   fSiTab = 0;

   fAspectRatio = 1;

   MakeTableOfCoSin();
}


//______________________________________________________________________________
void TTUBE::MakeTableOfCoSin()
{
   const Double_t PI  = TMath::ATan(1) * 4.0;
   const Double_t TWOPI  =2*PI;

   Int_t j;
   Int_t n = GetNumberOfDivisions ();
   if (fCoTab)
      delete [] fCoTab; // Delete the old tab if any
      fCoTab = new Double_t [n];
   if (!fCoTab ) {
      Error("MakeTableOfCoSin()","No cos table done");
      return;
   }

   if (fSiTab) delete [] fSiTab; // Delete the old tab if any
   fSiTab = new Double_t [n];
   if (!fSiTab ) {
      Error("MakeTableOfCoSin()","No sin table done");
      return;
   }

   Double_t range = TWOPI;

   Double_t angstep = range/n;

   Double_t ph = 0;
   for (j = 0; j < n; j++) {
      ph = j*angstep;
      fCoTab[j] = TMath::Cos(ph);
      fSiTab[j] = TMath::Sin(ph);
   }
}


//______________________________________________________________________________
TTUBE::~TTUBE()
{
   // TUBE shape default destructor

   delete [] fCoTab;
   delete [] fSiTab;
}


//______________________________________________________________________________
Int_t TTUBE::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a TUBE
   //
   // Compute the closest distance of approach from point px,py to each
   // computed outline point of the TUBE.

   Int_t n = GetNumberOfDivisions();
   Int_t numPoints = n*4;
   return ShapeDistancetoPrimitive(numPoints,px,py);
}


//______________________________________________________________________________
void TTUBE::Paint(Option_t *option)
{
   // Paint this 3-D shape with its current attributes

   Int_t i, j, indx;
   Int_t n = GetNumberOfDivisions();
   Int_t NbPnts = 4*n;
   Int_t NbSegs = 8*n;
   Int_t NbPols = 4*n;
   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 3*NbSegs, 6*NbPols);
   if (!buff) return;

   buff->fType = TBuffer3D::kTUBE;
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

   for (i = 0; i < 4; i++) {
      for (j = 0; j < n; j++) {
         buff->fSegs[(i*n+j)*3  ] = c;
         buff->fSegs[(i*n+j)*3+1] = i*n+j;
         buff->fSegs[(i*n+j)*3+2] = i*n+j+1;
      }
      buff->fSegs[(i*n+j-1)*3+2] = i*n;
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

   indx = 0;
   /*
   for (i = 0; i < 2; i++) {
      for (j = 0; j < n; j++) {
         indx = 6*(i*n+j);
         buff->fPols[indx  ] = c;
         buff->fPols[indx+1] = 4;
         buff->fPols[indx+2] = i*n+j;
         buff->fPols[indx+3] = (4+i)*n+j;
         buff->fPols[indx+4] = (2+i)*n+j;
         buff->fPols[indx+5] = (4+i)*n+j+1;
      }
      buff->fPols[indx+5] = (4+i)*n;
   }
   for (i = 2; i < 4; i++) {
      for (j = 0; j < n; j++) {
         indx = 6*(i*n+j);
         buff->fPols[indx  ] = c+(i-2)*2+1;
         buff->fPols[indx+1] = 4;
         buff->fPols[indx+2] = (i-2)*2*n+j;
         buff->fPols[indx+3] = (4+i)*n+j;
         buff->fPols[indx+4] = ((i-2)*2+1)*n+j;
         buff->fPols[indx+5] = (4+i)*n+j+1;
      }
      buff->fPols[indx+5] = (4+i)*n;
   }
   */
   i=0;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+5] = i*n+j;
      buff->fPols[indx+4] = (4+i)*n+j;
      buff->fPols[indx+3] = (2+i)*n+j;
      buff->fPols[indx+2] = (4+i)*n+j+1;
   }
   buff->fPols[indx+2] = (4+i)*n;
   i=1;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+2] = i*n+j;
      buff->fPols[indx+3] = (4+i)*n+j;
      buff->fPols[indx+4] = (2+i)*n+j;
      buff->fPols[indx+5] = (4+i)*n+j+1;
   }
   buff->fPols[indx+5] = (4+i)*n;
   i=2;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c+i;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+2] = (i-2)*2*n+j;
      buff->fPols[indx+3] = (4+i)*n+j;
      buff->fPols[indx+4] = ((i-2)*2+1)*n+j;
      buff->fPols[indx+5] = (4+i)*n+j+1;
   }
   buff->fPols[indx+5] = (4+i)*n;
   i=3;
   for (j = 0; j < n; j++) {
      indx = 6*(i*n+j);
      buff->fPols[indx  ] = c+i;
      buff->fPols[indx+1] = 4;
      buff->fPols[indx+5] = (i-2)*2*n+j;
      buff->fPols[indx+4] = (4+i)*n+j;
      buff->fPols[indx+3] = ((i-2)*2+1)*n+j;
      buff->fPols[indx+2] = (4+i)*n+j+1;
   }
   buff->fPols[indx+2] = (4+i)*n;


   // Paint gPad->fBuffer3D
   buff->Paint(option);
}

//______________________________________________________________________________
void TTUBE::SetNumberOfDivisions (Int_t ndiv)
{
   // Set number of divisions used to draw this tube

   fNdiv = ndiv;
   MakeTableOfCoSin();
}


//______________________________________________________________________________
void TTUBE::SetPoints(Double_t *buff)
{
   // Create TUBE points
	            
   Int_t j, n;
   Int_t indx = 0;
		                
   n = GetNumberOfDivisions();
   
   if (buff) {
      if (!fCoTab)   MakeTableOfCoSin();
      for (j = 0; j < n; j++) {
         buff[indx+6*n] = buff[indx] = fRmin * fCoTab[j];
         indx++;
         buff[indx+6*n] = buff[indx] = fAspectRatio*fRmin * fSiTab[j];
         indx++;
         buff[indx+6*n] = fDz;
         buff[indx]     =-fDz;
         indx++;
      }
      for (j = 0; j < n; j++) {
         buff[indx+6*n] = buff[indx] = fRmax * fCoTab[j];
         indx++;
         buff[indx+6*n] = buff[indx] = fAspectRatio*fRmax * fSiTab[j];
         indx++;
         buff[indx+6*n]= fDz;
         buff[indx]    =-fDz;
         indx++;
      }
   }
}


//______________________________________________________________________________
void TTUBE::Sizeof3D() const
{
   // Return total X3D needed by TNode::ls (when called with option "x")

   Int_t n = GetNumberOfDivisions();
	                    
   gSize3D.numPoints += n*4;
   gSize3D.numSegs   += n*8;
   gSize3D.numPolys  += n*4;
}  


//______________________________________________________________________________
void TTUBE::Streamer(TBuffer &R__b)
{
   // Stream an object of class TTUBE.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TTUBE::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;  
      }
      //====process old versions before automatic schema evolution
      TShape::Streamer(R__b);
      R__b >> fRmin;
      R__b >> fRmax;
      R__b >> fDz;
      R__b >> fNdiv;
      if (R__v > 1) R__b >> fAspectRatio;
      R__b.CheckByteCount(R__s, R__c, TTUBE::IsA());
      //====end of old versions
   } else {
      TTUBE::Class()->WriteBuffer(R__b,this);
   }
}
