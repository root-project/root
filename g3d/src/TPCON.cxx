// @(#)root/g3d:$Name:  $:$Id: TPCON.cxx,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $
// Author: Nenad Buncic   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPCON.h"
#include "TNode.h"
#include "TView.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"

ClassImp(TPCON)



//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/pcon.gif"> </P> End_Html
// PCON is a polycone. It has the following parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - phi1       the azimuthal angle phi at which the volume begins (angles
//                  are counted counterclockwise)
//     - dphi       opening angle of the volume, which extends from
//                  phi1 to phi1+dphi
//     - nz         number of planes perpendicular to the z axis where
//                  the dimension of the section is given -- this number
//                  should be at least 2
//     - rmin       array of dimension nz with minimum radius at a given plane
//     - rmax       array of dimension nz with maximum radius at a given plane
//     - z          array of dimension nz with z position of given plane



//______________________________________________________________________________
TPCON::TPCON()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PCON shape default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

    fRmin  = 0;
    fRmax  = 0;
    fDz    = 0;
    fCoTab = 0;
    fSiTab = 0;
}


//______________________________________________________________________________
TPCON::TPCON(const char *name, const char *title, const char *material, Float_t phi1, Float_t dphi1, Int_t nz)
      : TShape(name, title,material)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PCON shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================
//*-*  Parameters of the nz positions must be entered via TPCON::DefineSection.

    if (nz < 2 ) {
        Error(name, "number of z planes for %s must be at least two !", name);
        return;
    }
    fPhi1  = phi1;
    fDphi1 = dphi1;
    fNz    = nz;
    fNdiv  = 0;
    fRmin  = new Float_t [nz+1];
    fRmax  = new Float_t [nz+1];
    fDz    = new Float_t [nz+1];

    fCoTab = 0;
    fSiTab = 0;

    while (fDphi1 > 360) fDphi1 -= 360;

    MakeTableOfCoSin();
}


//______________________________________________________________________________
void TPCON::MakeTableOfCoSin()
{
    const Double_t PI  = TMath::ATan(1) * 4.0;
    const Double_t ragrad  = PI/180.0;

    Int_t n = GetNumberOfDivisions () + 1;
    if (fCoTab)
        delete [] fCoTab; // Delete the old tab if any
        fCoTab = new Double_t [n];
    if (!fCoTab ) return;

    if (fSiTab)
        delete [] fSiTab; // Delete the old tab if any
    fSiTab = new Double_t [n];
    if (!fSiTab ) return;

    Double_t range   = Double_t(fDphi1 * ragrad);
    Double_t phi1    = Double_t(fPhi1  * ragrad);
    Double_t angstep = range/(n-1);

    FillTableOfCoSin(phi1,angstep,n);
}
//______________________________________________________________________________
TPCON::~TPCON()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PCON shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    if (fRmin) delete [] fRmin;
    if (fRmax) delete [] fRmax;
    if (fDz)   delete [] fDz;
    if (fSiTab) delete [] fSiTab;
    if (fCoTab) delete [] fCoTab;

    fRmin = 0;
    fRmax = 0;
    fDz   = 0;
    fCoTab = 0;
    fSiTab = 0;

}


//______________________________________________________________________________
void TPCON::DefineSection(Int_t secNum, Float_t z, Float_t rmin, Float_t rmax)
{
//*-*-*-*-*-*-*-*-*-*Defines section secNum of the polycone*-*-*-*-*-*-*-*-*-*-*
//*-*                ======================================
//
//     - rmin  radius of the inner circle in the cross-section
//
//     - rmax  radius of the outer circle in the cross-section
//
//     - z     z coordinate of the section



    if ((secNum < 0) || (secNum >= fNz)) return;

    fRmin[secNum] = rmin;
    fRmax[secNum] = rmax;
    fDz[secNum]   = z;
}


//______________________________________________________________________________
Int_t TPCON::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to a PCON*-*-*-*-*-*-*
//*-*            ===========================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each
//*-*  computed outline point of the PCON.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t n = GetNumberOfDivisions()+1;
   Int_t numPoints = fNz*2*n;
   return ShapeDistancetoPrimitive(numPoints,px,py);
}

//______________________________________________________________________________
void  TPCON::FillTableOfCoSin(Double_t phi, Double_t angstep,Int_t n)
{
  // File the table of cos and sin to prepare drawing
  Double_t ph = phi-angstep;
  for (Int_t j = 0; j < n; j++)
  {
    ph += angstep;
    fCoTab[j] = TMath::Cos(ph);
    fSiTab[j] = TMath::Sin(ph);
  }
}
//______________________________________________________________________________
void TPCON::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*Paint this 3-D shape with its current attributes*-*-*-*-*-*-*-*
//*-*            ================================================

   Int_t i, j;
   if (fNz < 2) return;
   const Int_t n = GetNumberOfDivisions()+1;

   Int_t numpoints = fNz*2*n;
   if (numpoints <= 0) return;
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   SetPoints(points);

   Bool_t rangeView = strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) PaintGLPoints(points);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t specialCase = kFALSE;

   if (fDphi1 == 360)           //mark this as a very special case, when
        specialCase = kTRUE;     //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(fNz*n-1+(specialCase == kTRUE));
        buff->numPolys  = 2*(fNz*n-1+(specialCase == kTRUE));
    }

//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    Int_t indx, indx2, k;
    indx = indx2 = 0;

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {

        //inside & outside circles, number of segments: 2*fNz*(n-1)
        //             special case number of segments: 2*fNz*n
        for (i = 0; i < fNz*2; i++) {
            indx2 = i*n;
            for (j = 1; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2+j;
            }
            if (specialCase) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2;
            }
        }

        //bottom & top lines, number of segments: 2*n
        for (i = 0; i < 2; i++) {
            indx2 = i*(fNz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside cilindres, number of segments: 2*(fNz-1)*n
        for (i = 0; i < (fNz-1); i++) {

            //inside cilinder
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside cilinder
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(fNz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (fNz-1); i++) {
                for (j = 0; j < 2; j++) {
                    buff->segs[indx++] = c;
                    buff->segs[indx++] =  2*i    * n + j*(n-1);
                    buff->segs[indx++] = (2*i+1) * n + j*(n-1);
                }
            }
        }
    }


    Int_t m = n - 1 + (specialCase == kTRUE);

//*-* Allocate memory for polygons *-*

    indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];

    if (buff->polys) {

        //bottom & top, number of polygons: 2*(n-1)
        // special case number of polygons: 2*n
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*fNz*m+i*n+j;
                buff->polys[indx++] = i*(fNz*2-2)*m+m+j;
                buff->polys[indx++] = 2*fNz*m+i*n+j+1;
                buff->polys[indx++] = i*(fNz*2-2)*m+j;
            }
            if (specialCase) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*fNz*m+i*n+j;
                buff->polys[indx++] = i*(fNz*2-2)*m+m+j;
                buff->polys[indx++] = 2*fNz*m+i*n;
                buff->polys[indx++] = i*(fNz*2-2)*m+j;
            }
        }


        //inside & outside, number of polygons: (fNz-1)*2*(n-1)
        for (k = 0; k < (fNz-1); k++) {
            for (i = 0; i < 2; i++) {
                for (j = 0; j < n-1; j++) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = fNz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = fNz*2*m+(2*k+i*1+2)*n+j+1;
                }
                if (specialCase) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = fNz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = fNz*2*m+(2*k+i*1+2)*n;
                }
            }
        }


        //left & right sections, number of polygons: 2*(fNz-1)
        //          special case number of polygons: 0
        if (!specialCase) {
            indx2 = fNz*2*(n-1);
            for (k = 0; k < (fNz-1); k++) {
                for (i = 0; i < 2; i++) {
                    buff->polys[indx++] = c+2;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*fNz*n+2*(k-1)+i;
                    buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                    buff->polys[indx++] = indx2+2*fNz*n+2*k+i;
                    buff->polys[indx++] = indx2+(2*k+3)*n+i*(n-1);
                }
            }
            buff->polys[indx-8] = indx2+n;
            buff->polys[indx-2] = indx2+2*n-1;
        }
    }

    //*-* Paint in the pad
    PaintShape(buff,rangeView);

    if (strstr(option, "x3d")) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}

//______________________________________________________________________________
void TPCON::PaintGLPoints(Float_t *vertex)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Paint BRIK via OpenGL *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            =====================

    gVirtualGL->PaintCone(vertex,-(GetNumberOfDivisions()+1),fNz);
}
//______________________________________________________________________________
void TPCON::SetNumberOfDivisions (Int_t p)
{
    if (GetNumberOfDivisions () == p) return;
    fNdiv=p;
    MakeTableOfCoSin();
}
//______________________________________________________________________________
void TPCON::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create PCON points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    Int_t i, j;
    Int_t indx = 0;

    if (buff) {

        Int_t n            = GetNumberOfDivisions()+1;

//*-* We've to checxk whether the table does exist and create it
//*-* since fCoTab/fSiTab are not saved with any TShape::Streamer function
        if (!fCoTab)   MakeTableOfCoSin();

        for (i = 0; i < fNz; i++)
        {
            for (j = 0; j < n; j++)
            {
                buff[indx++] = fRmin[i] * fCoTab[j];
                buff[indx++] = fRmin[i] * fSiTab[j];
                buff[indx++] = fDz[i];
            }
            for (j = 0; j < n; j++)
            {
                buff[indx++] = fRmax[i] * fCoTab[j];
                buff[indx++] = fRmax[i] * fSiTab[j];
                buff[indx++] = fDz[i];
            }
        }
    }
}

//______________________________________________________________________________
void TPCON::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total X3D size of this shape with its attributes*-*-*-*-*-*
//*-*          =======================================================

    Int_t n;

    n = GetNumberOfDivisions()+1;

    gSize3D.numPoints += fNz*2*n;
    gSize3D.numSegs   += 4*(fNz*n-1+(fDphi1 == 360));
    gSize3D.numPolys  += 2*(fNz*n-1+(fDphi1 == 360));
}


//_______________________________________________________________________
void TPCON::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TPCON::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TShape::Streamer(b);
      b >> fPhi1;
      b >> fDphi1;
      b >> fNz;
      fRmin  = new Float_t [fNz];
      fRmax  = new Float_t [fNz];
      fDz    = new Float_t [fNz];
      b.ReadArray(fRmin);
      b.ReadArray(fRmax);
      b.ReadArray(fDz);
      b >> fNdiv;
      b.CheckByteCount(R__s, R__c, TPCON::IsA());
      //====end of old versions
      
   } else {
      TPCON::Class()->WriteBuffer(b,this);
   }
}


