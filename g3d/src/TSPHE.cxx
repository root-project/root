// @(#)root/g3d:$Name:  $:$Id: TSPHE.cxx,v 1.2 2000/11/21 20:18:22 brun Exp $
// Author: Rene Brun   13/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSPHE.h"
#include "TNode.h"
#include "TView.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"


ClassImp(TSPHE)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/sphe.gif"> </P> End_Html
// SPHE is a Sphere. It has 9 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - rmin       minimum radius
//     - rmax       maximum radius
//     - themin     theta min
//     - themax     theta max
//     - phimin     phi min
//     - phimax     phi max

// ROOT color indx = max(i-i0,j-j0);


//______________________________________________________________________________
TSPHE::TSPHE()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*SPHE shape default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

  fRmin       = 0;
  fRmax       = 0;
  fThemin     = 0;
  fThemax     = 0;
  fPhimin     = 0;
  fPhimax     = 0;
  fSiTab      = 0;
  fCoTab      = 0;
  fCoThetaTab = 0;
  fNdiv       = 0;
  fAspectRatio=1.0;
  faX = faY = faZ = 1.0;      // Coeff along Ox
}


//______________________________________________________________________________
TSPHE::TSPHE(const char *name, const char *title, const char *material, Float_t rmin, Float_t rmax, Float_t themin,
             Float_t themax, Float_t phimin, Float_t phimax)
     : TShape(name, title,material)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*SPHE shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fRmin   = rmin;
    fRmax   = rmax;
    fThemin = themin;
    fThemax = themax;
    fPhimin = phimin;
    fPhimax = phimax;

    fSiTab      = 0;
    fCoTab      = 0;
    fCoThetaTab = 0;

    fAspectRatio=1.0;
    faX = faY = faZ = 1.0;      // Coeff along Ox

    SetNumberOfDivisions (20);
}


//______________________________________________________________________________
TSPHE::TSPHE(const char *name, const char *title, const char *material, Float_t rmax)
     : TShape(name, title,material)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*SPHE shape "simplified" constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===================================

    fRmin   = 0;
    fRmax   = rmax;
    fThemin = 0;
    fThemax = 180;
    fPhimin = 0;
    fPhimax = 360;

    fSiTab      = 0;
    fCoTab      = 0;
    fCoThetaTab = 0;

    fAspectRatio=1.0;
    faX = faY = faZ = 1.0;      // Coeff along Ox

    SetNumberOfDivisions (20);
}
//______________________________________________________________________________
TSPHE::~TSPHE()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*SPHE shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================
    if (fCoThetaTab)   delete [] fCoThetaTab;
    if (fSiTab) delete [] fSiTab;
    if (fCoTab) delete [] fCoTab;

    fCoTab = 0;
    fSiTab = 0;
    fCoThetaTab=0;

}
//______________________________________________________________________________
Int_t TSPHE::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to a PSPHE-*-*-*-*-*-*-*-*-*-*
//*-*            ===========================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each
//*-*  computed outline point of the PSPHE (stolen from PCON).
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t n = GetNumberOfDivisions()+1;
   Int_t numPoints = 2*n*(fNz+1);
   return ShapeDistancetoPrimitive(numPoints,px,py);
}
//______________________________________________________________________________
void TSPHE::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*Paint this 3-D shape with its current attributes*-*-*-*-*-*-*-*
//*-*            ================================================
   Int_t i, j;
   const Int_t n = GetNumberOfDivisions()+1;
   Int_t nz = fNz+1;
   Int_t numpoints = 2*n*nz;
   if (nz < 2) return;

  if (numpoints <= 0) return;
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) PaintGLPoints(points);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t specialCase = kFALSE;

   if (TMath::Abs(TMath::Sin(2*(fPhimax - fPhimin))) <= 0.01)  //mark this as a very special case, when
         specialCase = kTRUE;                                  //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
        buff->numPolys  = 2*(nz*n-1+(specialCase == kTRUE));
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

        //inside & outside spheres, number of segments: 2*nz*(n-1)
        //             special case number of segments: 2*nz*n
        for (i = 0; i < nz*2; i++) {
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
            indx2 = i*(nz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside spheres, number of segments: 2*(nz-1)*n
        for (i = 0; i < (nz-1); i++) {

            //inside sphere
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside sphere
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(nz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (nz-1); i++) {
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
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n+j+1;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
            if (specialCase) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
        }


        //inside & outside, number of polygons: (nz-1)*2*(n-1)
        for (k = 0; k < (nz-1); k++) {
            for (i = 0; i < 2; i++) {
                for (j = 0; j < n-1; j++) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j+1;
                }
                if (specialCase) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n;
                }
            }
        }


        //left & right sections, number of polygons: 2*(nz-1)
        //          special case number of polygons: 0
        if (!specialCase) {
            indx2 = nz*2*(n-1);
            for (k = 0; k < (nz-1); k++) {
                for (i = 0; i < 2; i++) {
                    buff->polys[indx++] = c+2;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*nz*n+2*(k-1)+i;
                    buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                    buff->polys[indx++] = indx2+2*nz*n+2*k+i;
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
void TSPHE::SetEllipse(const Float_t *factors){

  if (factors[0] > 0) faX = factors[0];
  if (factors[1] > 0) faY = factors[1];
  if (factors[2] > 0) faZ = factors[2];
//  MakeTableOfCoSin();
}

//______________________________________________________________________________
void TSPHE::SetNumberOfDivisions (Int_t p)
{

    if (GetNumberOfDivisions () == p) return;
    fNdiv=p;
    fNz = Int_t(fAspectRatio*fNdiv*(fThemax - fThemin )/(fPhimax - fPhimin )) + 1;
    MakeTableOfCoSin();
}

//______________________________________________________________________________
void TSPHE::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create SPHE points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================
    Int_t i, j;
    Int_t indx = 0;

    if (buff) {

        Int_t n            = GetNumberOfDivisions()+1;

//*-* We've to check whether the table does exist and create it
//*-* since fCoTab/fSiTab are not saved with any TShape::Streamer function
        if (!fCoTab)   MakeTableOfCoSin();

        Float_t z;
        for (i = 0; i < fNz+1; i++)
        {
            z = fRmin * fCoThetaTab[i]; // fSinPhiTab[i];
            Float_t sithet = TMath::Sqrt(TMath::Abs(1-fCoThetaTab[i]*fCoThetaTab[i]));
            Float_t zi = fRmin*sithet;
            for (j = 0; j < n; j++)
            {
                buff[indx++] = faX*zi * fCoTab[j];
                buff[indx++] = faY*zi * fSiTab[j];
                buff[indx++] = faZ*z;
            }
            z = fRmax * fCoThetaTab[i];
            zi = fRmax*sithet;
            for (j = 0; j < n; j++)
            {
                buff[indx++] = faX*zi * fCoTab[j];
                buff[indx++] = faY*zi * fSiTab[j];
                buff[indx++] = faZ*z;
            }
        }
    }
}
//______________________________________________________________________________
void TSPHE::MakeTableOfCoSin()
{
    const Double_t PI  = TMath::ATan(1) * 4.0;
    const Double_t ragrad  = PI/180.0;

    Float_t dphi = fPhimax - fPhimin;
    while (dphi > 360) dphi -= 360;

    Float_t dtet = fThemax - fThemin;
    while (dtet > 180) dtet -= 180;

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

    Double_t range   = Double_t(dphi * ragrad);
    Double_t phi1    = Double_t(fPhimin  * ragrad);
    Double_t angstep = range/(n-1);

    Double_t ph = phi1;
    for (j = 0; j < n; j++)
    {
        ph = phi1 + j*angstep;
        fCoTab[j] = TMath::Cos(ph);
        fSiTab[j] = TMath::Sin(ph);
    }

    n  = fNz + 1;

    if (fCoThetaTab)
        delete [] fCoThetaTab; // Delete the old tab if any
    fCoThetaTab = new Double_t [n];
    if (!fCoThetaTab ) return;

    range   = Double_t(dtet * ragrad);
    phi1    = Double_t(fThemin  * ragrad);
    angstep = range/(n-1);

    ph = phi1;
    for (j = 0; j < n; j++)
    {
        fCoThetaTab[n-j-1] = TMath::Cos(ph);
        ph += angstep;
    }

}

//_______________________________________________________________________
void TSPHE::PaintGLPoints(Float_t *vertex)
{
 gVirtualGL->PaintCone(vertex,-(GetNumberOfDivisions()+1),fNz+1);
}

//______________________________________________________________________________
void TSPHE::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total X3D size of this shape with its attributes*-*-*-*-*-*
//*-*          =======================================================

    Int_t n;

    n = GetNumberOfDivisions()+1;
    Int_t nz = fNz+1;
    Bool_t specialCase = kFALSE;

    if (TMath::Abs(TMath::Sin(2*(fPhimax - fPhimin))) <= 0.01)  //mark this as a very special case, when
          specialCase = kTRUE;                                  //we have to draw this PCON like a TUBE

    gSize3D.numPoints += 2*n*nz;
    gSize3D.numSegs   += 4*(nz*n-1+(specialCase == kTRUE));
    gSize3D.numPolys  += 2*(nz*n-1+(specialCase == kTRUE));
}

//_______________________________________________________________________
void TSPHE::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TSPHE::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         SetNumberOfDivisions (fNdiv);
         return;
      }
      //====process old versions before automatic schema evolution
      TShape::Streamer(b);
      b >> fRmin;    // minimum radius
      b >> fRmax;    // maximum radius
      b >> fThemin;  // minimum theta
      b >> fThemax;  // maximum theta
      b >> fPhimin;  // minimum phi
      b >> fPhimax;  // maximum phi
      Int_t tNdiv;   // XXX added by RvdE XXX (fNdiv is set by SetNumberOfDivisions)
      b >> tNdiv;
      if (R__v > 1) {
        b >> faX;
        b >> faY;
        b >> faZ;
      }
      SetNumberOfDivisions (tNdiv); // XXX added by RvdE
      b.CheckByteCount(R__s, R__c, TSPHE::IsA());
      //====end of old versions
      
   } else {
      TSPHE::Class()->WriteBuffer(b,this);
   }
}

