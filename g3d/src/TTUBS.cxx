// @(#)root/g3d:$Name$:$Id$
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
#include "TView.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"


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
//*-*-*-*-*-*-*-*-*-*-*-*TUBS shape default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================


}


//______________________________________________________________________________
TTUBS::TTUBS(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi1, Float_t phi2)
      : TTUBE(name,title,material,rmin,rmax,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TUBS shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fPhi1 = phi1;
    fPhi2 = phi2;
    MakeTableOfCoSin();
}

//______________________________________________________________________________
TTUBS::TTUBS(const char *name, const char *title, const char *material, Float_t rmax, Float_t dz,
               Float_t phi1, Float_t phi2)
      : TTUBE(name,title,material,rmax,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TUBS shape "simplified" constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===================================

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
    for (j = 0; j < n; j++)
    {
        ph = phi1 + j*angstep;
        fCoTab[j] = TMath::Cos(ph);
        fSiTab[j] = TMath::Sin(ph);
    }

}

//______________________________________________________________________________
TTUBS::~TTUBS()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TUBS shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================


}



//______________________________________________________________________________
Int_t TTUBS::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to a TUBE*-*-*-*-*-*-*
//*-*            ===========================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each
//*-*  computed outline point of the TUBE.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t n = GetNumberOfDivisions()+1;
   Int_t numPoints = n*4;
   return ShapeDistancetoPrimitive(numPoints,px,py);
}

//______________________________________________________________________________
void TTUBS::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*Paint this 3-D shape with its current attributes*-*-*-*-*-*-*-*
//*-*            ================================================

   Int_t i, j;
   const Int_t n = GetNumberOfDivisions()+1;
   const Int_t numpoints = 4*n;

   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) PaintGLPoints(points);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);


    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints =   numpoints;
        buff->numSegs   = 2*numpoints;
        buff->numPolys  =   numpoints-2;
    }

    buff->points = points;

    Int_t c = ((GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    buff->segs = new Int_t[buff->numSegs*3];
    memset(buff->segs, 0, buff->numSegs*3*sizeof(Int_t));
    if (buff->segs) {
        for (i = 0; i < 4; i++) {
            for (j = 1; j < n; j++) {
                buff->segs[(i*n+j-1)*3  ] = c;
                buff->segs[(i*n+j-1)*3+1] = i*n+j-1;
                buff->segs[(i*n+j-1)*3+2] = i*n+j;
            }
        }
        for (i = 4; i < 6; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c+1;
                buff->segs[(i*n+j)*3+1] = (i-4)*n+j;
                buff->segs[(i*n+j)*3+2] = (i-2)*n+j;
            }
        }
        for (i = 6; i < 8; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c;
                buff->segs[(i*n+j)*3+1] = 2*(i-6)*n+j;
                buff->segs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
            }
        }
    }

//*-* Allocate memory for polygons *-*

    Int_t indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];
    memset(buff->polys, 0, buff->numPolys*6*sizeof(Int_t));
    if (buff->polys) {
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = i*n+j;
                buff->polys[indx++] = (4+i)*n+j;
                buff->polys[indx++] = (2+i)*n+j;
                buff->polys[indx++] = (4+i)*n+j+1;
            }
        }
        for (i = 2; i < 4; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c+(i-2)*2+1;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = (i-2)*2*n+j;
                buff->polys[indx++] = (4+i)*n+j;
                buff->polys[indx++] = ((i-2)*2+1)*n+j;
                buff->polys[indx++] = (4+i)*n+j+1;
            }
        }
        buff->polys[indx++] = c+2;
        buff->polys[indx++] = 4;
        buff->polys[indx++] = 6*n;
        buff->polys[indx++] = 4*n;
        buff->polys[indx++] = 7*n;
        buff->polys[indx++] = 5*n;

        buff->polys[indx++] = c+2;
        buff->polys[indx++] = 4;
        buff->polys[indx++] = 7*n-1;
        buff->polys[indx++] = 5*n-1;
        buff->polys[indx++] = 8*n-1;
        buff->polys[indx++] = 6*n-1;
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
void TTUBS::PaintGLPoints(Float_t *vertex)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Paint BRIK via OpenGL *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            =====================
    gVirtualGL->PaintCone(vertex,-(GetNumberOfDivisions()+1),2);
}

//______________________________________________________________________________
void TTUBS::SetPoints(Float_t *buff)
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
//*-*-*-*-*-*-*Return total X3D size of this shape with its attributes*-*-*-*-*-*
//*-*          =======================================================

    Int_t n = GetNumberOfDivisions()+1;

    gSize3D.numPoints += n*4;
    gSize3D.numSegs   += n*8;
    gSize3D.numPolys  += n*4-2;
}

