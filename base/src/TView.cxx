// @(#)root/base:$Name:  $:$Id: TView.cxx,v 1.4 2000/08/18 13:44:34 brun Exp $
// Author: Rene Brun, Nenad Buncic, Evgueni Tcherniaev, Olivier Couet   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"
#include "TView.h"
#include "TVirtualX.h"
#include "TROOT.h"
#include "TList.h"
#include "TFile.h"

ClassImp(TView)

//const Int_t kPerspective = BIT(14);

const Int_t kCARTESIAN   = 1;
const Int_t kPOLAR       = 2;
const Double_t kRad = 3.14159265358979323846/180.0;

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*The V I E W class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =================                              *
//*-*                                                                     *
//*-*   This package was originally written by Evgueni Tcherniaev         *
//*-*   from IHEP/Protvino.                                               *
//*-*                                                                     *
//*-*   The original Fortran implementation was adapted to HIGZ/PAW       *
//*-*   by Olivier Couet and  Evgueni Tcherniaev.                         *
//*-*                                                                     *
//*-*   This View class is a subset of the original system                *
//*-*   It has been converted to a C++ class  by Rene Brun                *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


//____________________________________________________________________________
TView::TView()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*View default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ========================

   fOutline = 0;
   fDefaultOutline = kFALSE;
   fAutoRange      = kFALSE;
}

//____________________________________________________________________________
TView::~TView()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*View default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        =======================

   if (fOutline) fOutline->Delete();
   delete fOutline;
   fOutline = 0;
}

//____________________________________________________________________________
TView::TView(Int_t system)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*View constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ================
//*-*    Creates a 3-D view in the current pad
//*-*    By default pad range in 3-D view is (-1,-1,1,1), so ...
//*-*
//*-*    Before drawing a 3-D object in a pad, a 3-D view must be created.
//*-*    Note that a view is automatically created when drawing legos or surfaces.
//*-*
//*-*   The coordinate system is selected via system:
//*-*    system = 1  Cartesian
//*-*    system = 2  Polar
//*-*    system = 3  Cylindrical
//*-*    system = 4  Spherical
//*-*    system = 5  PseudoRapidity/Phi
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t irep;

   fSystem = system;
   fOutline = 0;
   fDefaultOutline = kFALSE;
   fAutoRange      = kFALSE;

   if (system == kCARTESIAN || system == kPOLAR) fPsi = 0;
   else fPsi = 90;

   //By default pad range in 3-D view is (-1,-1,1,1), so ...
   if (gPad) {
      gPad->Range(-1, -1, 1, 1);

      for (Int_t i = 0; i < 3; fRmin[i] = 0, fRmax[i] = 1, i++);

      fLongitude = -90 - gPad->GetPhi();
      fLatitude  =  90 - gPad->GetTheta();
      ResetView(fLongitude, fLatitude, fPsi, irep);

      gPad->SetView(this);
   }
}


//____________________________________________________________________________
TView::TView(Float_t *rmin, Float_t *rmax, Int_t system)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*View constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ================
//*-*    Creates a 3-D view in the current pad
//*-*  rmin[3], rmax[3] are the limits of the object depending on
//*-*  the selected coordinate system
//*-*
//*-*   Before drawing a 3-D object in a pad, a 3-D view must be created.
//*-*   Note that a view is automatically created when drawing legos or surfaces.
//*-*
//*-*   The coordinate system is selected via system:
//*-*    system = 1  Cartesian
//*-*    system = 2  Polar
//*-*    system = 3  Cylindrical
//*-*    system = 4  Spherical
//*-*    system = 5  PseudoRapidity/Phi
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t irep;

   fSystem = system;
   fOutline = 0;
   fDefaultOutline = kFALSE;

   if (system == kCARTESIAN || system == kPOLAR) fPsi = 0;
   else fPsi = 90;

   //By default pad range in 3-D view is (-1,-1,1,1), so ...
   gPad->Range(-1, -1, 1, 1);

   for (Int_t i = 0; i < 3; fRmin[i] = rmin[i], fRmax[i] = rmax[i], i++);

        fLongitude = -90 - gPad->GetPhi();
        fLatitude  =  90 - gPad->GetTheta();
        ResetView(fLongitude, fLatitude, fPsi, irep);

   if (gPad)
                gPad->SetView(this);
}


//____________________________________________________________________________
TView::TView(Double_t *rmin, Double_t *rmax, Int_t system)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*View constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ================
//*-*    Creates a 3-D view in the current pad
//*-*  rmin[3], rmax[3] are the limits of the object depending on
//*-*  the selected coordinate system
//*-*
//*-*   Before drawing a 3-D object in a pad, a 3-D view must be created.
//*-*   Note that a view is automatically created when drawing legos or surfaces.
//*-*
//*-*   The coordinate system is selected via system:
//*-*    system = 1  Cartesian
//*-*    system = 2  Polar
//*-*    system = 3  Cylindrical
//*-*    system = 4  Spherical
//*-*    system = 5  PseudoRapidity/Phi
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t irep;

   fSystem = system;
   fOutline = 0;
   fDefaultOutline = kFALSE;

   if (system == kCARTESIAN || system == kPOLAR) fPsi = 0;
   else fPsi = 90;

   //By default pad range in 3-D view is (-1,-1,1,1), so ...
   gPad->Range(-1, -1, 1, 1);

   for (Int_t i = 0; i < 3; fRmin[i] = rmin[i], fRmax[i] = rmax[i], i++);

        fLongitude = -90 - gPad->GetPhi();
        fLatitude  =  90 - gPad->GetTheta();
        ResetView(fLongitude, fLatitude, fPsi, irep);

   if (gPad)
                gPad->SetView(this);
}


//______________________________________________________________________________
void TView::AxisVertex(Double_t ang, Double_t *av, Int_t &ix1, Int_t &ix2, Int_t &iy1, Int_t &iy2, Int_t &iz1, Int_t &iz2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Define axis  vertices*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================                          *
//*-*                                                                     *
//*-*    Input  ANG     - angle between X and Y axis                      *
//*-*                                                                     *
//*-*    Output: AV(3,8) - axis vertixes                                  *
//*-*            IX1     - 1st point of X-axis                            *
//*-*            IX2     - 2nd point of X-axis                            *
//*-*            IY1     - 1st point of Y-axis                            *
//*-*            IY2     - 2nd point of Y-axis                            *
//*-*            IZ1     - 1st point of Z-axis                            *
//*-*            IZ2     - 2nd point of Z-axis                            *
//*-*                                                                     *
//*-*                 8                        6                          *
//*-*                / \                      /|\                         *
//*-*             5 /   \ 7                5 / | \ 7                      *
//*-*              |\   /|                  |  |  |                       *
//*-*  THETA < 90  | \6/ |     THETA > 90   | /2\ |                       *
//*-*  (Top view)  |  |  |   (Bottom view)  |/   \|                       *
//*-*             1 \ | /3                 1 \   /3                       *
//*-*                \|/                      \ /                         *
//*-*                 2                        4                          *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    /* Local variables */
    Double_t cosa, sina;
    Int_t i, k;
    Double_t p[8]        /* was [2][4] */;
    Int_t i1, i2, i3, i4, ix, iy;
    ix = 0;

    /* Parameter adjustments */
    av -= 4;

    sina = TMath::Sin(ang*kRad);
    cosa = TMath::Cos(ang*kRad);
    p[0] = fRmin[0];
    p[1] = fRmin[1];
    p[2] = fRmax[0];
    p[3] = fRmin[1];
    p[4] = fRmax[0];
    p[5] = fRmax[1];
    p[6] = fRmin[0];
    p[7] = fRmax[1];
//*-*-           F I N D   T H E   M O S T   L E F T   P O I N T */
    i1 = 1;
    if (fTN[0] < 0) i1 = 2;
    if (fTN[0]*cosa + fTN[1]*sina < 0) i1 = 5 - i1;

//*-*-          S E T   O T H E R   P O I N T S */
    i2 = i1 % 4 + 1;
    i3 = i2 % 4 + 1;
    i4 = i3 % 4 + 1;

//*-*-          S E T   A X I S   V E R T I X E S */

    av[4] = p[(i1 << 1) - 2];
    av[5] = p[(i1 << 1) - 1];
    av[7] = p[(i2 << 1) - 2];
    av[8] = p[(i2 << 1) - 1];
    av[10] = p[(i3 << 1) - 2];
    av[11] = p[(i3 << 1) - 1];
    av[13] = p[(i4 << 1) - 2];
    av[14] = p[(i4 << 1) - 1];
    for (i = 1; i <= 4; ++i) {
        av[i*3 +  3] = fRmin[2];
        av[i*3 + 13] = av[i*3 + 1];
        av[i*3 + 14] = av[i*3 + 2];
        av[i*3 + 15] = fRmax[2];
    }

//*-*-          S E T   A X I S

    if (av[4] == av[7]) ix = 2;
    if (av[5] == av[8]) ix = 1;
    iy = 3 - ix;
//*-*-          X - A X I S
    ix1 = ix;
    if (av[ix*3 + 1] > av[(ix + 1)*3 + 1])      ix1 = ix + 1;
    ix2 = (ix << 1) - ix1 + 1;
//*-*-          Y - A X I S
    iy1 = iy;
    if (av[iy*3 + 2] > av[(iy + 1)*3 + 2])      iy1 = iy + 1;
    iy2 = (iy << 1) - iy1 + 1;
//*-*-          Z - A X I S
    iz1 = 1;
    iz2 = 5;

    if (fTN[10] >= 0)   return;
    k = (ix1 - 1)*3 + ix2;
    if (k%2) return;
    if (k == 2) {
       ix1 = 4;
       ix2 = 3;
    }
    if (k == 4) {
       ix1 = 3;
       ix2 = 4;
    }
    if (k == 6) {
       ix1 = 1;
       ix2 = 4;
    }
    if (k == 8) {
       ix1 = 4;
       ix2 = 1;
    }

    k = (iy1 - 1)*3 + iy2;
    if (k%2) return;
    if (k == 2) {
       iy1 = 4;
       iy2 = 3;
       return;
    }
    if (k == 4) {
       iy1 = 3;
       iy2 = 4;
       return;
    }
    if (k == 6) {
       iy1 = 1;
       iy2 = 4;
       return;
    }
    if (k == 8) {
       iy1 = 4;
       iy2 = 1;
    }
}



//______________________________________________________________________________
void TView::DefineViewDirection(Double_t *s, Double_t *c,
                                Double_t cosphi, Double_t sinphi,
                                Double_t costhe, Double_t sinthe,
                                Double_t cospsi, Double_t sinpsi,
                                Double_t *tnorm, Double_t *tback)
{
//*-*-*-*-*-*-*-*-*Define view direction (in spherical coordinates)-*-*-*-*
//*-*              ================================================       *
//*-*                                                                     *
//*-*              Compute transformation matrix from world coordinates   *
//*-*              to normalised coordinates (-1 to +1)                   *
//*-*                                                                     *
//*-*    Input: S(3)    - scale factors                                   *
//*-*           C(3)    - centre of scope                                 *
//*-*           COSPHI  - longitude COS                                   *
//*-*           SINPHI  - longitude SIN                                   *
//*-*           COSTHE  - latitude COS (angle between +Z and view direc.) *
//*-*           SINTHE  - latitude SIN                                    *
//*-*           COSPSI  - screen plane rotation angle COS                 *
//*-*           SINPSI  - screen plane rotation angle SIN                 *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, k;
    Double_t tran[16]   /* was [4][4] */, rota[16]      /* was [4][4] */;
    Double_t c1, c2, c3, s1, s2, s3, scalex, scaley, scalez;

    /* Parameter adjustments */
    tback -= 5;
    tnorm -= 5;

    scalex = s[0];
    scaley = s[1];
    scalez = s[2];

//*-*-        S E T   T R A N S L A T I O N   M A T R I X

    tran[0] = 1 / scalex;
    tran[1] = 0;
    tran[2] = 0;
    tran[3] = -c[0] / scalex;

    tran[4] = 0;
    tran[5] = 1 / scaley;
    tran[6] = 0;
    tran[7] = -c[1] / scaley;

    tran[8] = 0;
    tran[9] = 0;
    tran[10] = 1 / scalez;
    tran[11] = -c[2] / scalez;

    tran[12] = 0;
    tran[13] = 0;
    tran[14] = 0;
    tran[15] = 1;

//*-*-        S E T    R O T A T I O N   M A T R I X

//    ( C(PSI) S(PSI) 0)   (1      0          0 )   ( C(90+PHI) S(90+PHI) 0)
//    (-S(PSI) C(PSI) 0) * (0  C(THETA) S(THETA)) * (-S(90+PHI) C(90+PHI) 0)
//    (   0      0    1)   (0 -S(THETA) C(THETA))   (     0           0   1)

    c1 = cospsi;
    s1 = sinpsi;
    c2 = costhe;
    s2 = sinthe;
    c3 = -sinphi;
    s3 = cosphi;

    rota[0] = c1*c3 - s1*c2*s3;
    rota[1] = c1*s3 + s1*c2*c3;
    rota[2] = s1*s2;
    rota[3] = 0;

    rota[4] = -s1*c3 - c1* c2*s3;
    rota[5] = -s1*s3 + c1* c2*c3;
    rota[6] = c1*s2;
    rota[7] = 0;

    rota[8] = s2*s3;
    rota[9] = -s2*c3;
    rota[10] = c2;
    rota[11] = 0;

    rota[12] = 0;
    rota[13] = 0;
    rota[14] = 0;
    rota[15] = 1;

//*-*-        F I N D   T R A N S F O R M A T I O N   M A T R I X

    for (i = 1; i <= 3; ++i) {
        for (k = 1; k <= 4; ++k) {
            tnorm[k + (i << 2)] = rota[(i << 2) - 4]*tran[k - 1] + rota[(i
                    << 2) - 3]*tran[k + 3] + rota[(i << 2) - 2]*tran[k +7]
                    + rota[(i << 2) - 1]*tran[k + 11];
        }
    }

//*-*-        S E T   B A C K   T R A N S L A T I O N   M A T R I X

    tran[0] = scalex;
    tran[3] = c[0];

    tran[5] = scaley;
    tran[7] = c[1];

    tran[10] = scalez;
    tran[11] = c[2];

//*-*-        F I N D   B A C K   T R A N S F O R M A T I O N

    for (i = 1; i <= 3; ++i) {
        for (k = 1; k <= 4; ++k) {
            tback[k + (i << 2)] = tran[(i << 2) - 4]*rota[(k << 2) - 4] +
                    tran[(i << 2) - 3]*rota[(k << 2) - 3] + tran[(i << 2) -2]
                    *rota[(k << 2) - 2] + tran[(i << 2) - 1]*rota[(k <<2) - 1];
        }
    }
}



//______________________________________________________________________________
void TView::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*-*-*-*-*-*

   ExecuteRotateView(event,px,py);
}

//______________________________________________________________________________
void TView::ExecuteRotateView(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*-*-*-*-*-*-*
//*-*              =========================================                    *
//*-*  This member function is called when a object is clicked with the locator *
//*-*                                                                           *
//*-*  If Left button clicked in the object area, while the button is kept down *
//*-*  the cube representing the surrounding frame for the corresponding        *
//*-*  new latitude and longitude position is drawn.                                         *
//*-*                                                                           *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   static Int_t system, framewasdrawn;
   static Double_t xrange, yrange, xmin, ymin, longitude1, latitude1, longitude2, latitude2;
   static Double_t newlatitude, newlongitude, oldlatitude, oldlongitude;
   Double_t dlatitude, dlongitude, x, y;
   Int_t irep = 0;
   Double_t psideg;

   // all coordinate transformation are from absolute to relative

   if (!gPad->IsEditable()) return;
   gPad->AbsCoordinates(kTRUE);

   switch (event) {

   case kKeyPress :
       MoveViewCommand(Char_t(px), py);
       break;
   case kMouseMotion:
      gPad->SetCursor(kRotate);
      break;

   case kButton1Down:

//*-*-    Remember position of the cube
      xmin   = gPad->GetX1();
      ymin   = gPad->GetY1();
      xrange = gPad->GetX2() - xmin;
      yrange = gPad->GetY2() - ymin;
      x      = gPad->PixeltoX(px);
      y      = gPad->PixeltoY(py);
      system = GetSystem();
      framewasdrawn = 0;
      if (system == kCARTESIAN || system == kPOLAR) {
         longitude1 = 180*(x-xmin)/xrange;
         latitude1  =  90*(y-ymin)/yrange;
      } else {
         latitude1  =  90*(x-xmin)/xrange;
         longitude1 = 180*(y-ymin)/yrange;
      }
      newlongitude = oldlongitude = -90 - gPad->GetPhi();
      newlatitude  = oldlatitude  =  90 - gPad->GetTheta();
      psideg       = GetPsi();

      //if outline isn't set, make it look like a cube
      if(!fOutline)
          SetOutlineToCube();
      break;

   case kButton1Motion:

//*-*-    Draw the surrounding frame for the current mouse position
//*-*-       First: Erase old frame
      if (framewasdrawn) fOutline->Paint();
      framewasdrawn = 1;
      x = gPad->PixeltoX(px);
      y = gPad->PixeltoY(py);
      if (system == kCARTESIAN || system == kPOLAR) {
         longitude2 = 180*(x-xmin)/xrange;
         latitude2  =  90*(y-ymin)/yrange;
      } else {
         latitude2  =  90*(x-xmin)/xrange;
         longitude2 = 180*(y-ymin)/yrange;
      }
      dlongitude   = longitude2   - longitude1;
      dlatitude    = latitude2    - latitude1;
      newlatitude  = oldlatitude  + dlatitude;
      newlongitude = oldlongitude - dlongitude;
      psideg       = GetPsi();
      ResetView(newlongitude, newlatitude, psideg, irep);
      fOutline->Paint();
      break;

   case kButton1Up:

//*-*-   Recompute new view matrix and redraw

      psideg = GetPsi();
      SetView(newlongitude, newlatitude, psideg, irep);
      gPad->SetPhi(-90-newlongitude);
      gPad->SetTheta(90-newlatitude);
      gPad->Modified(kTRUE);

//*-*-    Set line color, style and width
      gVirtualX->SetLineColor(-1);
      gVirtualX->SetLineStyle(-1);
      gVirtualX->SetLineWidth(-1);
      break;
   }

   // set back to default transformation mode
   gPad->AbsCoordinates(kFALSE);
}


//______________________________________________________________________________
void TView::FindNormal(Double_t x, Double_t  y, Double_t z, Double_t &zn)
{
//*-*-*-*-*-*-*Find Z component of NORMAL in normalized coordinates-*-*-*-*
//*-*          ====================================================       *
//*-*                                                                     *
//*-*    Input: X - X-component of NORMAL                                 *
//*-*           Y - Y-component of NORMAL                                 *
//*-*           Z - Z-component of NORMAL                                 *
//*-*                                                                     *
//*-*    Output: ZN - Z-component of NORMAL in normalized coordinates     *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    zn = x*(fTN[1] * fTN[6] - fTN[2] * fTN[5]) + y*(fTN[2] * fTN[4] -
            fTN[0] * fTN[6]) + z*(fTN[0] * fTN[5] - fTN[1] * fTN[4]);
}

//______________________________________________________________________________
void TView::FindPhiSectors(Int_t iopt, Int_t &kphi, Double_t *aphi, Int_t &iphi1, Int_t &iphi2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Find critical PHI sectors*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =========================                      *
//*-*                                                                     *
//*-*    Input: IOPT    - options: 1 - from BACK to FRONT 'BF'            *
//*-*                              2 - from FRONT to BACK 'FB'            *
//*-*           KPHI    - number of phi sectors                           *
//*-*           APHI(*) - PHI separatrices                                *
//*-*                                                                     *
//*-*    Output: IPHI1  - initial sector                                  *
//*-*            IPHI2  - final sector                                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t iphi[2], i, k;
    Double_t dphi;
    Double_t x1, x2, z1, z2, phi1, phi2;

    /* Parameter adjustments */
    --aphi;

    if (aphi[kphi + 1] == aphi[1]) aphi[kphi + 1] += 360;
    dphi = TMath::Abs(aphi[kphi + 1] - aphi[1]);
    if (dphi != 360) {
        aphi[kphi + 2] = (aphi[1] + aphi[kphi + 1]) / (float)2. + 180;
        aphi[kphi + 3] = aphi[1] + 360;
        kphi += 2;
    }

//*-*-       F I N D   C R I T I C A L   S E C T O R S

    k = 0;
    for (i = 1; i <= kphi; ++i) {
        phi1 = kRad*aphi[i];
        phi2 = kRad*aphi[i + 1];
        x1 = fTN[0]*TMath::Cos(phi1) + fTN[1]*TMath::Sin(phi1);
        x2 = fTN[0]*TMath::Cos(phi2) + fTN[1]*TMath::Sin(phi2);
        if (x1 >= 0 && x2 > 0) continue;
        if (x1 <= 0 && x2 < 0) continue;
        ++k;
        if (k == 3) break;
        iphi[k - 1] = i;
    }
    if (k != 2) {
       Error("FindPhiSectors", "something strange: num. of critical sector not equal 2");
       iphi1 = 1;
       iphi2 = 2;
       return;
    }

//*-*-       F I N D   O R D E R   O F   C R I T I C A L   S E C T O R S

    phi1 = kRad*(aphi[iphi[0]] + aphi[iphi[0] + 1]) / (float)2.;
    phi2 = kRad*(aphi[iphi[1]] + aphi[iphi[1] + 1]) / (float)2.;
    z1 = fTN[8]*TMath::Cos(phi1) + fTN[9]*TMath::Sin(phi1);
    z2 = fTN[8]*TMath::Cos(phi2) + fTN[9]*TMath::Sin(phi2);
    if (z1 <= z2 && iopt == 1 || z1 > z2 && iopt == 2) {
        iphi1 = iphi[0];
        iphi2 = iphi[1];
    } else {
        iphi1 = iphi[1];
        iphi2 = iphi[0];
    }
}

//______________________________________________________________________________
void TView::FindThetaSectors(Int_t iopt, Double_t phi, Int_t &kth, Double_t *ath, Int_t &ith1, Int_t &ith2)
{
//*-*-*-*-*-*-*-Find critical THETA sectors for given PHI sector*-*-*-*-*-*
//*-*           ================================================          *
//*-*                                                                     *
//*-*    Input: IOPT        - options: 1 - from BACK to FRONT 'BF'        *
//*-*                                  2 - from FRONT to BACK 'FB'        *
//*-*           PHI         - PHI sector                                  *
//*-*           KTH         - number of THETA sectors                     *
//*-*           ATH(*)      - THETA separatrices                          *
//*-*                                                                     *
//*-*    Output: ITH1  - initial sector                                   *
//*-*            ITH2  - final sector                                     *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, k, ith[2];
    Double_t z1, z2, cosphi, sinphi, tncons, th1, th2, dth;

    /* Parameter adjustments */
    --ath;

    /* Function Body */
    dth = TMath::Abs(ath[kth + 1] - ath[1]);
    if (dth != 360) {
        ath[kth + 2] = 0.5*(ath[1] + ath[kth + 1]) + 180;
        ath[kth + 3] = ath[1] + 360;
        kth += 2;
    }

//*-*-       F I N D   C R I T I C A L   S E C T O R S

    cosphi = TMath::Cos(phi*kRad);
    sinphi = TMath::Sin(phi*kRad);
    k = 0;
    for (i = 1; i <= kth; ++i) {
        th1 = kRad*ath[i];
        th2 = kRad*ath[i + 1];
        FindNormal(TMath::Cos(th1)*cosphi, TMath::Cos(th1)*sinphi, -TMath::Sin(th1), z1);
        FindNormal(TMath::Cos(th2)*cosphi, TMath::Cos(th2)*sinphi, -TMath::Sin(th2), z2);
        if (z1 >= 0 && z2 > 0) continue;
        if (z1 <= 0 && z2 < 0) continue;
        ++k;
        if (k == 3) break;
        ith[k - 1] = i;
    }
    if (k != 2) {
       Error("FindThetaSectors", "Something strange: num. of critical sectors not equal 2");
       ith1 = 1;
       ith2 = 2;
       return;
    }

//*-*-       F I N D   O R D E R   O F   C R I T I C A L   S E C T O R S

    tncons = fTN[8]*TMath::Cos(phi*kRad) + fTN[9]*TMath::Sin(phi*kRad);
    th1    = kRad*(ath[ith[0]] + ath[ith[0] + 1]) / (float)2.;
    th2    = kRad*(ath[ith[1]] + ath[ith[1] + 1]) / (float)2.;
    z1     = tncons*TMath::Sin(th1) + fTN[10]*TMath::Cos(th1);
    z2     = tncons*TMath::Sin(th2) + fTN[10]*TMath::Cos(th2);
    if (z1 <= z2 && iopt == 1 || z1 > z2 && iopt == 2) {
        ith1 = ith[0];
        ith2 = ith[1];
    } else {
        ith1 = ith[1];
        ith2 = ith[0];
    }
}


//______________________________________________________________________________
void TView::FindScope(Double_t *scale, Double_t *center, Int_t &irep)
{
//*-*-*-*-*-*-*-*Find centre of a MIN-MAX scope and scale factors-*-*-*-*-*
//*-*            ================================================         *
//*-*                                                                     *
//*-*    Output: SCALE(3)  - scale factors                                *
//*-*            CENTER(3) - centre                                       *
//*-*            IREP      - reply (-1 if error in min-max)               *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    irep = 0;
    Double_t sqrt3 = 0.5*TMath::Sqrt(3.0);

    for (Int_t i = 0; i < 3; i++) {
        if (fRmin[i] >= fRmax[i]) { irep = -1; return;}
                scale[i]  = sqrt3*(fRmax[i] - fRmin[i]);
                center[i] = 0.5*(fRmax[i] + fRmin[i]);
    }
}

//______________________________________________________________________________
Int_t TView::GetDistancetoAxis(Int_t axis, Int_t px, Int_t py, Double_t &ratio)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Return distance to axis from point px,py*-*-*-*
//*-*                      ========================================
//*-*
//*-*
//*-*  Algorithm:
//*-*
//*-*    A(x1,y1)         P                             B(x2,y2)
//*-*    ------------------------------------------------
//*-*                     I
//*-*                     I
//*-*                     I
//*-*                     I
//*-*                    M(x,y)
//*-*
//*-*  Let us call  a = distance AM     A=a**2
//*-*               b = distance BM     B=b**2
//*-*               c = distance AB     C=c**2
//*-*               d = distance PM     D=d**2
//*-*               u = distance AP     U=u**2
//*-*               v = distance BP     V=v**2     c = u + v
//*-*
//*-*  D = A - U
//*-*  D = B - V  = B -(c-u)**2
//*-*     ==> u = (A -B +C)/2c
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Double_t x1,y1,x2,y2;
   Double_t x     = px;
   Double_t y     = py;
   ratio = 0;

   if (fSystem != 1) return 9998; // only implemented for Cartesian coordinates
   if (axis == 1) {
      x1 = gPad->XtoAbsPixel(fX1[0]);
      y1 = gPad->YtoAbsPixel(fX1[1]);
      x2 = gPad->XtoAbsPixel(fX2[0]);
      y2 = gPad->YtoAbsPixel(fX2[1]);
   } else if (axis == 2) {
      x1 = gPad->XtoAbsPixel(fY1[0]);
      y1 = gPad->YtoAbsPixel(fY1[1]);
      x2 = gPad->XtoAbsPixel(fY2[0]);
      y2 = gPad->YtoAbsPixel(fY2[1]);
   } else {
      x1 = gPad->XtoAbsPixel(fZ1[0]);
      y1 = gPad->YtoAbsPixel(fZ1[1]);
      x2 = gPad->XtoAbsPixel(fZ2[0]);
      y2 = gPad->YtoAbsPixel(fZ2[1]);
   }
   Double_t xx1   = x  - x1;
   Double_t xx2   = x  - x2;
   Double_t x1x2  = x1 - x2;
   Double_t yy1   = y  - y1;
   Double_t yy2   = y  - y2;
   Double_t y1y2  = y1 - y2;
   Double_t A     = xx1*xx1   + yy1*yy1;
   Double_t B     = xx2*xx2   + yy2*yy2;
   Double_t C     = x1x2*x1x2 + y1y2*y1y2;
   if (C <= 0) return 9999;
   Double_t c     = TMath::Sqrt(C);
   Double_t u     = (A - B + C)/(2*c);
   Double_t D     = TMath::Abs(A - u*u);

   Int_t dist = Int_t(TMath::Sqrt(D) - 0.5);
   ratio = u/c;
   return dist;
}


//______________________________________________________________________________
void TView::GetRange(Float_t *min, Float_t *max)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Get Range function-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================
//*-*
        for (Int_t i = 0; i < 3; max[i] = fRmax[i], min[i] = fRmin[i], i++);
}


//______________________________________________________________________________
void TView::GetRange(Double_t *min, Double_t *max)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Get Range function-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================
//*-*
        for (Int_t i = 0; i < 3; max[i] = fRmax[i], min[i] = fRmin[i], i++);
}

//______________________________________________________________________________
void TView::NDCtoWC(Float_t* pn, Float_t* pw)
{
//*-*-*-*-*-*-*Transfer point from normalized to world coordinates*-*-*-*-*
//*-*          ===================================================        *
//*-*                                                                     *
//*-*    Input: PN(3) - point in world coordinate system                  *
//*-*           PW(3) - point in normalized coordinate system             *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  pw[0] = fTback[0]*pn[0] + fTback[1]*pn[1] + fTback[2]*pn[2]  + fTback[3];
  pw[1] = fTback[4]*pn[0] + fTback[5]*pn[1] + fTback[6]*pn[2]  + fTback[7];
  pw[2] = fTback[8]*pn[0] + fTback[9]*pn[1] + fTback[10]*pn[2] + fTback[11];
}

//______________________________________________________________________________
void TView::NDCtoWC(Double_t* pn, Double_t* pw)
{
//*-*-*-*-*-*-*Transfer point from normalized to world coordinates*-*-*-*-*
//*-*          ===================================================        *
//*-*                                                                     *
//*-*    Input: PN(3) - point in world coordinate system                  *
//*-*           PW(3) - point in normalized coordinate system             *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  pw[0] = fTback[0]*pn[0] + fTback[1]*pn[1] + fTback[2]*pn[2]  + fTback[3];
  pw[1] = fTback[4]*pn[0] + fTback[5]*pn[1] + fTback[6]*pn[2]  + fTback[7];
  pw[2] = fTback[8]*pn[0] + fTback[9]*pn[1] + fTback[10]*pn[2] + fTback[11];
}

//______________________________________________________________________________
void TView::NormalWCtoNDC(Float_t *pw, Float_t *pn)
{
//*-*-*Transfer vector of NORMAL from word to normalized coodinates-*-*-*-*
//*-*  ============================================================
//*-*
//*-*    Input: PW(3) - vector of NORMAL in word coordinate system
//*-*           PN(3) - vector of NORMAL in normalized coordinate system
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t x, y, z, a1, a2, a3, b1, b2, b3, c1, c2, c3;

    x = pw[0];
    y = pw[1];
    z = pw[2];
    a1 = fTnorm[0];
    a2 = fTnorm[1];
    a3 = fTnorm[2];
    b1 = fTnorm[4];
    b2 = fTnorm[5];
    b3 = fTnorm[6];
    c1 = fTnorm[8];
    c2 = fTnorm[9];
    c3 = fTnorm[10];
    pn[0] = x*(b2*c3 - b3*c2) + y*(b3*c1 - b1*c3) + z*(b1*c2 - b2*c1);
    pn[1] = x*(c2*a3 - c3*a2) + y*(c3*a1 - c1*a3) + z*(c1*a2 - c2*a1);
    pn[2] = x*(a2*b3 - a3*b2) + y*(a3*b1 - a1*b3) + z*(a1*b2 - a2*b1);
}

//______________________________________________________________________________
void TView::NormalWCtoNDC(Double_t *pw, Double_t *pn)
{
//*-*-*Transfer vector of NORMAL from word to normalized coodinates-*-*-*-*
//*-*  ============================================================
//*-*
//*-*    Input: PW(3) - vector of NORMAL in word coordinate system
//*-*           PN(3) - vector of NORMAL in normalized coordinate system
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t x, y, z, a1, a2, a3, b1, b2, b3, c1, c2, c3;

    x = pw[0];
    y = pw[1];
    z = pw[2];
    a1 = fTnorm[0];
    a2 = fTnorm[1];
    a3 = fTnorm[2];
    b1 = fTnorm[4];
    b2 = fTnorm[5];
    b3 = fTnorm[6];
    c1 = fTnorm[8];
    c2 = fTnorm[9];
    c3 = fTnorm[10];
    pn[0] = x*(b2*c3 - b3*c2) + y*(b3*c1 - b1*c3) + z*(b1*c2 - b2*c1);
    pn[1] = x*(c2*a3 - c3*a2) + y*(c3*a1 - c1*a3) + z*(c1*a2 - c2*a1);
    pn[2] = x*(a2*b3 - a3*b2) + y*(a3*b1 - a1*b3) + z*(a1*b2 - a2*b1);
}

//______________________________________________________________________________
void TView::PadRange(Double_t rback)
{
//*-*-*-*-*Set the correct window size for lego and surface plots*-*-*-*-*
//*-*      ======================================================
//*-*
//*-*  Set the correct window size for lego and surface plots.
//*-*  And draw the background if necessary.
//*-*
//*-*    Input parameters:
//*-*
//*-*   RBACK : Background colour
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, k;
    Double_t x, y, z, r1, r2, r3, xx, yy, smax[2];
    Double_t xgraf[6], ygraf[6];

    for (i = 1; i <= 2; ++i) {
        smax[i - 1] = fTnorm[(i << 2) - 1];
        for (k = 1; k <= 3; ++k) {
            if (fTnorm[k + (i << 2) - 5] < 0) {
                smax[i - 1] += fTnorm[k + (i << 2) - 5]*fRmin[k-1];
            } else {
                smax[i - 1] += fTnorm[k + (i << 2) - 5]*fRmax[k-1];
            }
        }
    }

//*-*- Compute x,y range
   Double_t xmin = -smax[0];
   Double_t xmax = smax[0];
   Double_t ymin = -smax[1];
   Double_t ymax = smax[1];
   Double_t dx   = xmax-xmin;
   Double_t dy   = ymax-ymin;
   Double_t dxr  = dx/(1 - gPad->GetLeftMargin() - gPad->GetRightMargin());
   Double_t dyr  = dy/(1 - gPad->GetBottomMargin() - gPad->GetTopMargin());

   // Range() could change the size of the pad pixmap and therefore should
   // be called before the other paint routines
   gPad->Range(xmin - dxr*gPad->GetLeftMargin(),
                      ymin - dyr*gPad->GetBottomMargin(),
                      xmax + dxr*gPad->GetRightMargin(),
                      ymax + dyr*gPad->GetTopMargin());
   gPad->RangeAxis(xmin, ymin, xmax, ymax);

//*-*-             Draw the background if necessary

    if (rback > 0) {
        r1 = -1;
        r2 = -1;
        r3 = -1;
        xgraf[1] = -smax[0];
        xgraf[2] = -smax[0];
        xgraf[4] =  smax[0];
        xgraf[5] =  smax[0];
        ygraf[0] = -smax[1];
        ygraf[3] =  smax[1];
        ygraf[1] =  smax[1];
        ygraf[2] = -smax[1];
        ygraf[5] =  smax[1];
        ygraf[4] = -smax[1];
        for (i = 1; i <= 8; ++i) {
            x = 0.5*((1 - r1)*fRmin[0] + (r1 + 1)*fRmax[0]);
            y = 0.5*((1 - r2)*fRmin[1] + (r2 + 1)*fRmax[1]);
            z = 0.5*((1 - r3)*fRmin[2] + (r3 + 1)*fRmax[2]);
            xx = fTnorm[0]*x + fTnorm[1]*y + fTnorm[2]*z + fTnorm[3];
            yy = fTnorm[4]*x + fTnorm[5]*y + fTnorm[6]*z + fTnorm[7];
            if (TMath::Abs(xx - xgraf[1]) <= 1e-4) {
                if (ygraf[1] >= yy) ygraf[1] = yy;
                if (ygraf[2] <= yy) ygraf[2] = yy;
            }
            if (TMath::Abs(xx - xgraf[5]) <= 1e-4) {
                if (ygraf[5] >= yy) ygraf[5] = yy;
                if (ygraf[4] <= yy) ygraf[4] = yy;
            }
            if (TMath::Abs(yy - ygraf[0]) <= 1e-4) xgraf[0] = xx;
            if (TMath::Abs(yy - ygraf[3]) <= 1e-4) xgraf[3] = xx;
            r1 = -r1;
            if (i % 2 == 0) r2 = -r2;
            if (i >= 4)     r3 = 1;
        }
        gPad->PaintFillArea(6, xgraf, ygraf);
    }
}


//______________________________________________________________________________
void  TView::SetAxisNDC(Double_t *x1, Double_t *x2, Double_t *y1, Double_t *y2, Double_t *z1, Double_t *z2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Store axis coordinates in the NDC system*-*-*-*
//*-*                      ========================================
//*-*

   for (Int_t i=0;i<3;i++) {
      fX1[i] = x1[i];
      fX2[i] = x2[i];
      fY1[i] = y1[i];
      fY2[i] = y2[i];
      fZ1[i] = z1[i];
      fZ2[i] = z2[i];
   }
}

//______________________________________________________________________________
void TView::SetOutlineToCube()
{
//*-*-*-*-*-*-*This is a function which creates default outline*-*-*-*-*-*
//*-*          ================================================          *
//*-*                                                                    *
//*-*      x = fRmin[0]        X = fRmax[0]                              *
//*-*      y = fRmin[1]        Y = fRmax[1]                              *
//*-*      z = fRmin[2]        Z = fRmax[2]                              *
//*-*                                                                    *
//*-*                                                                    *
//*-*            (x,Y,Z) +---------+ (X,Y,Z)                             *
//*-*                   /         /|                                     *
//*-*                  /         / |                                     *
//*-*                 /         /  |                                     *
//*-*        (x,y,Z) +---------+   |                                     *
//*-*                |         |   + (X,Y,z)                             *
//*-*                |         |  /                                      *
//*-*                |         | /                                       *
//*-*                |         |/                                        *
//*-*                +---------+                                         *
//*-*             (x,y,z)   (X,y,z)                                      *
//*-*                                                                    *
//*-*                                                                    *
//*-*                                                                    *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**

   if (!fOutline) {
      fDefaultOutline = kTRUE;
      fOutline = new TList();
   }
   gROOT->ProcessLineFast(Form("TPolyLine3D::DrawOutlineCube((TList *)0x%lx,"
                           "(Double_t*)0x%lx,(Double_t*)0x%lx);",
                           (Long_t)fOutline,(Long_t)fRmin,(Long_t)fRmax));
}

//______________________________________________________________________________
void TView::SetRange(Double_t *min, Double_t *max)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Set Range function-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================
//*-*
        Int_t irep;

        for (Int_t i = 0; i < 3; fRmax[i] = max[i], fRmin[i] = min[i], i++);
        ResetView(fLongitude, fLatitude, fPsi, irep);
        if(irep < 0)
                Error("SetRange", "problem setting view");
        if(fDefaultOutline) SetOutlineToCube();
}


//______________________________________________________________________________
void TView::SetRange(Double_t x0, Double_t y0, Double_t z0, Double_t x1, Double_t y1, Double_t z1, Int_t flag)
{
//*-*-*-*-*-*-*-*-*-*-*-*Set 3-D View range*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==================
//*-*
//*-* Input:  x0, y0, z0 are minimum coordinates
//*-*         x1, y1, z1 are maximum coordinates
//*-*
//*-*         flag values are: 0 (set always) <- default
//*-*                          1 (shrink view)
//*-*                          2 (expand view)
//*-*


    Double_t rmax[3], rmin[3];

    switch (flag) {
        case 2:                     // expand view
            GetRange(rmin, rmax);
            rmin[0] = x0 < rmin[0] ? x0 : rmin[0];
            rmin[1] = y0 < rmin[1] ? y0 : rmin[1];
            rmin[2] = z0 < rmin[2] ? z0 : rmin[2];
            rmax[0] = x1 > rmax[0] ? x1 : rmax[0];
            rmax[1] = y1 > rmax[1] ? y1 : rmax[1];
            rmax[2] = z1 > rmax[2] ? z1 : rmax[2];
            break;

        case 1:                     // shrink view
            GetRange(rmin, rmax);
            rmin[0] = x0 > rmin[0] ? x0 : rmin[0];
            rmin[1] = y0 > rmin[1] ? y0 : rmin[1];
            rmin[2] = z0 > rmin[2] ? z0 : rmin[2];
            rmax[0] = x1 < rmax[0] ? x1 : rmax[0];
            rmax[1] = y1 < rmax[1] ? y1 : rmax[1];
            rmax[2] = z1 < rmax[2] ? z1 : rmax[2];
            break;

        default:
            rmin[0] = x0; rmax[0] = x1;
            rmin[1] = y0; rmax[1] = y1;
            rmin[2] = z0; rmax[2] = z1;
    }
    SetRange(rmin, rmax);
}

//______________________________________________________________________________
void TView::SetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep)
{
    ResetView(longitude, latitude, psi, irep);
}

//______________________________________________________________________________
void TView::ResetView(Double_t longitude, Double_t latitude, Double_t psi, Int_t &irep)
{
//*-*-*-*-*-*-*-*-*Set view direction (in spherical coordinates)*-*-*-*-*-*
//*-*              =============================================          *
//*-*                                                                     *
//*-*    Input  PHI     - longitude                                       *
//*-*           THETA   - latitude (angle between +Z and view direction)  *
//*-*           PSI     - rotation in screen plane                        *
//*-*                                                                     *
//*-*    Output: IREP   - reply (-1 if error in min-max)                  *
//*-*                                                                     *
//*-*    Errors: error in min-max scope                                   *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t scale[3],  centre[3];
    Double_t c1, c2, c3, s1, s2, s3;

//*-*-        F I N D   C E N T E R   O F   S C O P E   A N D
//*-*-        S C A L E   F A C T O R S

    FindScope(scale, centre, irep);
    if (irep < 0) {
        Error("SetView", "Error in min-max scope");
        return;
    }

//*-*-        S E T   T R A N S F O R M A T I O N   M A T R I C E S


    fLongitude = longitude;
    fPsi       = psi;
    fLatitude  = latitude;

    c1 = TMath::Cos(longitude*kRad);
    s1 = TMath::Sin(longitude*kRad);
    c2 = TMath::Cos(latitude*kRad);
    s2 = TMath::Sin(latitude*kRad);
    c3 = TMath::Cos(psi*kRad);
    s3 = TMath::Sin(psi*kRad);
    DefineViewDirection(scale, centre, c1, s1, c2, s2, c3, s3, fTnorm, fTback);
    c3 = 1;
    s3 = 0;
    DefineViewDirection(scale, centre, c1, s1, c2, s2, c3, s3, fTN, fTB);
}


//______________________________________________________________________________
void TView::WCtoNDC(Float_t *pw, Float_t *pn)
{
//*-*-*-*-*-*-*Transfer point from world to normalized coordinates*-*-*-*-*
//*-*          ===================================================        *
//*-*                                                                     *
//*-*    Input: PW(3) - point in world coordinate system                  *
//*-*           PN(3) - point in normalized coordinate system             *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    pn[0] = fTnorm[0]*pw[0] + fTnorm[1]*pw[1] + fTnorm[2]*pw[2]  + fTnorm[3];
    pn[1] = fTnorm[4]*pw[0] + fTnorm[5]*pw[1] + fTnorm[6]*pw[2]  + fTnorm[7];
    pn[2] = fTnorm[8]*pw[0] + fTnorm[9]*pw[1] + fTnorm[10]*pw[2] + fTnorm[11];

       // following lines to take into account perspective views
//    if (TestBit(kPerspective) == 0) return;
//    if (pn[2] == 0) return;
//    printf("pw= %f, %f, %f, pn= %f, %f, %f, ppx=%f, ppy=%f\n",pw[0],pw[1],pw[2],pn[0],pn[1],pn[2],pn[0]/pn[2],pn[1]/pn[2]);
//    pn[0] /= pn[2];
//    pn[1] /= pn[2];
}


//______________________________________________________________________________
void TView::WCtoNDC(Double_t *pw, Double_t *pn)
{
//*-*-*-*-*-*-*Transfer point from world to normalized coordinates*-*-*-*-*
//*-*          ===================================================        *
//*-*                                                                     *
//*-*    Input: PW(3) - point in world coordinate system                  *
//*-*           PN(3) - point in normalized coordinate system             *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    pn[0] = fTnorm[0]*pw[0] + fTnorm[1]*pw[1] + fTnorm[2]*pw[2]  + fTnorm[3];
    pn[1] = fTnorm[4]*pw[0] + fTnorm[5]*pw[1] + fTnorm[6]*pw[2]  + fTnorm[7];
    pn[2] = fTnorm[8]*pw[0] + fTnorm[9]*pw[1] + fTnorm[10]*pw[2] + fTnorm[11];

       // following lines to take into account perspective views
//    if (TestBit(kPerspective) == 0) return;
//    if (pn[2] == 0) return;
//    printf("pw= %f, %f, %f, pn= %f, %f, %f, ppx=%f, ppy=%f\n",pw[0],pw[1],pw[2],pn[0],pn[1],pn[2],pn[0]/pn[2],pn[1]/pn[2]);
//    pn[0] /= pn[2];
//    pn[1] /= pn[2];
}

//_______________________________________________________________________________________
void TView::AdjustPad(TVirtualPad *pad)
{
  // Force the current pad to be updated
  TVirtualPad *thisPad = pad;
  if (!thisPad) thisPad = gPad;
  if (thisPad) {
    thisPad->Modified();
    thisPad->Update();
  }
}
//_______________________________________________________________________________________
void TView::RotateView(Double_t phi, Double_t theta, TVirtualPad *pad)
{
  // API to rotate view and adjust the pad provided it the current one.

  Int_t iret;
  Double_t p = phi;
  Double_t t = theta;
  SetView(p, t, 0, iret);

  // Adjust current pad too

  TVirtualPad *thisPad = pad;
  if (!thisPad) thisPad = gPad;
  if (thisPad) {
      thisPad->SetPhi(-90-p);
      thisPad->SetTheta(90-t);
      thisPad->Modified();
      thisPad->Update();
  }
}

//_______________________________________________________________________________________
void TView::SideView(TVirtualPad *pad){
  RotateView(0,90.0,pad);
}
//_______________________________________________________________________________________
void TView::FrontView(TVirtualPad *pad){
  RotateView(270.0,90.0,pad);
}
//_______________________________________________________________________________________
void TView::TopView(TVirtualPad *pad){
  RotateView(270.0,0.0,pad);
}
//_______________________________________________________________________________________
void TView::ToggleRulers(TVirtualPad *pad)
{
  // Turn on /off 3D axis
  if (pad) gROOT->ProcessLine(Form("TAxis3D::ToggleRulers((TVirtualPad *)0x%lx);",(Long_t)pad));
  else     gROOT->ProcessLine(Form("TAxis3D::ToggleRulers();"));
}

//_______________________________________________________________________________________
void TView::ToggleZoom(TVirtualPad *pad)
{
  // Turn on /off the interactive option to
  //  Zoom / Move / Change attributes of 3D axis correspond this view
  if (pad) gROOT->ProcessLine(Form("TAxis3D::ToggleZoom((TVirtualPad *)0x%lx);",(Long_t)pad));
  else     gROOT->ProcessLine(Form("TAxis3D::ToggleZoom();"));
}

//_______________________________________________________________________________________
 void TView::AdjustScales(TVirtualPad *pad)
{
  // Adjust all sides of view in respewct of the biggest one
  Double_t min[3],max[3];
  GetRange(min,max);
  int i;
  Double_t maxSide = 0;
  // Find the largest side
  for (i=0;i<3; i++) maxSide = TMath::Max(maxSide,max[i]-min[i]);
  //Adjust scales:
  for (i=0;i<3; i++) max[i] += maxSide - (max[i]-min[i]);
  SetRange(min,max);

  AdjustPad(pad);
}
//_______________________________________________________________________________________
void TView::Centered3DImages(TVirtualPad *pad)
{
  // Move view into the center of the scene

  Double_t min[3],max[3];
  GetRange(min,max);
  int i;
  for (i=0;i<3; i++) min[i]=-max[i];
  SetRange(min,max);
  AdjustPad(pad);
}

//_______________________________________________________________________________________
 void TView::UnzoomView(TVirtualPad *pad,Double_t unZoomFactor )
{
  // unZOOM this view
  if (TMath::Abs(unZoomFactor) < 0.001) return;
  ZoomView(pad,1./unZoomFactor);
}

//_______________________________________________________________________________________
void TView::ZoomView(TVirtualPad *pad,Double_t zoomFactor)
{
  // ZOOM this view
  if (TMath::Abs(zoomFactor) < 0.001) return;
  Double_t min[3],max[3];
  GetRange(min,max);
  int i;
  for (i=0;i<3; i++) {
    // Find center
    Double_t c = (max[i]+min[i])/2;
    // Find a new size
    Double_t s = (max[i]-min[i])/(2*zoomFactor);
    // Set a new size
    max[i] = c + s;
    min[i] = c - s;
  }
  SetRange(min,max);
  AdjustPad(pad);
}

//_______________________________________________________________________________________
void TView::MoveViewCommand(Char_t option, Int_t count)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*          'a' //*-*  increase  scale factor (clip cube borders)
//*-*          's' //*-*  decrease  scale factor (clip cube borders)
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (count <= 0) count = 1;
   switch (option) {
       case '+':
            ZoomView();
            break;
       case '-':
            UnzoomView();
            break;
       case 's':
       case 'S':
           UnzoomView();
           break;
       case 'a':
       case 'A':
           ZoomView();
           break;
       default:
           break;
  }
}

//______________________________________________________________________________
void TView::Streamer(TBuffer &R__b)
{
   // Stream an object of class TView.

   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(); if (R__v) {};
      //unfortunately we forgot to increment the TView version number
      //when the class was upgraded to double precision in version 2.25.
      //we are forced to use the file version number to recognize old files.
      if (gFile->GetVersion() < 22500) { //old version in single precision
         TObject::Streamer(R__b);
         TAttLine::Streamer(R__b);
         Float_t single, sa[12];
         Int_t i;
         R__b >> fSystem;
         R__b >> single; fLatitude = single;
         R__b >> single; fLongitude = single;
         R__b >> single; fPsi = single;
         R__b.ReadStaticArray(sa);   for (i=0;i<12;i++) fTN[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<12;i++) fTB[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fRmax[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fRmin[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<12;i++) fTnorm[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<12;i++) fTback[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fX1[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fX2[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fY1[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fY2[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fZ1[i] = sa[i];
         R__b.ReadStaticArray(sa);   for (i=0;i<3;i++)  fZ2[i] = sa[i];
         R__b >> fOutline;
         R__b >> fDefaultOutline;
         R__b >> fAutoRange;
      } else {
         TObject::Streamer(R__b);
         TAttLine::Streamer(R__b);
         R__b >> fLatitude;
         R__b >> fLongitude;
         R__b >> fPsi;
         R__b.ReadStaticArray(fTN);
         R__b.ReadStaticArray(fTB);
         R__b.ReadStaticArray(fRmax);
         R__b.ReadStaticArray(fRmin);
         R__b.ReadStaticArray(fTnorm);
         R__b.ReadStaticArray(fTback);
         R__b.ReadStaticArray(fX1);
         R__b.ReadStaticArray(fX2);
         R__b.ReadStaticArray(fY1);
         R__b.ReadStaticArray(fY2);
         R__b.ReadStaticArray(fZ1);
         R__b.ReadStaticArray(fZ2);
         R__b >> fSystem;
         R__b >> fOutline;
         R__b >> fDefaultOutline;
         R__b >> fAutoRange;
      }
   } else {
      R__b.WriteVersion(TView::IsA());
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      R__b << fLatitude;
      R__b << fLongitude;
      R__b << fPsi;
      R__b.WriteArray(fTN, 12);
      R__b.WriteArray(fTB, 12);
      R__b.WriteArray(fRmax, 3);
      R__b.WriteArray(fRmin, 3);
      R__b.WriteArray(fTnorm, 12);
      R__b.WriteArray(fTback, 12);
      R__b.WriteArray(fX1, 3);
      R__b.WriteArray(fX2, 3);
      R__b.WriteArray(fY1, 3);
      R__b.WriteArray(fY2, 3);
      R__b.WriteArray(fZ1, 3);
      R__b.WriteArray(fZ2, 3);
      R__b << fSystem;
      R__b << fOutline;
      R__b << fDefaultOutline;
      R__b << fAutoRange;
   }
}
