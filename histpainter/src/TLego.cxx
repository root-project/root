// @(#)root/histpainter:$Name:  $:$Id: TLego.cxx,v 1.5 2001/07/20 13:49:53 brun Exp $
// Author: Rene Brun, Evgueni Tcherniaev, Olivier Couet   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*Legos and Surfaces package-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==========================                     *
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

#include <stdlib.h>

#include "TROOT.h"
#include "TLego.h"
#include "TVirtualPad.h"
#include "THistPainter.h"
#include "TH1.h"
#include "TView.h"
#include "TVirtualX.h"
#include "Hoption.h"
#include "Hparam.h"
#include "TMath.h"
#include "TStyle.h"
#include "TObjArray.h"

#ifdef R__SUNCCBUG
const Double_t kRad = 1.74532925199432955e-02;
#else
const Double_t kRad = TMath::ATan(1)*Double_t(4)/Double_t(180);
#endif

  R__EXTERN TH1  *gCurrentHist;
  R__EXTERN Hoption_t Hoption;
  R__EXTERN Hparam_t  Hparam;

ClassImp(TLego)

//______________________________________________________________________________
TLego::TLego(): TObject(), TAttLine(1,1,1), TAttFill(1,0)
{
//*-*-*-*-*-*-*-*-*-*-*Lego default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   Int_t i;
   fIfrast       = 0;
   fMesh         = 1;
   fRaster       = 0;
   fColorTop     = 1;
   fColorBottom  = 1;
   fNlevel       = 0;
   fSystem       = kCARTESIAN;
   for (i=0;i<10;i++) { fColorMain[i] = 1; fColorDark[i] = 1; }
   for (i=0;i<3;i++)  { fRmin[i] = 0, fRmax[i] = 1; }
   for (i=0;i<4;i++)  { fYls[i] = 0; }
}

//______________________________________________________________________________
TLego::TLego(Double_t *rmin, Double_t *rmax, Int_t system)
      : TObject(), TAttLine(1,1,1), TAttFill(1,0)
{
//*-*-*-*-*-*-*-*-*-*-*Normal default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
//*-*  rmin[3], rmax[3] are the limits of the lego object depending on
//*-*  the selected coordinate system
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t i;
   Double_t psi;

   fIfrast       = 0;
   fMesh         = 1;
   fRaster       = 0;
   fColorTop     = 1;
   fColorBottom  = 1;
   fNlevel       = 0;
   fSystem       = system;
   if (system == kCARTESIAN || system == kPOLAR) psi =  0;
   else                                          psi = 90;
   for (i=0;i<10;i++) { fColorMain[i] = 1; fColorDark[i] = 1; }
   for (i=0;i<3;i++)  { fRmin[i] = rmin[i], fRmax[i] = rmax[i]; }
   for (i=0;i<4;i++)  { fYls[i] = 0; }

   TView *view = gPad->GetView();
   if (!view) view = new TView(rmin, rmax, fSystem);
   view->SetView(gPad->GetPhi(), gPad->GetTheta(), psi, i);
   view->SetRange(rmin,rmax);
}

//______________________________________________________________________________
TLego::~TLego()
{
//*-*-*-*-*-*-*-*-*-*-*Lego default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

   delete [] fRaster;
   fRaster = 0;
}
//______________________________________________________________________________
void TLego::BackBox(Double_t ang)
{
//*-*-*-*-*-*-*-*-*-*Draw back surfaces of surrounding box*-*-*-*-*-*-*-*-*
//*-*                =====================================                *
//*-*                                                                     *
//*-*    Input  ANG     - angle between X and Y axis                      *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */

    static Int_t iface1[4] = { 1,4,8,5 };
    static Int_t iface2[4] = { 4,3,7,8 };
    TView *view = 0;

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
    	   Error("BackBox", "no TView in current pad");
    		return;
    	}
    }


    /* Local variables */
    Double_t cosa, sina;
    Int_t i;
    Double_t r[24]	/* was [3][8] */, av[24]	/* was [3][8] */;
    Int_t icodes[3];
    Double_t tt[4];
    Int_t ix1, ix2, iy1, iy2, iz1, iz2;

    cosa = TMath::Cos(kRad*ang);
    sina = TMath::Sin(kRad*ang);
	view->AxisVertex(ang, av, ix1, ix2, iy1, iy2, iz1, iz2);
    for (i = 1; i <= 8; ++i) {
	r[i*3 - 3] = av[i*3 - 3] + av[i*3 - 2]*cosa;
	r[i*3 - 2] = av[i*3 - 2]*sina;
	r[i*3 - 1] = av[i*3 - 1];
    }

//*-*-          D R A W   F O R W A R D   F A C E S */

    icodes[0] = 0;
    icodes[1] = 0;
    icodes[2] = 0;
    tt[0] = r[iface1[0]*3 - 1];
    tt[1] = r[iface1[1]*3 - 1];
    tt[2] = r[iface1[2]*3 - 1];
    tt[3] = r[iface1[3]*3 - 1];
    (this->*fDrawFace)(icodes, r, 4, iface1, tt);
    tt[0] = r[iface2[0]*3 - 1];
    tt[1] = r[iface2[1]*3 - 1];
    tt[2] = r[iface2[2]*3 - 1];
    tt[3] = r[iface2[3]*3 - 1];
    (this->*fDrawFace)(icodes, r, 4, iface2, tt);
}


//______________________________________________________________________________
void TLego::ClearRaster()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*Clear screen*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ============
    Int_t nw = (fNxrast*fNyrast + 29) / 30;
    for (Int_t i = 0; i < nw; ++i) fRaster[i] = 0;
    fIfrast = 0;
}

//______________________________________________________________________________
void TLego::ColorFunction(Int_t nl, Double_t *fl, Int_t *icl, Int_t &irep)
{
//*-*-*-*-*-*Set correspondance between function and color levels-*-*-*-*-*
//*-*        ====================================================         *
//*-*                                                                     *
//*-*    Input: NL        - number of levels                              *
//*-*           FL(NL)    - function levels                               *
//*-*           ICL(NL+1) - colors for levels                             *
//*-*                                                                     *
//*-*    Output: IREP     - reply: 0 O.K.                                 *
//*-*                             -1 error in parameters:                 *
//*-*                         illegal number of levels                    *
//*-*                         function levels must be in increasing order *
//*-*                         negative color index                        *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    static const char *where = "ColorFunction";

    /* Local variables */
    Int_t i;

    irep = 0;
    if (nl == 0) {fNlevel = 0;	return; }

//*-*-          C H E C K   P A R A M E T E R S

    if (nl < 0 || nl > 256) {
       Error(where, "illegal number of levels (%d)", nl);
       irep = -1;
       return;
    }
    for (i = 1; i < nl; ++i) {
	if (fl[i] <= fl[i - 1]) {
//         Error(where, "function levels must be in increasing order");
           irep = -1;
           return;
        }
    }
    for (i = 0; i < nl; ++i) {
	if (icl[i] < 0) {
//         Error(where, "negative color index (%d)", icl[i]);
           irep = -1;
           return;
        }
    }

//*-*-          S E T   L E V E L S

    fNlevel = nl;
    for (i = 0; i < fNlevel; ++i) fFunLevel[i]   = fl[i];
    for (i = 0; i < fNlevel+1; ++i) fColorLevel[i] = icl[i];
}


//______________________________________________________________________________
void TLego::DrawFaceMode1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw face - 1st variant*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =======================                          *
//*-*                                                                     *
//*-*    Function: Draw face - 1st variant                                *
//*-*              (2 colors: 1st for external surface, 2nd for internal) *
//*-*                                                                     *
//*-*    References: WCtoNDC                                              *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           T(NP)     - additional function defined on this face      *
//*-*                       (not used in this routine)                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Int_t i, k,ifneg,i1, i2;
    Double_t x[13], y[13];
    Double_t z;
    Double_t p3[24]	/* was [2][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


//*-*-          T R A N S F E R   T O   N O R M A L I S E D   COORDINATES

    /* Parameter adjustments */
    --t;
    --iface;
    xyz -= 4;
    --icodes;

    ifneg = 0;
    for (i = 1; i <= np; ++i) {
	k = iface[i];
	if (k < 0) ifneg = 1;
	if (k < 0) k = -k;
	view->WCtoNDC(&xyz[k*3 + 1], &p3[2*i - 2]);
	x[i - 1] = p3[2*i - 2];
	y[i - 1] = p3[2*i - 1];
    }

//*-*-          F I N D   N O R M A L

    z = 0;
    for (i = 1; i <= np; ++i) {
	i1 = i;
	i2 = i1 + 1;
	if (i2 > np) i2 = 1;
	z = z + p3[2*i1 - 1]*p3[2*i2 - 2] - p3[2*i1 - 2] *
		p3[2*i2 - 1];
    }

//*-*-          D R A W   F A C E

    if (z > 0) 	SetFillColor(2);
    if (z <= 0) SetFillColor(3);
    SetFillStyle(1001);
    TAttFill::Modify();
    gPad->PaintFillArea(np, x, y);

//*-*-          D R A W   B O R D E R

    if (ifneg == 0) {
	SetFillStyle(0);
	SetFillColor(1);
        TAttFill::Modify();
	gPad->PaintFillArea(np, x, y);
    } else {
	x[np] = x[0];
	y[np] = y[0];
	SetLineColor(1);
        TAttLine::Modify();
	for (i = 1; i <= np; ++i) {
	    if (iface[i] > 0) gPad->PaintPolyLine(2, &x[i-1], &y[i-1]);
	}
    }
}

//______________________________________________________________________________
void TLego::DrawFaceMode2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-Draw face - 2nd option*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                   ======================                            *
//*-*                                                                     *
//*-*    Function: Draw face - 2nd option                                 *
//*-*              (fill in correspondance with function levels)          *
//*-*                                                                     *
//*-*    References: WCtoNDC, FillPolygon                                 *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           T(NP)     - additional function defined on this face      *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Int_t i, k;
    Double_t x[12], y[12];
    Double_t p3[36]	/* was [3][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


//*-*-          T R A N S F E R   T O   N O R M A L I S E D   COORDINATES

    /* Parameter adjustments */
    --t;
    --iface;
    xyz -= 4;
    --icodes;

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	view->WCtoNDC(&xyz[k*3 + 1], &p3[i*3 - 3]);
	x[i - 1] = p3[i*3 - 3];
	y[i - 1] = p3[i*3 - 2];
    }

//*-*-          D R A W   F A C E   &   B O R D E R

    FillPolygon(np, p3, &t[1]);
    if (fMesh == 1) {
	SetFillColor(1);
	SetFillStyle(0);
        TAttFill::Modify();
	gPad->PaintFillArea(np, x, y);
    }
}

//______________________________________________________________________________
void TLego::DrawFaceMode3(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw face - 3rd option-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ======================                           *
//*-*                                                                     *
//*-*    Function: Draw face - 3rd option                                 *
//*-*              (draw face for stacked lego plot)                      *
//*-*                                                                     *
//*-*    References: WCtoNDC                                              *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line                     *
//*-*             ICODES(1) - IX coordinate of the line cell              *
//*-*             ICODES(2) - IY coordinate of the line cell              *
//*-*             ICODES(3) - lego number                                 *
//*-*             ICODES(4) - side: 1-face,2-right,3-back,4-left,         *
//*-*                               5-bottom, 6-top                       *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes                             *
//*-*             IFACE(NP) - face                                        *
//*-*             T(*)      - additional function (not used here)         *
//*-*                                                                     *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, k;
    Int_t icol = 0;
    Double_t x[4], y[4], p3[12]	/* was [3][4] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


    /* Parameter adjustments */
    --t;
    --iface;
    xyz -= 4;
    --icodes;

    if (icodes[4] == 6) icol = fColorTop;
    if (icodes[4] == 5) icol = fColorBottom;
    if (icodes[4] == 1) icol = fColorMain[icodes[3] - 1];
    if (icodes[4] == 2) icol = fColorDark[icodes[3] - 1];
    if (icodes[4] == 3) icol = fColorMain[icodes[3] - 1];
    if (icodes[4] == 4) icol = fColorDark[icodes[3] - 1];

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	view->WCtoNDC(&xyz[k*3 + 1], &p3[i*3 - 3]);
	x[i - 1] = p3[i*3 - 3];
	y[i - 1] = p3[i*3 - 2];
    }

    SetFillStyle(1001);
    SetFillColor(icol);
    TAttFill::Modify();
    gPad->PaintFillArea(np, x, y);
    if (fMesh) {
	SetFillStyle(0);
	SetFillColor(1);
        TAttFill::Modify();
	gPad->PaintFillArea(np, x, y);
    }
}

//______________________________________________________________________________
void TLego::DrawFaceMove1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt)
{
//*-*-*-*-*-*Draw face - 1st variant for "MOVING SCREEN" algorithm -*-*-*-*
//*-*        =====================================================        *
//*-*                                                                     *
//*-*    Function: Draw face - 1st variant for "MOVING SCREEN" algorithm  *
//*-*              (draw face with level lines)                           *
//*-*                                                                     *
//*-*    References: FindLevelLines, WCtoNDC,                             *
//*-*                FindVisibleDraw, ModifyScreen                        *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           TT(NP)    - additional function defined on this face      *
//*-*                       (not used in this routine)                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t xdel, ydel;
    Int_t i, k, i1, i2, il, it;
    Double_t x[2], y[2];
    Double_t p1[3], p2[3], p3[36]	/* was [3][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


//*-*-          C O P Y   P O I N T S   T O   A R R A Y

    /* Parameter adjustments */
    --tt;
    --iface;
    xyz -= 4;
    --icodes;

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	p3[i*3 - 3] = xyz[k*3 + 1];
	p3[i*3 - 2] = xyz[k*3 + 2];
	p3[i*3 - 1] = xyz[k*3 + 3];
    }

//*-*-          F I N D   L E V E L   L I N E S

    FindLevelLines(np, p3, &tt[1]);

//*-*-          D R A W   L E V E L   L I N E S

    SetLineStyle(3);
    TAttLine::Modify();
    for (il = 1; il <= fNlines; ++il) {
	FindVisibleDraw(&fPlines[(2*il + 1)*3 - 9], &fPlines[(2*il + 2)*3 - 9]);
	view->WCtoNDC(&fPlines[(2*il + 1)*3 - 9], p1);
	view->WCtoNDC(&fPlines[(2*il + 2)*3 - 9], p2);
	xdel = p2[0] - p1[0];
	ydel = p2[1] - p1[1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = p1[0] + xdel*fT[2*it - 2];
	    y[0] = p1[1] + ydel*fT[2*it - 2];
	    x[1] = p1[0] + xdel*fT[2*it - 1];
	    y[1] = p1[1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          D R A W   F A C E

    SetLineStyle(1);
    TAttLine::Modify();
    for (i = 1; i <= np; ++i) {
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	FindVisibleDraw(&p3[i1*3 - 3], &p3[i2*3 - 3]);
	view->WCtoNDC(&p3[i1*3 - 3], p1);
	view->WCtoNDC(&p3[i2*3 - 3], p2);
	xdel = p2[0] - p1[0];
	ydel = p2[1] - p1[1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = p1[0] + xdel*fT[2*it - 2];
	    y[0] = p1[1] + ydel*fT[2*it - 2];
	    x[1] = p1[0] + xdel*fT[2*it - 1];
	    y[1] = p1[1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          M O D I F Y    S C R E E N

    for (i = 1; i <= np; ++i) {
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	ModifyScreen(&p3[i1*3 - 3], &p3[i2*3 - 3]);
    }
}

//______________________________________________________________________________
void TLego::DrawFaceMove2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt)
{
//*-*-*-*-*-*Draw face - 2nd variant for "MOVING SCREEN" algorithm*-*-*-*-*
//*-*        =====================================================        *
//*-*                                                                     *
//*-*    Function: Draw face - 2nd variant for "MOVING SCREEN" algorithm  *
//*-*              (draw face for stacked lego plot)                      *
//*-*                                                                     *
//*-*    References: FindLevelLines, WCtoNDC,                             *
//*-*                FindVisibleDraw, ModifyScreen                        *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*             ICODES(3) - line code (N of lego)                       *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           TT(NP)    - additional function defined on this face      *
//*-*                       (not used in this routine)                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t xdel, ydel;
    Int_t i, k, icol, i1, i2, it;
    Double_t x[2], y[2];
    Double_t p1[3], p2[3], p3[36]	/* was [3][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


//*-*-          C O P Y   P O I N T S   T O   A R R A Y

    /* Parameter adjustments */
    --tt;
    --iface;
    xyz -= 4;
    --icodes;

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	p3[i*3 - 3] = xyz[k*3 + 1];
	p3[i*3 - 2] = xyz[k*3 + 2];
	p3[i*3 - 1] = xyz[k*3 + 3];
    }

//*-*-          D R A W   F A C E

    icol = icodes[3];
    if (icol) SetLineColor(fColorMain[icol - 1]);
    else      SetLineColor(1);
    TAttLine::Modify();
    for (i = 1; i <= np; ++i) {
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	FindVisibleDraw(&p3[i1*3 - 3], &p3[i2*3 - 3]);
	view->WCtoNDC(&p3[i1*3 - 3], p1);
	view->WCtoNDC(&p3[i2*3 - 3], p2);
	xdel = p2[0] - p1[0];
	ydel = p2[1] - p1[1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = p1[0] + xdel*fT[2*it - 2];
	    y[0] = p1[1] + ydel*fT[2*it - 2];
	    x[1] = p1[0] + xdel*fT[2*it - 1];
	    y[1] = p1[1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          M O D I F Y    S C R E E N

    for (i = 1; i <= np; ++i) {
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	ModifyScreen(&p3[i1*3 - 3], &p3[i2*3 - 3]);
    }
}

//______________________________________________________________________________
void TLego::DrawFaceRaster1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt)
{
//*-*-*-*-*-*-*Draw face - 1st variant for "RASTER SCREEN" algorithm*-*-*-*
//*-*          =====================================================      *
//*-*                                                                     *
//*-*    Function: Draw face - 1st variant for "RASTER SCREEN" algorithm  *
//*-*              (draw face with level lines)                           *
//*-*                                                                     *
//*-*    References: FindLevelLines, WCtoNDC,                             *
//*-*                FindVisibleLine, FillPolygonBorder                   *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           TT(NP)    - additional function defined on this face      *
//*-*                       (not used in this routine)                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t xdel, ydel;
    Int_t i, k, i1, i2, il, it;
    Double_t x[2], y[2];
    Double_t p1[3], p2[3], p3[36]	/* was [3][12] */;
    Double_t pp[24]	/* was [2][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


//*-*-          C O P Y   P O I N T S   T O   A R R A Y

    /* Parameter adjustments */
    --tt;
    --iface;
    xyz -= 4;
    --icodes;

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	if (k < 0) k = -k;
	p3[i*3 - 3] = xyz[k*3 + 1];
	p3[i*3 - 2] = xyz[k*3 + 2];
	p3[i*3 - 1] = xyz[k*3 + 3];
	view->WCtoNDC(&p3[i*3 - 3], &pp[2*i - 2]);
    }

//*-*-          F I N D   L E V E L   L I N E S

    FindLevelLines(np, p3, &tt[1]);

//*-*-          D R A W   L E V E L   L I N E S

    SetLineStyle(3);
    TAttLine::Modify();
    for (il = 1; il <= fNlines; ++il) {
	view->WCtoNDC(&fPlines[(2*il + 1)*3 - 9], p1);
	view->WCtoNDC(&fPlines[(2*il + 2)*3 - 9], p2);
	FindVisibleLine(p1, p2, 100, fNT, fT);
	xdel = p2[0] - p1[0];
	ydel = p2[1] - p1[1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = p1[0] + xdel*fT[2*it - 2];
	    y[0] = p1[1] + ydel*fT[2*it - 2];
	    x[1] = p1[0] + xdel*fT[2*it - 1];
	    y[1] = p1[1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          D R A W   F A C E

    SetLineStyle(1);
    TAttLine::Modify();
    for (i = 1; i <= np; ++i) {
	if (iface[i] < 0) continue;
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	FindVisibleLine(&pp[2*i1 - 2], &pp[2*i2 - 2], 100, fNT, fT);
	xdel = pp[2*i2 - 2] - pp[2*i1 - 2];
	ydel = pp[2*i2 - 1] - pp[2*i1 - 1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = pp[2*i1 - 2] + xdel*fT[2*it - 2];
	    y[0] = pp[2*i1 - 1] + ydel*fT[2*it - 2];
	    x[1] = pp[2*i1 - 2] + xdel*fT[2*it - 1];
	    y[1] = pp[2*i1 - 1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          M O D I F Y    S C R E E N

    FillPolygonBorder(np, pp);

}

//______________________________________________________________________________
void TLego::DrawFaceRaster2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt)
{
//*-*-*-*-*-*Draw face - 2nd variant for "RASTER SCREEN" algorithm*-*-*-*-*
//*-*        =====================================================        *
//*-*                                                                     *
//*-*    Function: Draw face - 2nd variant for "RASTER SCREEN" algorithm  *
//*-*              (draw face for stacked lego plot)                      *
//*-*                                                                     *
//*-*    References: WCtoNDC, FindVisibleLine, FillPolygonBorder          *
//*-*                                                                     *
//*-*    Input: ICODES(*) - set of codes for the line (not used)          *
//*-*             ICODES(1) - IX                                          *
//*-*             ICODES(2) - IY                                          *
//*-*             ICODES(3) - line code (N of lego)                       *
//*-*           XYZ(3,*)  - coordinates of nodes                          *
//*-*           NP        - number of nodes                               *
//*-*           IFACE(NP) - face                                          *
//*-*           TT(NP)    - additional function defined on this face      *
//*-*                       (not used in this routine)                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t xdel, ydel;
    Int_t i, k, icol, i1, i2, it;
    Double_t p[3], x[2], y[2];
    Double_t pp[24]	/* was [2][12] */;


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!

//*-*-          C O P Y   P O I N T S   T O   A R R A Y

    /* Parameter adjustments */
    --tt;
    --iface;
    xyz -= 4;
    --icodes;

    for (i = 1; i <= np; ++i) {
	k = iface[i];
	if (k < 0) k = -k;
	view->WCtoNDC(&xyz[k*3 + 1], p);
	pp[2*i - 2] = p[0];
	pp[2*i - 1] = p[1];
    }

//*-*-          D R A W   F A C E

    icol = icodes[3];
    if (icol) SetLineColor(fColorMain[icol - 1]);
    else      SetLineColor(1);
    TAttLine::Modify();
    for (i = 1; i <= np; ++i) {
	if (iface[i] < 0) continue;
	i1 = i;
	i2 = i + 1;
	if (i == np) i2 = 1;
	FindVisibleLine(&pp[2*i1 - 2], &pp[2*i2 - 2], 100, fNT, fT);
	xdel = pp[2*i2 - 2] - pp[2*i1 - 2];
	ydel = pp[2*i2 - 1] - pp[2*i1 - 1];
	for (it = 1; it <= fNT; ++it) {
	    x[0] = pp[2*i1 - 2] + xdel*fT[2*it - 2];
	    y[0] = pp[2*i1 - 1] + ydel*fT[2*it - 2];
	    x[1] = pp[2*i1 - 2] + xdel*fT[2*it - 1];
	    y[1] = pp[2*i1 - 1] + ydel*fT[2*it - 1];
	    gPad->PaintPolyLine(2, x, y);
	}
    }

//*-*-          M O D I F Y    R A S T E R   S C R E E N

    FillPolygonBorder(np, pp);

}


//______________________________________________________________________________
void TLego::FillPolygon(Int_t n, Double_t *p, Double_t *f)
{
//*-*-*-*-*-*-*-*Fill polygon with function values at vertexes*-*-*-*-*-*-*
//*-*            =============================================            *
//*-*                                                                     *
//*-*    Input: N      - number of vertexes                               *
//*-*           P(3,*) - polygon                                          *
//*-*           F(*)   - function values at nodes                         *
//*-*                                                                     *
//*-*    Errors: - illegal number of vertexes in polygon                  *
//*-*            - illegal call of FillPolygon: no levels                 *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t ilev, i, k, icol, i1, i2, nl, np;
    Double_t fmin, fmax;
    Double_t x[12], y[12], f1, f2;
    Double_t p3[36]	/* was [3][12] */;
    Double_t funmin, funmax;


    /* Parameter adjustments */
    --f;
    p -= 4;

    if (n < 3) {
       Error("FillPolygon", "illegal number of vertices in polygon (%d)", n);
       return;
    }
    if (fNlevel == 0) {
//     Error("FillPolygon", "illegal call of FillPolygon: no levels");
       return;
    }
    np = n;
    nl = fNlevel;
    if (nl < 0) nl = -nl;
    fmin = f[1];
    fmax = f[1];
    for (i = 2; i <= np; ++i) {
	if (fmin > f[i]) fmin = f[i];
	if (fmax < f[i]) fmax = f[i];
    }
    funmin = fFunLevel[0] - 1;
    if (fmin < funmin) funmin = fmin - 1;
    funmax = fFunLevel[nl - 1] + 1;
    if (fmax > funmax) funmax = fmax + 1;

//*-*-          F I N D   A N D   D R A W   S U B P O L Y G O N S

    f2 = funmin;
    for (ilev = 1; ilev <= nl+1; ++ilev) {
//*-*-         S E T   L E V E L   L I M I T S
	f1 = f2;
	if (ilev == nl + 1) f2 = funmax;
	else                f2 = fFunLevel[ilev - 1];
	if (fmax < f1)  return;
	if (fmin > f2)  continue;
//*-*-         F I N D   S U B P O L Y G O N
	k = 0;
	for (i = 1; i <= np; ++i) {
	    i1 = i;
	    i2 = i + 1;
	    if (i == np) i2 = 1;
	    FindPartEdge(&p[i1*3 + 1], &p[i2*3 + 1], f[i1], f[i2], f1, f2, k, p3);
	}
//*-*-         D R A W   S U B P O L Y G O N
	if (k < 3) continue;
	for (i = 1; i <= k; ++i) {
	    x[i - 1] = p3[i*3 - 3];
	    y[i - 1] = p3[i*3 - 2];
	}
	icol = fColorLevel[ilev - 1];
	SetFillColor(icol);
	SetFillStyle(1001);
        TAttFill::Modify();
	gPad->PaintFillArea(k, x, y);
    }
}

//______________________________________________________________________________
void TLego::FillPolygonBorder(Int_t nn, Double_t *xy)
{
//*-*-*-*-*-*-*Fill a polygon including border ("RASTER SCREEN")*-*-*-*-*-*
//*-*          =================================================          *
//*-*                                                                     *
//*-*    Input: NN      - number of polygon nodes                         *
//*-*           XY(2,*) - polygon nodes                                   *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t kbit, nbit, step, ymin, ymax, test[12], xcur[12], xnex[12],
	     i, j, k, n, ibase, t, x, y, xscan[24]	/* was [2][12] */,
	    yscan, x1[14], y1[14], x2[12], y2[12], ib, nb, dx, dy, iw, nx, xx,
	     yy, signdx, nstart, xx1, xx2, nxa, nxb;

//*-*-          T R A N S F E R   T O   S C R E E N   C O O R D I N A T E S

    /* Parameter adjustments */
    xy -= 3;

    if (fIfrast) return;

    n = nn;
    for (i = 1; i <= n; ++i) {
	x1[i - 1] = Int_t(fNxrast*((xy[2*i + 1] - fXrast) /fDXrast) - 0.01);
	y1[i - 1] = Int_t(fNyrast*((xy[2*i + 2] - fYrast) /fDYrast) - 0.01);
    }
    x1[n] = x1[0];
    y1[n] = y1[0];

//*-*-          F I N D   Y - M I N   A N D   Y - M A X
//*-*-          S E T   R I G H T   E D G E   O R I E N T A T I O N

    ymin = y1[0];
    ymax = y1[0];
    for (i = 1; i <= n; ++i) {
	if (ymin > y1[i - 1])   ymin = y1[i - 1];
	if (ymax < y1[i - 1])   ymax = y1[i - 1];
	if (y1[i - 1] <= y1[i]) {x2[i - 1] = x1[i]; y2[i - 1] = y1[i];}
	else {
	    x2[i - 1] = x1[i - 1];
	    y2[i - 1] = y1[i - 1];
	    x1[i - 1] = x1[i];
	    y1[i - 1] = y1[i];
	}
    }
    if (ymin >= fNyrast) return;
    if (ymax < 0)        return;
    if (ymax >= fNyrast) ymax = fNyrast - 1;

//*-*-          S O R T   L I N E S

    for (i = 1; i < n; ++i) {
	if (y1[i] >= y1[i - 1]) continue;
	y = y1[i];
	k = 1;
	for (j = i - 1; j >= 1; --j) {
	    if (y < y1[j - 1]) continue;
	    k = j + 1;
	    break;
	}
	x = x1[i];
	xx = x2[i];
	yy = y2[i];
	for (j = i; j >= k; --j) {
	    x1[j] = x1[j - 1];
	    y1[j] = y1[j - 1];
	    x2[j] = x2[j - 1];
	    y2[j] = y2[j - 1];
	}
	x1[k - 1] = x;
	y1[k - 1] = y;
	x2[k - 1] = xx;
	y2[k - 1] = yy;
    }

//*-*-          S E T   I N I T I A L   V A L U E S

    for (i = 1; i <= n; ++i) {
	xcur[i - 1] = x1[i - 1];
	dy = y2[i - 1] - y1[i - 1];
	dx = x2[i - 1] - x1[i - 1];
	signdx = 1;
	if (dx < 0)   signdx = -1;
	if (dx < 0)   dx = -dx;
	if (dx <= dy) {
	    t = -(dy + 1) / 2 + dx;
	    if (t < 0) {
		test[i - 1] = t;
		xnex[i - 1] = xcur[i - 1];
	    } else {
		test[i - 1] = t - dy;
		xnex[i - 1] = xcur[i - 1] + signdx;
	    }
	} else if (dy != 0) {
	    step = (dx - 1) / (dy + dy) + 1;
	    test[i - 1] = step*dy - (dx + 1) / 2 - dx;
	    xnex[i - 1] = xcur[i - 1] + signdx*step;
	}
    }

//*-*-          L O O P   O N   S C A N   L I N E S

    nstart = 1;
    for (yscan = ymin; yscan <= ymax; ++yscan) {
	nx  = 0;
	nxa = 0;
	nxb = 13;
	for (i = nstart; i <= n; ++i) {
	    if (y1[i - 1] > yscan) goto L500;
	    if (y2[i - 1] <= yscan) {
               if (i == nstart)       ++nstart;
               if (y2[i - 1] != yscan)continue;
               --nxb;
               if (x2[i - 1] >= xcur[i - 1]) {
                  xscan[2*nxb - 2] = xcur[i - 1];
                  xscan[2*nxb - 1] = x2[i - 1];
               } else {
                  xscan[2*nxb - 2] = x2[i - 1];
                  xscan[2*nxb - 1] = xcur[i - 1];
               }
               continue;
            }

//*-*-          S T O R E   C U R R E N T  X
//*-*-          P R E P A R E   X   F O R   N E X T   S C A N - L I N E

	    ++nxa;
	    dy = y2[i - 1] - y1[i - 1];
	    dx = x2[i - 1] - x1[i - 1];
	    if (dx >= 0) {
		signdx = 1;
		xscan[2*nxa - 2] = xcur[i - 1];
		xscan[2*nxa - 1] = xnex[i - 1];
		if (xscan[2*nxa - 2] != xscan[2*nxa - 1]) {
		    --xscan[2*nxa - 1];
		}
	    } else {
		dx = -dx;
		signdx = -1;
		xscan[2*nxa - 2] = xnex[i - 1];
		xscan[2*nxa - 1] = xcur[i - 1];
		if (xscan[2*nxa - 2] != xscan[2*nxa - 1]) {
		    ++xscan[2*nxa - 2];
		}
	    }
	    xcur[i - 1] = xnex[i - 1];
	    if (dx <= dy) {
               test[i - 1] += dx;
               if (test[i - 1] < 0) continue;
               test[i - 1] -= dy;
               xnex[i - 1] += signdx;
               continue;
            }
	    step = dx / dy;
	    t = test[i - 1] + step*dy;
	    if (t >= 0) {
		test[i - 1] = t - dx;
		xnex[i - 1] += signdx*step;
	    } else {
		test[i - 1] = t + dy - dx;
		xnex[i - 1] += signdx*(step + 1);
	    }
	}

//*-*-          S O R T   P O I N T S   A L O N G   X

L500:
	if (yscan < 0) continue;
	ibase = yscan*fNxrast;
	if (nxa >= 2) {
           for (i = 1; i < nxa; ++i) {
              for (j = i; j >= 1; --j) {
                 if (xscan[2*j] >= xscan[2*j - 2]) continue;
                 x = xscan[2*j];
                 xscan[2*j] = xscan[2*j - 2];
                 xscan[2*j - 2] = x;
                 x = xscan[2*j - 1];
                 xscan[2*j + 1] = xscan[2*j - 1];
                 xscan[2*j - 1] = x;
              }
           }
           for (i = 1; i <= nxa; i += 2) {
              ++nx;
              xscan[2*nx - 2] = xscan[2*i - 2];
              x = xscan[2*i + 1];
              if (xscan[2*i - 1] > x) x = xscan[2*i - 1];
              xscan[2*nx - 1] = x;
           }
        }
	if (nxb <= 12) {
           for (i = nxb; i <= 12; ++i) {
              ++nx;
              xscan[2*nx - 2] = xscan[2*i - 2];
              xscan[2*nx - 1] = xscan[2*i - 1];
           }
        }
//*-*-          C O N C A T E N A T E   A N D   F I L L

        while (nx) {
           xx1 = xscan[2*nx - 2];
           xx2 = xscan[2*nx - 1];
           --nx;
           k = 1;
           while (k <= nx) {
              if ((xscan[2*k - 2] <= xx2 + 1) && (xscan[2*k - 1] >= xx1 - 1)) {
                 if (xscan[2*k - 2] < xx1)     xx1 = xscan[2*k - 2];
                 if (xscan[2*k - 1] > xx2)     xx2 = xscan[2*k - 1];
                 xscan[2*k - 2] = xscan[2*nx - 2];
                 xscan[2*k - 1] = xscan[2*nx - 1];
                 --nx;
              } else  ++k;
           }
           if (xx1 < 0)        xx1 = 0;
           if (xx2 >= fNxrast) xx2 = fNxrast - 1;
           nbit = xx2 - xx1 + 1;
           kbit = ibase + xx1;
           iw = kbit / 30;
           ib = kbit - iw*30 + 1;
           iw = iw + 1;
           nb = 30 - ib + 1;
           if (nb > nbit) nb = nbit;
           fRaster[iw - 1] = fRaster[iw - 1] | fMask[fJmask[nb - 1] + ib - 1];
           nbit -= nb;
           if (nbit) {
              while(nbit > 30) {
                 fRaster[iw] = fMask[464];
                 ++iw;
                 nbit += -30;
              }
              fRaster[iw] = fRaster[iw] | fMask[fJmask[nbit - 1]];
              ++iw;
           }
        }
    }
}

//______________________________________________________________________________
void TLego::FindLevelLines(Int_t np, Double_t *f, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-*Find level lines for face*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =========================                        *
//*-*                                                                     *
//*-*    Input: NP      - number of nodes                                 *
//*-*           F(3,NP) - face                                            *
//*-*           T(NP)   - additional function                             *
//*-*                                                                     *
//*-*    Error: number of points for line not equal 2                     *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, k, i1, i2, il, nl;
    Double_t tmin, tmax, d1, d2;

    /* Parameter adjustments */
    --t;
    f -= 4;

    /* Function Body */
    fNlines = 0;
    if (fNlevel == 0) return;
    nl = fNlevel;
    if (nl < 0) nl = -nl;

//*-*-         F I N D   Tmin   A N D   Tmax
    tmin = t[1];
    tmax = t[1];
    for (i = 2; i <= np; ++i) {
	if (t[i] < tmin) tmin = t[i];
	if (t[i] > tmax) tmax = t[i];
    }
    if (tmin >= fFunLevel[nl - 1]) return;
    if (tmax <= fFunLevel[0])      return;

//*-*-          F I N D   L E V E L S   L I N E S

    for (il = 1; il <= nl; ++il) {
	if (tmin >= fFunLevel[il - 1]) continue;
	if (tmax <= fFunLevel[il - 1]) return;
	if (fNlines >= 200)            return;
	++fNlines;
	fLevelLine[fNlines - 1] = il;
	k = 0;
	for (i = 1; i <= np; ++i) {
	    i1 = i;
	    i2 = i + 1;
	    if (i == np) i2 = 1;
	    d1 = t[i1] - fFunLevel[il - 1];
	    d2 = t[i2] - fFunLevel[il - 1];
	    if (d1) {
               if (d1*d2 < 0) goto L320;
               continue;
            }
	    ++k;
	    fPlines[(k + 2*fNlines)*3 - 9] = f[i1*3 + 1];
	    fPlines[(k + 2*fNlines)*3 - 8] = f[i1*3 + 2];
	    fPlines[(k + 2*fNlines)*3 - 7] = f[i1*3 + 3];
	    if (k == 1) continue;
	    goto L340;
L320:
	    ++k;
	    d1 /= t[i2] - t[i1];
	    d2 /= t[i2] - t[i1];
	    fPlines[(k + 2*fNlines)*3 - 9] = d2*f[i1*3 + 1] - d1*f[i2*3 + 1];
	    fPlines[(k + 2*fNlines)*3 - 8] = d2*f[i1*3 + 2] - d1*f[i2*3 + 2];
	    fPlines[(k + 2*fNlines)*3 - 7] = d2*f[i1*3 + 3] - d1*f[i2*3 + 3];
	    if (k != 1) goto L340;
	}
	if (k != 2) {
	    Error("FindLevelLines", "number of points for line not equal 2");
	    --fNlines;
	}
L340:
        if (il < 0) return;
    }
}


//______________________________________________________________________________
void TLego::FindPartEdge(Double_t *p1, Double_t *p2, Double_t f1, Double_t f2, Double_t fmin, Double_t fmax, Int_t &kpp, Double_t *pp)
{
//*-*-*-*-*-*-*-*-*-*-*-*-* Find part of edge *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                       =================                             *
//*-*                                                                     *
//*-*    Function: Find part of edge where function defined on this edge  *
//*-*              has value from FMIN to FMAX                            *
//*-*                                                                     *
//*-*    Input: P1(3) - 1st point                                         *
//*-*           P2(3) - 2nd point                                         *
//*-*           F1    - function value at 1st point                       *
//*-*           F2    - function value at 2nd point                       *
//*-*           FMIN  - min value of layer                                *
//*-*           FMAX  - max value of layer                                *
//*-*                                                                     *
//*-*    Output: KPP - current number of point                            *
//*-*            PP(3,*) - coordinates of new face                        *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t d1, d2;
    Int_t k1, k2, kk;

    /* Parameter adjustments */
    pp -= 4;
    --p2;
    --p1;

    k1 = 0;
    if (f1 < fmin)  k1 = -2;
    if (f1 == fmin) k1 = -1;
    if (f1 == fmax) k1 = 1;
    if (f1 > fmax)  k1 = 2;
    k2 = 0;
    if (f2 < fmin)  k2 = -2;
    if (f2 == fmin) k2 = -1;
    if (f2 == fmax) k2 = 1;
    if (f2 > fmax)  k2 = 2;
    kk = (k1 + 2)*5 + (k2 + 2) + 1;

//*-*-    K2:    -2  -1   0  +1  +2
//*-*-    K1:    -2 -1 0 +1 +2
    switch ((int)kk) {
	case 1:  return;
	case 2:  return;
	case 3:  goto L200;
	case 4:  goto L200;
	case 5:  goto L600;
	case 6:  goto L100;
	case 7:  goto L100;
	case 8:  goto L100;
	case 9:  goto L100;
	case 10:  goto L500;
	case 11:  goto L400;
	case 12:  goto L100;
	case 13:  goto L100;
	case 14:  goto L100;
	case 15:  goto L500;
	case 16:  goto L400;
	case 17:  goto L100;
	case 18:  goto L100;
	case 19:  goto L100;
	case 20:  goto L100;
	case 21:  goto L700;
	case 22:  goto L300;
	case 23:  goto L300;
	case 24:  return;
	case 25:  return;
    }

//*-*-          1 - S T   P O I N T

L100:
    ++kpp;
    pp[kpp*3 + 1] = p1[1];
    pp[kpp*3 + 2] = p1[2];
    pp[kpp*3 + 3] = p1[3];
    return;

//*-*-           I N T E R S E C T I O N   W I T H   Fmin

L200:
    ++kpp;
    d1 = (fmin - f1) / (f1 - f2);
    d2 = (fmin - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    return;

//*-*-           I N T E R S E C T I O N   W I T H   Fmax

L300:
    ++kpp;
    d1 = (fmax - f1) / (f1 - f2);
    d2 = (fmax - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    return;

//*-*-          1 - S T   P O I N T,   I N T E R S E C T I O N  WITH  Fmin

L400:
    ++kpp;
    pp[kpp*3 + 1] = p1[1];
    pp[kpp*3 + 2] = p1[2];
    pp[kpp*3 + 3] = p1[3];
    ++kpp;
    d1 = (fmin - f1) / (f1 - f2);
    d2 = (fmin - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    return;

//*-*-          1 - S T   P O I N T,   I N T E R S E C T I O N  WITH  Fmax

L500:
    ++kpp;
    pp[kpp*3 + 1] = p1[1];
    pp[kpp*3 + 2] = p1[2];
    pp[kpp*3 + 3] = p1[3];
    ++kpp;
    d1 = (fmax - f1) / (f1 - f2);
    d2 = (fmax - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    return;

//*-*-           I N T E R S E C T I O N   W I T H   Fmin, Fmax

L600:
    ++kpp;
    d1 = (fmin - f1) / (f1 - f2);
    d2 = (fmin - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    ++kpp;
    d1 = (fmax - f1) / (f1 - f2);
    d2 = (fmax - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    return;

//*-*-          I N T E R S E C T I O N   W I T H   Fmax, Fmin

L700:
    ++kpp;
    d1 = (fmax - f1) / (f1 - f2);
    d2 = (fmax - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
    ++kpp;
    d1 = (fmin - f1) / (f1 - f2);
    d2 = (fmin - f2) / (f1 - f2);
    pp[kpp*3 + 1] = d2*p1[1] - d1*p2[1];
    pp[kpp*3 + 2] = d2*p1[2] - d1*p2[2];
    pp[kpp*3 + 3] = d2*p1[3] - d1*p2[3];
}


//______________________________________________________________________________
void TLego::FindVisibleDraw(Double_t *r1, Double_t *r2)
{
//*-*-*-*-*-*-*-*-*Find visible parts of line (draw line)-*-*-*-*-*-*-*-*-*
//*-*              ======================================                 *
//*-*                                                                     *
//*-*    Input: R1(3)  - 1-st point of the line                           *
//*-*           R2(3)  - 2-nd point of the line                           *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t yy1u, yy2u;
    Int_t i, icase, i1, i2, icase1, icase2, iv, ifback;
    Double_t x1, x2, y1, y2, z1, z2, dd, di;
    Double_t dt, dy;
    Double_t tt, uu, ww, yy, yy1, yy2, yy1d, yy2d;
    Double_t *tn = 0;
    const Double_t kEpsil = 1.e-6;
    /* Parameter adjustments */
    --r2;
    --r1;

    if(gPad->GetView()) {
    	tn = gPad->GetView()->GetTN();
        x1 = tn[0]*r1[1] + tn[1]*r1[2] + tn[2]*r1[3]  + tn[3];
    	x2 = tn[0]*r2[1] + tn[1]*r2[2] + tn[2]*r2[3]  + tn[3];
    	y1 = tn[4]*r1[1] + tn[5]*r1[2] + tn[6]*r1[3]  + tn[7];
    	y2 = tn[4]*r2[1] + tn[5]*r2[2] + tn[6]*r2[3]  + tn[7];
    	z1 = tn[8]*r1[1] + tn[9]*r1[2] + tn[10]*r1[3] + tn[11];
    	z2 = tn[8]*r2[1] + tn[9]*r2[2] + tn[10]*r2[3] + tn[11];
    }
    else {
      Error("FindVisibleDraw", "no TView in current pad");
    	return;
    }

    ifback = 0;
    if (x1 >= x2) {
       ifback = 1;
       ww = x1;
       x1 = x2;
       x2 = ww;
       ww = y1;
       y1 = y2;
       y2 = ww;
       ww = z1;
       z1 = z2;
       z2 = ww;
    }
    fNT = 0;
    i1 = Int_t((x1 - fX0) / fDX) + 15;
    i2 = Int_t((x2 - fX0) / fDX) + 15;
    x1 = fX0 + (i1 - 1)*fDX;
    x2 = fX0 + (i2 - 1)*fDX;
  if (i1 != i2) {

//*-*-          F I N D   V I S I B L E   P A R T S   O F   T H E   L I N E

    di = (Double_t) (i2 - i1);
    dy = (y2 - y1) / di;
    dt = 1 / di;
    iv = -1;
    for (i = i1; i <= i2 - 1; ++i) {
	yy1 = y1 + dy*(i - i1);
	yy2 = yy1 + dy;
	yy1u = yy1 - fU[2*i - 2];
	yy1d = yy1 - fD[2*i - 2];
	yy2u = yy2 - fU[2*i - 1];
	yy2d = yy2 - fD[2*i - 1];
	tt = dt*(i - i1);
//*-*-         A N A L I Z E   L E F T   S I D E
	icase1 = 1;
	if (yy1u >  kEpsil) icase1 = 0;
	if (yy1d < -kEpsil) icase1 = 2;
	if ((icase1 == 0 || icase1 == 2) && iv <= 0) {
           iv = 1;
           ++fNT;
           fT[2*fNT - 2] = tt;
        }
	if (icase1 == 1 && iv >= 0) {
           iv = -1;
           fT[2*fNT - 1] = tt;
        }
//*-*-         A N A L I Z E   R I G H T   S I D E
	icase2 = 1;
	if (yy2u >  kEpsil) icase2 = 0;
	if (yy2d < -kEpsil) icase2 = 2;
	icase = icase1*3 + icase2;
	if (icase == 1) {
           iv = -1;
	   fT[2*fNT - 1] = tt + dt*(yy1u / (yy1u - yy2u));
        }
	if (icase == 2) {
           fT[2*fNT - 1] = tt + dt*(yy1u / (yy1u - yy2u));
	   ++fNT;
	   fT[2*fNT - 2] = tt + dt*(yy1d / (yy1d - yy2d));
        }
	if (icase == 3) {
           iv = 1;
	   ++fNT;
	   fT[2*fNT - 2] = tt + dt*(yy1u / (yy1u - yy2u));
        }
	if (icase == 5) {
           iv = 1;
	   ++fNT;
	   fT[2*fNT - 2] = tt + dt*(yy1d / (yy1d - yy2d));
        }
	if (icase == 6) {
           fT[2*fNT - 1] = tt + dt*(yy1d / (yy1d - yy2d));
	   ++fNT;
	   fT[2*fNT - 2] = tt + dt*(yy1u / (yy1u - yy2u));
        }
	if (icase == 7) {
           iv = -1;
           fT[2*fNT - 1] = tt + dt*(yy1d / (yy1d - yy2d));
        }
	if (fNT + 1 >= 100) break;
    }
    if (iv > 0) fT[2*fNT - 1] = 1;
 } else {

//*-*-          V E R T I C A L   L I N E

    fNT = 1;
    fT[0] = 0;
    fT[1] = 1;
    if (y2 <= y1) {
       if (y2 == y1) { fNT = 0; return;}
       ifback = 1 - ifback;
       yy = y1;
       y1 = y2;
       y2 = yy;
    }
    uu = fU[2*i1 - 2];
    dd = fD[2*i1 - 2];
    if (i1 != 1) {
       if (uu < fU[2*i1 - 3]) uu = fU[2*i1 - 3];
       if (dd > fD[2*i1 - 3]) dd = fD[2*i1 - 3];
    }
//*-*-         F I N D   V I S I B L E   P A R T   O F   L I N E
    if (y1 <  uu && y2 >  dd) {
       if (y1 >= dd && y2 <= uu) {fNT = 0; return;}
       fNT = 0;
       if (dd > y1) {
          ++fNT;
          fT[2*fNT - 2] = 0;
          fT[2*fNT - 1] = (dd - y1) / (y2 - y1);
       }
       if (uu < y2) {
          ++fNT;
          fT[2*fNT - 2] = (uu - y1) / (y2 - y1);
          fT[2*fNT - 1] = 1;
       }
    }
 }

    if (ifback == 0) return;
    if (fNT == 0)    return;
    for (i = 1; i <= fNT; ++i) {
	fT[2*i - 2] = 1 - fT[2*i - 2];
	fT[2*i - 1] = 1 - fT[2*i - 1];
    }
}


//______________________________________________________________________________
void TLego::FindVisibleLine(Double_t *p1, Double_t *p2, Int_t ntmax, Int_t &nt, Double_t *t)
{
//*-*-*-*-*-*-*-*Find visible part of a line ("RASTER SCREEN")*-*-*-*-*-*-*
//*-*            =============================================            *
//*-*                                                                     *
//*-*    Input: P1(2) - 1st point of the line                             *
//*-*           P2(2) - 2nd point of the line                             *
//*-*           NTMAX - max allowed number of visible segments            *
//*-*                                                                     *
//*-*    Output: NT     - number of visible segments of the line          *
//*-*            T(2,*) - visible segments                                *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t ddtt;
    Double_t tcur;
    Int_t i, incrx, ivis, x1, y1, x2, y2, ib, kb, dx, dy, iw, ix, iy, ifinve, dx2, dy2;
    Double_t t1, t2;
    Double_t dt;
    Double_t tt;
    /* Parameter adjustments */
    t -= 3;
    --p2;
    --p1;

    if (fIfrast) {
	nt = 1;
	t[3] = 0;
	t[4] = 1;
	return;
    }
    x1 = Int_t(fNxrast*((p1[1] - fXrast) / fDXrast) - 0.01);
    y1 = Int_t(fNyrast*((p1[2] - fYrast) / fDYrast) - 0.01);
    x2 = Int_t(fNxrast*((p2[1] - fXrast) / fDXrast) - 0.01);
    y2 = Int_t(fNyrast*((p2[2] - fYrast) / fDYrast) - 0.01);
    ifinve = 0;
    if (y1 > y2) {
	ifinve = 1;
	iw = x1;
	x1 = x2;
	x2 = iw;
	iw = y1;
	y1 = y2;
	y2 = iw;
    }
    nt   = 0;
    ivis = 0;
    if (y1 >= fNyrast) return;
    if (y2 < 0)        return;
    if (x1 >= fNxrast && x2 >= fNxrast) return;
    if (x1 < 0 && x2 < 0)               return;

//*-*-          S E T   I N I T I A L   V A L U E S

    incrx = 1;
    dx = x2 - x1;
    if (dx < 0) {
	dx = -dx;
	incrx = -1;
    }
    dy  = y2 - y1;
    dx2 = dx + dx;
    dy2 = dy + dy;
    if (dy > dx) goto L200;

//*-*-          D X   . G T .   D Y

    dt = 1 / (dx + 1);
    ddtt = dt*(float).5;
    tcur = -(Double_t)dt;
    tt = (Double_t) (-(dx + dy2));
    iy = y1;
    kb = iy*fNxrast + x1 - incrx;
    for (ix = x1; incrx < 0 ? ix >= x2 : ix <= x2; ix += incrx) {
	kb += incrx;
	tcur += dt;
	tt += dy2;
	if (tt >= 0) {
	    ++iy;
	    tt -= dx2;
	    kb += fNxrast;
	}
	if (iy < 0)        goto L110;
	if (iy >= fNyrast) goto L110;
	if (ix < 0)        goto L110;
	if (ix >= fNxrast) goto L110;
	iw = kb / 30;
	ib = kb - iw*30 + 1;
	if (fRaster[iw] & fMask[ib - 1]) goto L110;
	if (ivis > 0)      continue;
	ivis = 1;
	++nt;
	t[2*nt + 1] = tcur;
	continue;
L110:
	if (ivis == 0) continue;
	ivis = 0;
	t[2*nt + 2] = tcur;
	if (nt == ntmax)  goto L300;
    }
    if (ivis > 0) t[2*nt + 2] = tcur + dt + ddtt;
    goto L300;

//*-*-          D Y   . G T .   D X

L200:
    dt = 1 / (dy + 1);
    ddtt = dt*(float).5;
    tcur = -(Double_t)dt;
    tt = (Double_t) (-(dy + dx2));
    ix = x1;
    if (y2 >= fNyrast) y2 = fNyrast - 1;
    kb = (y1 - 1)*fNxrast + ix;
    for (iy = y1; iy <= y2; ++iy) {
	kb += fNxrast;
	tcur += dt;
	tt += dx2;
	if (tt >= 0) {
	    ix += incrx;
	    tt -= dy2;
	    kb += incrx;
	}
	if (iy < 0)        goto L210;
	if (ix < 0)        goto L210;
	if (ix >= fNxrast) goto L210;
	iw = kb / 30;
	ib = kb - iw*30 + 1;
	if (fRaster[iw] & fMask[ib - 1]) goto L210;
	if (ivis > 0) continue;
	ivis = 1;
	++nt;
	t[2*nt + 1] = tcur;
	continue;
L210:
	if (ivis == 0) continue;
	ivis = 0;
	t[2*nt + 2] = tcur;
	if (nt == ntmax) goto L300;
    }
    if (ivis > 0) t[2*nt + 2] = tcur + dt;

//*-*-          C H E C K   D I R E C T I O N   O F   P A R A M E T E R

L300:
    if (nt == 0) return;
    dt *= 11;
    if (t[3] <= dt) t[3] = 0;
    if (t[2*nt + 2] >= 1 - dt) t[2*nt + 2] = 1;
    if (ifinve == 0) return;
    for (i = 1; i <= nt; ++i) {
	t1 = t[2*i + 1];
	t2 = t[2*i + 2];
	t[2*i + 1] = 1 - t2;
	t[2*i + 2] = 1 - t1;
    }
}

//______________________________________________________________________________
void TLego::FrontBox(Double_t ang)
{
//*-*-*-*-*-*-*-*Draw forward faces of surrounding box & axes-*-*-*-*-*-*-*
//*-*            ============================================             *
//*-*                                                                     *
//*-*    Function: Draw forward faces of surrounding box & axes           *
//*-*                                                                     *
//*-*    References: AxisVertex, Gaxis                                    *
//*-*                                                                     *
//*-*    Input  ANG     - angle between X and Y axis                      *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */

    static Int_t iface1[4] = { 1,2,6,5 };
    static Int_t iface2[4] = { 2,3,7,6 };

    Double_t cosa, sina;
    Double_t r[24]	/* was [3][8] */, av[24]	/* was [3][8] */;
    Int_t icodes[3];
    Double_t fdummy[1];
    Int_t i, ix1, ix2, iy1, iy2, iz1, iz2;
    TView *view = 0;

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
      	Error("FrontBox", "no TView in current pad");
    		return;
    	}
    }


    cosa = TMath::Cos(kRad*ang);
    sina = TMath::Sin(kRad*ang);
	view->AxisVertex(ang, av, ix1, ix2, iy1, iy2, iz1, iz2);
    for (i = 1; i <= 8; ++i) {
	r[i*3 - 3] = av[i*3 - 3] + av[i*3 - 2] * cosa;
	r[i*3 - 2] = av[i*3 - 2] * sina;
	r[i*3 - 1] = av[i*3 - 1];
    }

//*-*-          D R A W   F O R W A R D   F A C E S

    icodes[0] = 0;
    icodes[1] = 0;
    icodes[2] = 0;
    (this->*fDrawFace)(icodes, r, 4, iface1, fdummy);
    (this->*fDrawFace)(icodes, r, 4, iface2, fdummy);
}

//______________________________________________________________________________
void TLego::GouraudFunction(Int_t ia, Int_t ib, Double_t *face, Double_t *t)
{
//*-*-*-*-*-* Find part of surface with luminosity in the corners*-*-*-*-*-*
//*-*         ===================================================
//*-*
//*-*              This routine is used for Gouraud shading
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t iphi;
   static Double_t f[108];	/* was [3][4][3][3] */
   Int_t i, j, k;
   Double_t r, s, x[36];	/* was [4][3][3] */
   Double_t y[36];	/* was [4][3][3] */
   Double_t z[36];	/* was [4][3][3] */
   Int_t incrx[3], incry[3];

   Double_t x1, x2, y1, y2, z1, z2, th, an[27];	/* was [3][3][3] */
   Double_t bn[12];    /* was [3][2][2] */

   Double_t rad;
   Double_t phi;
   Int_t ixt, iyt;

    /* Parameter adjustments */
   --t;
   face -= 4;

   iphi = 1;
   rad = TMath::ATan(1) * (float)4 / (float)180;

//*-*-        Find real cell indexes

   ixt = ia + Hparam.xfirst - 1;
   iyt = ib + Hparam.yfirst - 1;

//*-*-        Find increments of neighboring cells

   incrx[0] = -1;
   incrx[1] = 0;
   incrx[2] = 1;
   if (ixt == 1) incrx[0] = 0;
   if (ixt == Hparam.xlast - 1) incrx[2] = 0;
   incry[0] = -1;
   incry[1] = 0;
   incry[2] = 1;
   if (iyt == 1) incry[0] = 0;
   if (iyt == Hparam.ylast - 1) incry[2] = 0;

//*-*-        Find neighboring faces

   Int_t i1, i2;
   for (j = 1; j <= 3; ++j) {
      for (i = 1; i <= 3; ++i) {
         i1 = ia + incrx[i - 1];
         i2 = ib + incry[j - 1];
         SurfaceFunction(i1, i2, &f[((i + j*3 << 2) + 1)*3 - 51], &t[1]);
      }
   }

//*-*-       Set face

   for (k = 1; k <= 4; ++k) {
      for (i = 1; i <= 3; ++i) {
         face[i + k*3] = f[i + (k + 32)*3 - 52];
      }
   }

//*-*-       Find coordinates and normales

   for (j = 1; j <= 3; ++j) {
      for (i = 1; i <= 3; ++i) {
         for (k = 1; k <= 4; ++k) {
            if (Hoption.System == kPOLAR) {
               phi = f[iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
	       r = f[3 - iphi + (k + (i + j*3 << 2))*3 - 52];
	       x[k + (i + j*3 << 2) - 17] = r * TMath::Cos(phi);
	       y[k + (i + j*3 << 2) - 17] = r * TMath::Sin(phi);
	       z[k + (i + j*3 << 2) - 17] = f[(k + (i + j*3 << 2))*3 - 49];
            } else if (Hoption.System == kCYLINDRICAL) {
	       phi = f[iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
               r = f[(k + (i + j*3 << 2))*3 - 49];
	       x[k + (i + j*3 << 2) - 17] = r*TMath::Cos(phi);
	       y[k + (i + j*3 << 2) - 17] = r*TMath::Sin(phi);
	       z[k + (i + j*3 << 2) - 17] = f[3 - iphi + (k + (i + j*3 << 2))*3 - 52];
            } else if (Hoption.System == kSPHERICAL) {
	       phi = f[iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
	       th = f[3 - iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
	       r = f[(k + (i + j*3 << 2))*3 - 49];
	       x[k + (i + j*3 << 2) - 17] = r*TMath::Sin(th)*TMath::Cos(phi);
	       y[k + (i + j*3 << 2) - 17] = r*TMath::Sin(th)*TMath::Sin(phi);
	       z[k + (i + j*3 << 2) - 17] = r*TMath::Cos(th);
            } else if (Hoption.System == kRAPIDITY) {
	       phi = f[iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
	       th = f[3 - iphi + (k + (i + j*3 << 2))*3 - 52]*rad;
	       r = f[(k + (i + j*3 << 2))*3 - 49];
	       x[k + (i + j*3 << 2) - 17] = r*TMath::Cos(phi);
	       y[k + (i + j*3 << 2) - 17] = r*TMath::Sin(phi);
	       z[k + (i + j*3 << 2) - 17] = r*TMath::Cos(th) / TMath::Sin(th);
           } else {
	       x[k + (i + j*3 << 2) - 17] = f[(k + (i + j*3 << 2))*3 - 51];
	       y[k + (i + j*3 << 2) - 17] = f[(k + (i + j*3 << 2))*3 - 50];
	       z[k + (i + j*3 << 2) - 17] = f[(k + (i + j*3 << 2))*3 - 49];
           }
        }
        x1 = x[(i + j*3 << 2) - 14] - x[(i + j*3 << 2) - 16];
        x2 = x[(i + j*3 << 2) - 13] - x[(i + j*3 << 2) - 15];
        y1 = y[(i + j*3 << 2) - 14] - y[(i + j*3 << 2) - 16];
        y2 = y[(i + j*3 << 2) - 13] - y[(i + j*3 << 2) - 15];
        z1 = z[(i + j*3 << 2) - 14] - z[(i + j*3 << 2) - 16];
        z2 = z[(i + j*3 << 2) - 13] - z[(i + j*3 << 2) - 15];
        an[(i + j*3)*3 - 12] = y1*z2 - y2*z1;
        an[(i + j*3)*3 - 11] = z1*x2 - z2*x1;
        an[(i + j*3)*3 - 10] = x1*y2 - x2*y1;
        s = TMath::Sqrt(an[(i + j*3)*3 - 12]*an[(i + j*3)*3 - 12] + an[
                        (i + j*3)*3 - 11]*an[(i + j*3)*3 - 11] + an[(i
                        + j*3)*3 - 10]*an[(i + j*3)*3 - 10]);

        an[(i + j*3)*3 - 12] /= s;
        an[(i + j*3)*3 - 11] /= s;
        an[(i + j*3)*3 - 10] /= s;
     }
  }

//*-*-         Find average normals

   for (j = 1; j <= 2; ++j) {
      for (i = 1; i <= 2; ++i) {
         for (k = 1; k <= 3; ++k) {
            bn[k + (i + 2*j)*3 - 10] = an[k + (i + j*3)*3 - 13]
	      + an[k + (i + 1 + j*3)*3 - 13] + an[k + (i + 1 +
                          (j + 1)*3)*3 - 13] + an[k + (i + (j + 1)*3)*3 - 13];
         }
      }
   }

//*-*-        Set luminosity

   Luminosity(bn,     t[1]);
   Luminosity(&bn[3], t[2]);
   Luminosity(&bn[9], t[3]);
   Luminosity(&bn[6], t[4]);
}


//______________________________________________________________________________
void TLego::InitMoveScreen(Double_t xmin, Double_t xmax)
{
//*-*-*-*-*-*-*-*-*-*-*Initialize "MOVING SCREEN" method*-*-*-*-*-*-*-*-*-*
//*-*                  =================================                  *
//*-*                                                                     *
//*-*    Input: XMIN - left boundary                                      *
//*-*           XMAX - right boundary                                     *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    fX0 = xmin;
    fDX = (xmax - xmin) / 1000;
    for (Int_t i = 1; i <= 1000; ++i) {
	fU[2*i - 2] = (float)-999;
	fU[2*i - 1] = (float)-999;
	fD[2*i - 2] = (float)999;
	fD[2*i - 1] = (float)999;
    }
}
//______________________________________________________________________________
void TLego::InitRaster(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, Int_t nx, Int_t ny  )
{
//*-*-*Initialize hidden lines removal algorithm (RASTER SCREEN)*-*-*-*-*-*
//*-*  =========================================================          *
//*-*                                                                     *
//*-*    Input: XMIN - Xmin in the normalized coordinate system           *
//*-*           YMIN - Ymin in the normalized coordinate system           *
//*-*           XMAX - Xmax in the normalized coordinate system           *
//*-*           YMAX - Ymax in the normalized coordinate system           *
//*-*           NX   - number of pixels along X                           *
//*-*           NY   - number of pixels along Y                           *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, j, k, ib, nb;

    fNxrast = nx;
    fNyrast = ny;
    fXrast  = xmin;
    fDXrast = xmax - xmin;
    fYrast  = ymin;
    fDYrast = ymax - ymin;

//*-*-  Create buffer for raster
    Int_t buffersize = nx*ny/30 + 1;
    fRaster = new Int_t[buffersize];

//*-*-          S E T   M A S K S

    k = 0;
    Int_t pow2 = 1;
    for (i = 1; i <= 30; ++i) {
	fJmask[i - 1] = k;
	k = k + 30 - i + 1;
	fMask[i - 1] = pow2;
        pow2 *= 2;
    }
    j = 30;
    for (nb = 2; nb <= 30; ++nb) {
	for (ib = 1; ib < 30 - nb; ++ib) {
	    k = 0;
	    for (i = ib; i <= ib + nb - 1; ++i) k = k | fMask[i - 1];
	    ++j;
	    fMask[j - 1] = k;
	}
    }

//*-*-          C L E A R   R A S T E R   S C R E E N

    ClearRaster();

}


//______________________________________________________________________________
void TLego::LegoFunction(Int_t ia, Int_t ib, Int_t &nv, Double_t *ab, Double_t *vv, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Service function for Legos-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==========================
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t i, j, ixt, iyt;
    Double_t xval1l, xval2l, yval1l,  yval2l;
    Double_t rinrad = gStyle->GetLegoInnerR();
    Double_t dangle = 10; //Delta angle for Rapidity option

    /* Parameter adjustments */
    t -= 5;
    --vv;
    ab -= 3;

    ixt = ia + Hparam.xfirst - 1;
    iyt = ib + Hparam.yfirst - 1;

//*-*-             Compute the cell position in cartesian coordinates
//*-*-             and compute the LOG if necessary

    Double_t xwid = gCurrentHist->GetXaxis()->GetBinWidth(ixt);
    Double_t ywid = gCurrentHist->GetYaxis()->GetBinWidth(iyt);
    ab[3] = gCurrentHist->GetXaxis()->GetBinLowEdge(ixt) + xwid*Hparam.baroffset;
    ab[4] = gCurrentHist->GetYaxis()->GetBinLowEdge(iyt) + ywid*Hparam.baroffset;
    ab[5] = ab[3] + xwid*Hparam.barwidth;
    ab[8] = ab[4] + ywid*Hparam.barwidth;

    if (Hoption.Logx) {
       ab[3]  = TMath::Log10(ab[3]);
       ab[5]  = TMath::Log10(ab[5]);
    }
	xval1l = Hparam.xmin;
	xval2l = Hparam.xmax;
    if (Hoption.Logy) {
       ab[4]  = TMath::Log10(ab[4]);
       ab[8]  = TMath::Log10(ab[8]);
    }
	yval1l = Hparam.ymin;
	yval2l = Hparam.ymax;

//*-*-       Transform the cell position in the required coordinate system

    if (Hoption.System == kPOLAR) {
	ab[3] = 360*(ab[3] - xval1l) / (xval2l - xval1l);
	ab[5] = 360*(ab[5] - xval1l) / (xval2l - xval1l);
	ab[4] = (ab[4] - yval1l) / (yval2l - yval1l);
	ab[8] = (ab[8] - yval1l) / (yval2l - yval1l);
    } else if (Hoption.System == kCYLINDRICAL) {
	ab[3] = 360*(ab[3] - xval1l) / (xval2l - xval1l);
	ab[5] = 360*(ab[5] - xval1l) / (xval2l - xval1l);
    } else if (Hoption.System == kSPHERICAL) {
	ab[3] = 360*(ab[3] - xval1l) / (xval2l - xval1l);
	ab[5] = 360*(ab[5] - xval1l) / (xval2l - xval1l);
	ab[4] = 180*(ab[4] - yval1l) / (yval2l - yval1l);
	ab[8] = 180*(ab[8] - yval1l) / (yval2l - yval1l);
    } else if (Hoption.System == kRAPIDITY) {
	ab[3] = 360*(ab[3] - xval1l) / (xval2l - xval1l);
	ab[5] = 360*(ab[5] - xval1l) / (xval2l - xval1l);
	ab[4] = (180 - dangle*2)*(ab[4] - yval1l) / (yval2l - yval1l) + dangle;
	ab[8] = (180 - dangle*2)*(ab[8] - yval1l) / (yval2l - yval1l) + dangle;
    }

//*-*-             Complete the cell coordinates

    ab[6]  = ab[4];
    ab[7]  = ab[5];
    ab[9]  = ab[3];
    ab[10] = ab[8];

//*-*-              Get the content of the table, and loop on the
//*-*-              stack if necessary.

    vv[1] = Hparam.zmin;
    vv[2] = gCurrentHist->GetCellContent(ixt, iyt);
    TObjArray *stack = gCurrentHist->GetPainter()->GetStack();
    Int_t nids = 0; //not yet implemented
    if (stack) nids = stack->GetEntriesFast();
    if (nids) {
	for (i = 2; i <= nids + 1; ++i) {
            TH1 *hid = (TH1*)stack->At(i-2);
            vv[i + 1] = hid->GetCellContent(ixt, iyt) + vv[i];
	    vv[i + 1] = TMath::Max(Hparam.zmin, vv[i + 1]);
	    //vv[i + 1] = TMath::Min(Hparam.zmax, vv[i + 1]);
	}
    }

    nv = nids + 2;
    for (i = 2; i <= nv; ++i) {
	if (Hoption.Logz) {
            if (vv[i] > 0)
               vv[i] = TMath::Max(Hparam.zmin, (Double_t)TMath::Log10(vv[i]));
            else
               vv[i] = Hparam.zmin;
            vv[i] = TMath::Min(vv[i], Hparam.zmax);
        } else {
            vv[i] = TMath::Max(Hparam.zmin, vv[i]);
            vv[i] = TMath::Min(Hparam.zmax, vv[i]);
        }
    }

    if (!Hoption.Logz) {
	i = 3;
	while (i <= nv) {
	    if (vv[i] < vv[i - 1]) {
		vv[i - 1] = vv[i];
		i = 3;
                continue;
	    }
	    ++i;
	}
    }

//*-*-          For cylindrical, spherical and pseudo-rapidity, the content
//*-*-          is mapped onto the radius

    if (Hoption.System == kCYLINDRICAL || Hoption.System == kSPHERICAL || Hoption.System == kRAPIDITY) {
	for (i = 1; i <= nv; ++i) {
	    vv[i] = (1 - rinrad)*((vv[i] - Hparam.zmin) /
		    (Hparam.zmax - Hparam.zmin)) + rinrad;
	}
    }

    for (i = 1; i <= nv; ++i) {
	for (j = 1; j <= 4; ++j) t[j + (i << 2)] = vv[i];
    }

}

//______________________________________________________________________________
void TLego::LegoCartesian(Double_t ang, Int_t nx, Int_t ny, const char *chopt)
{
//*-*-*-*-*-*-*Draw stack of lego-plots in cartesian coordinates*-*-*-*-*-*
//*-*          =================================================          *
//*-*                                                                     *
//*-*    Input: ANG      - angle between X ang Y                          *
//*-*           NX       - number of cells along X                        *
//*-*           NY       - number of cells along Y                        *
//*-*                                                                     *
//*-*           FUN(IX,IY,NV,XY,V,T) - external routine                   *
//*-*             IX     - X number of the cell                           *
//*-*             IY     - Y number of the cell                           *
//*-*             NV     - number of values for given cell                *
//*-*             XY(2,4)- coordinates of the cell corners                *
//*-*             V(NV)  - cell values                                    *
//*-*             T(4,NV)- additional function (for example: temperature) *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this line                  *
//*-*               ICODES(1) - IX                                        *
//*-*               ICODES(2) - IY                                        *
//*-*               ICODES(3) - IV                                        *
//*-*               ICODES(4) - side: 1-face,2-right,3-back,4-left,       *
//*-*                                 5-bottom, 6-top                     *
//*-*               XYZ(3,*)  - coordinates of nodes                      *
//*-*               NP        - number of nodes                           *
//*-*               IFACE(NP) - face                                      *
//*-*                T(4)   - additional function (here Z-coordinate)      *
//*-*                                                                     *
//*-*           CHOPT - options: 'BF' - from BACK to FRONT                *
//*-*                            'FB' - from FRONT to BACK                *
//*-*                                                                     *
//Begin_Html
/*
<img src="gif/Lego1Cartesian.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Double_t cosa, sina;
    Int_t ivis[4], iface[4];
    Double_t tface[4], v[20];
    Int_t incrx, incry, i1, k1, k2, ix1, iy1, ix2, iy2, i, iv, ix, iy, nv;
    Double_t tt[80]	/* was [4][20] */;
    Int_t icodes[4];
    Double_t zn, xy[8]	/* was [2][4] */;
    Double_t xyz[24]	/* was [3][8] */;
    Double_t *tn = 0;
    TView *view = 0;
	
    sina = TMath::Sin(ang*kRad);
    cosa = TMath::Cos(ang*kRad);

//*-*-          F I N D   T H E   M O S T   L E F T   P O I N T

	if(gPad) {
		view = gPad->GetView();
		if(!view) {
			Error("LegoCartesian", "no TView in current pad");
			return;
		}
		tn = gPad->GetView()->GetTN();
	}

    i1 = 1;
    if (tn[0] < 0) i1 = 2;
    if (tn[0]*cosa + tn[1]*sina < 0) i1 = 5 - i1;

//*-*-          D E F I N E   O R D E R   O F   D R A W I N G

    if (*chopt == 'B' || *chopt == 'b') {
	incrx = -1;
	incry = -1;
    } else {
	incrx = 1;
	incry = 1;
    }
    if (i1 == 1 || i1 == 2) incrx = -incrx;
    if (i1 == 2 || i1 == 3) incry = -incry;
    ix1 = 1;
    iy1 = 1;
    if (incrx < 0) ix1 = nx;
    if (incry < 0) iy1 = ny;
    ix2 = nx - ix1 + 1;
    iy2 = ny - iy1 + 1;

//*-*-          F I N D   V I S I B I L I T Y   O F   S I D E S

    ivis[0] = 0;
    ivis[1] = 0;
    ivis[2] = 0;
    ivis[3] = 0;
    nv      = 0;
    view->FindNormal(0, 1, 0, zn);
    if (zn < 0) ivis[0] = 1;
    if (zn > 0) ivis[2] = 1;
    view->FindNormal(sina, cosa, 0, zn);
    if (zn > 0) ivis[1] = 1;
    if (zn < 0) ivis[3] = 1;

//*-*-          D R A W   S T A C K   O F   L E G O - P L O T S

    THistPainter *painter = (THistPainter*)gCurrentHist->GetPainter();
    for (iy = iy1; incry < 0 ? iy >= iy2 : iy <= iy2; iy += incry) {
	for (ix = ix1; incrx < 0 ? ix >= ix2 : ix <= ix2; ix += incrx) {
	    if (!painter->IsInside(ix,iy)) continue;
            (this->*fLegoFunction)(ix, iy, nv, xy, v, tt);
	    if (nv < 2 || nv > 20) continue;
	    icodes[0] = ix;
	    icodes[1] = iy;
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = xy[2*i - 2] + xy[2*i - 1]*cosa;
		xyz[i*3 - 2] = xy[2*i - 1]*sina;
		xyz[(i + 4)*3 - 3] = xyz[i*3 - 3];
		xyz[(i + 4)*3 - 2] = xyz[i*3 - 2];
	    }
//*-*-         D R A W   S T A C K
	    for (iv = 1; iv < nv; ++iv) {
		for (i = 1; i <= 4; ++i) {
		    xyz[i*3 - 1] = v[iv - 1];
		    xyz[(i + 4)*3 - 1] = v[iv];
		}
		if (v[iv - 1] == v[iv]) continue;
		icodes[2] = iv;
		for (i = 1; i <= 4; ++i) {
		    if (ivis[i - 1] == 0) continue;
		    k1 = i;
		    k2 = i + 1;
		    if (i == 4) k2 = 1;
		    icodes[3] = k1;
		    iface[0] = k1;
		    iface[1] = k2;
		    iface[2] = k2 + 4;
		    iface[3] = k1 + 4;
		    tface[0] = tt[k1 + (iv << 2) - 5];
		    tface[1] = tt[k2 + (iv << 2) - 5];
		    tface[2] = tt[k2 + (iv + 1 << 2) - 5];
		    tface[3] = tt[k1 + (iv + 1 << 2) - 5];
		    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
		}
	    }
//*-*-         D R A W   B O T T O M   F A C E
	    view->FindNormal(0, 0, 1, zn);
	    if (zn < 0) {
		icodes[2] = 1;
		icodes[3] = 5;
		for (i = 1; i <= 4; ++i) {
		    xyz[i*3 - 1] = v[0];
		    iface[i - 1] = 5 - i;
		    tface[i - 1] = tt[5 - i - 1];
		}
		(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	    }
//*-*-         D R A W   T O P   F A C E
	    if (zn > 0) {
		icodes[2] = nv - 1;
		icodes[3] = 6;
		for (i = 1; i <= 4; ++i) {
		    iface[i - 1] = i + 4;
		    tface[i - 1] = tt[i + (nv << 2) - 5];
		}
		(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	    }
	}
    }
}

//______________________________________________________________________________
void TLego::LegoPolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*-* Draw stack of lego-plots in polar coordinates *-*-*-*-*-*-*
//*-*           =============================================             *
//*-*                                                                     *
//*-*    Input: IORDR - order of variables (0 - R,PHI; 1 - PHI,R)         *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NB    - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,NV,AB,V,TT) - external routine                  *
//*-*             IA      - cell number for 1st variable                  *
//*-*             IB      - cell number for 2nd variable                  *
//*-*             NV      - number of values for given cell               *
//*-*             AB(2,4) - coordinates of the cell corners               *
//*-*             V(NV)   - cell values                                   *
//*-*             TT(4,*) - additional function                           *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*               ICODES(3) - IV                                        *
//*-*               ICODES(4) - side: 1-internal,2-right,3-external,4-left*
//*-*                                 5-bottom, 6-top                     *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*            CHOPT       - options: 'BF' - from BACK to FRONT         *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//Begin_Html
/*
<img src="gif/Lego1Polar.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t iphi, jphi, kphi, incr, nphi, ivis[6], iopt, iphi1, iphi2, iface[4], i, j;
    Double_t tface[4], v[20];
    Int_t incrr, k1, k2, ia, ib, ir1, ir2;
    Double_t ab[8]	/* was [2][4] */;
    Int_t ir, jr, iv, nr, nv, icodes[4];
    Double_t tt[80]	/* was [4][20] */;
    Double_t xyz[24]	/* was [3][8] */;
    TView *view = 0;
    ia = ib = 0;	
	if(gPad) {
		view = gPad->GetView();
		if(!view) {
			Error("LegoPolar", "no TView in current pad");
			return;
		}
	}
	
    if (iordr == 0) {
	jr   = 1;
	jphi = 2;
	nr   = na;
	nphi = nb;
    } else {
	jr   = 2;
	jphi = 1;
	nr   = nb;
	nphi = na;
    }
    if (nphi > 180) {
       Error("LegoPolar", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-     P R E P A R E   P H I   A R R A Y
//*-*-     F I N D    C R I T I C A L   S E C T O R S

    nv   = 0;
    kphi = nphi;
    if (iordr == 0) ia = nr;
    if (iordr != 0) ib = nr;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (i == 1) fAphi[0] = ab[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + ab[jphi - 1]) / (float)2.;
	fAphi[i] = ab[jphi + 3];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-      E N C O D E   V I S I B I L I T Y   O F   S I D E S
//*-*-      A N D   O R D E R   A L O N G   R

    for (i = 1; i <= nphi; ++i) {
	if (!iordr) ib = i;
	if (iordr)  ia = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	SideVisibilityEncode(iopt, ab[jphi - 1]*kRad, ab[jphi + 3]*kRad, fAphi[i - 1]);
    }

//*-*-       D R A W   S T A C K   O F   L E G O - P L O T S

    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L300;

//*-*-     D E C O D E   V I S I B I L I T Y   O F   S I D E S
    SideVisibilityDecode(fAphi[iphi - 1], ivis[0], ivis[1], ivis[2], ivis[3], ivis[4], ivis[5], incrr);
    ir1 = 1;
    if (incrr < 0) ir1 = nr;
    ir2 = nr - ir1 + 1;
//*-*-      D R A W   L E G O S   F O R   S E C T O R
    for (ir = ir1; incrr < 0 ? ir >= ir2 : ir <= ir2; ir += incrr) {
	if (iordr == 0) { ia = ir;   ib = iphi; }
	else            { ia = iphi; ib = ir; }
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (nv < 2 || nv > 20) continue;
	icodes[0] = ia;
	icodes[1] = ib;
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr != 0 && i == 2) j = 4;
	    if (iordr != 0 && i == 4) j = 2;
	    xyz[j*3 - 3] = ab[jr + 2*i - 3]*TMath::Cos(ab[jphi + 2*i - 3]*kRad);
	    xyz[j*3 - 2] = ab[jr + 2*i - 3]*TMath::Sin(ab[jphi + 2*i - 3]*kRad);
	    xyz[(j + 4)*3 - 3] = xyz[j*3 - 3];
	    xyz[(j + 4)*3 - 2] = xyz[j*3 - 2];
	}
//*-*-      D R A W   S T A C K
	for (iv = 1; iv < nv; ++iv) {
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 1] = v[iv - 1];
		xyz[(i + 4)*3 - 1] = v[iv];
	    }
	    if (v[iv - 1] >= v[iv]) continue;
	    icodes[2] = iv;
	    for (i = 1; i <= 4; ++i) {
		if (ivis[i - 1] == 0) continue;
		k1 = i - 1;
		if (i == 1) k1 = 4;
		k2 = i;
		if (xyz[k1*3 - 3] == xyz[k2*3 - 3] && xyz[k1*3 - 2] ==
			xyz[k2*3 - 2]) continue;
		iface[0] = k1;
		iface[1] = k2;
		iface[2] = k2 + 4;
		iface[3] = k1 + 4;
		tface[0] = tt[k1 + (iv << 2) - 5];
		tface[1] = tt[k2 + (iv << 2) - 5];
		tface[2] = tt[k2 + (iv + 1 << 2) - 5];
		tface[3] = tt[k1 + (iv + 1 << 2) - 5];
		icodes[3] = i;
		(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	    }
	}
//*-*-         D R A W   B O T T O M   F A C E
	if (ivis[4] != 0) {
	    icodes[2] = 1;
	    icodes[3] = 5;
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 1] = v[0];
		iface[i - 1] = 5 - i;
		tface[i - 1] = tt[5 - i - 1];
	    }
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	}
//*-*-         D R A W   T O P   F A C E
	if (ivis[5] != 0) {
	    icodes[2] = nv - 1;
	    icodes[3] = 6;
	    for (i = 1; i <= 4; ++i) {
		iface[i - 1] = i + 4;
		tface[i - 1] = tt[i + (nv << 2) - 5];
	    }
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	}
    }
//*-*-      N E X T   P H I
L300:
    iphi += incr;
    if (iphi == 0)      iphi = kphi;
    if (iphi > kphi)    iphi = 1;
    if (iphi != iphi2)  goto L100;
    if (incr == 0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L300;
}

//______________________________________________________________________________
void TLego::LegoCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*Draw stack of lego-plots in cylindrical coordinates*-*-*-*-*-*
//*-*        ===================================================          *
//*-*                                                                     *
//*-*    Input: IORDR - order of variables (0 - Z,PHI; 1 - PHI,Z)         *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NPHI  - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,NV,AB,V,TT) - external routine                  *
//*-*             IA      - cell number for 1st variable                  *
//*-*             IB      - cell number for 2nd variable                  *
//*-*             NV      - number of values for given cell               *
//*-*             AB(2,4) - coordinates of the cell corners               *
//*-*             V(NV)   - cell values                                   *
//*-*             TT(4,*) - additional function                           *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*               ICODES(3) - IV                                        *
//*-*               ICODES(4) - side: 1,2,3,4 - ordinary sides            *
//*-*                                 5-bottom,6-top                      *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT       - options: 'BF' - from BACK to FRONT          *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//Begin_Html
/*
<img src="gif/Lego1Cylindrical.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t iphi, jphi, kphi, incr, nphi, ivis[6], iopt, iphi1, iphi2, iface[4], i, j;
    Double_t tface[4], v[20], z;
    Double_t ab[8]	/* was [2][4] */;
    Int_t ia, ib, idummy, iz1, iz2, nz, incrz, k1, k2, nv;
    Int_t iv, iz, jz, icodes[4];
    Double_t tt[80]	/* was [4][20] */;
    Double_t cosphi[4];
    Double_t sinphi[4];
    Double_t xyz[24]	/* was [3][8] */;
    TView *view = 0;
    ia = ib = 0;	
	
	if(gPad) {
		view = gPad->GetView();
		if(!view) {
			Error("LegoCylindrical", "no TView in current pad");
			return;
		}
	}
    if (iordr == 0) {
	jz   = 1;
	jphi = 2;
	nz   = na;
	nphi = nb;
    } else {
	jz   = 2;
	jphi = 1;
	nz   = nb;
	nphi = na;
    }
    if (nphi > 180) {
       Error("LegoCylindrical", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-       P R E P A R E   P H I   A R R A Y
//*-*-       F I N D    C R I T I C A L   S E C T O R S

    nv   = 0;
    kphi = nphi;
    if (iordr == 0) ia = nz;
    if (iordr != 0) ib = nz;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (i == 1)  fAphi[0] = ab[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + ab[jphi - 1]) / (float)2.;
	fAphi[i] = ab[jphi + 3];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-      E N C O D E   V I S I B I L I T Y   O F   S I D E S
//*-*-      A N D   O R D E R   A L O N G   R

    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	SideVisibilityEncode(iopt, ab[jphi - 1]*kRad, ab[jphi + 3]*kRad, fAphi[i - 1]);
    }

//*-*-       F I N D   O R D E R   A L O N G   Z

    incrz = 1;
    iz1 = 1;
    view->FindNormal(0, 0, 1, z);
    if (z <= 0 && iopt == 1 || z > 0 && iopt == 2) {
	incrz = -1;
	iz1 = nz;
    }
    iz2 = nz - iz1 + 1;

//*-*-       D R A W   S T A C K   O F   L E G O - P L O T S

    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L400;
//*-*-     D E C O D E   V I S I B I L I T Y   O F   S I D E S
    idummy = 0;
    SideVisibilityDecode(fAphi[iphi - 1], ivis[4], ivis[1], ivis[5], ivis[3], ivis[0], ivis[2], idummy);
    for (iz = iz1; incrz < 0 ? iz >= iz2 : iz <= iz2; iz += incrz) {
	if (iordr == 0) {ia = iz;   ib = iphi;}
        else            {ia = iphi; ib = iz;}
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (nv < 2 || nv > 20) continue;
	icodes[0] = ia;
	icodes[1] = ib;
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr != 0 && i == 2) j = 4;
	    if (iordr != 0 && i == 4) j = 2;
	    cosphi[j - 1] = TMath::Cos(ab[jphi + 2*i - 3]*kRad);
	    sinphi[j - 1] = TMath::Sin(ab[jphi + 2*i - 3]*kRad);
	    xyz[j*3 - 1] = ab[jz + 2*i - 3];
	    xyz[(j + 4)*3 - 1] = ab[jz + 2*i - 3];
	}
//*-*-      D R A W   S T A C K
	for (iv = 1; iv < nv; ++iv) {
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = v[iv - 1]*cosphi[i - 1];
		xyz[i*3 - 2] = v[iv - 1]*sinphi[i - 1];
		xyz[(i + 4)*3 - 3] = v[iv]*cosphi[i - 1];
		xyz[(i + 4)*3 - 2] = v[iv]*sinphi[i - 1];
	    }
	    if (v[iv - 1] >= v[iv]) continue;
	    icodes[2] = iv;
	    for (i = 1; i <= 4; ++i) {
		if (ivis[i - 1] == 0) continue;
		k1 = i;
		k2 = i - 1;
		if (i == 1) k2 = 4;
		iface[0] = k1;
		iface[1] = k2;
		iface[2] = k2 + 4;
		iface[3] = k1 + 4;
		tface[0] = tt[k1 + (iv << 2) - 5];
		tface[1] = tt[k2 + (iv << 2) - 5];
		tface[2] = tt[k2 + (iv + 1 << 2) - 5];
		tface[3] = tt[k1 + (iv + 1 << 2) - 5];
		icodes[3] = i;
		(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	    }
	}
//*-*-       D R A W   B O T T O M   F A C E
	if (ivis[4] != 0 && v[0] > 0) {
	    icodes[2] = 1;
	    icodes[3] = 5;
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = v[0]*cosphi[i - 1];
		xyz[i*3 - 2] = v[0]*sinphi[i - 1];
		iface[i - 1] = i;
		tface[i - 1] = tt[i - 1];
	    }
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	}
//*-*-      D R A W   T O P   F A C E
	if (ivis[5] != 0 && v[nv - 1] > 0) {
	    icodes[2] = nv - 1;
	    icodes[3] = 6;
	    for (i = 1; i <= 4; ++i) {
		iface[i - 1] = 5 - i + 4;
		tface[i - 1] = tt[5 - i + (nv << 2) - 5];
	    }
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	}
    }
//*-*-      N E X T   P H I
L400:
    iphi += incr;
    if (iphi == 0)     iphi = kphi;
    if (iphi > kphi)   iphi = 1;
    if (iphi != iphi2) goto L100;
    if (incr == 0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L400;
}

//______________________________________________________________________________
void TLego::LegoSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*-*-*-*Draw stack of lego-plots spheric coordinates-*-*-*-*-*-*
//*-*              ============================================           *
//*-*                                                                     *
//*-*    Input: IPSDR - pseudo-rapidity flag                              *
//*-*           IORDR - order of variables (0 - THETA,PHI; 1 - PHI,THETA) *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NB    - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,NV,AB,V,TT) - external routine                  *
//*-*             IA      - cell number for 1st variable                  *
//*-*             IB      - cell number for 2nd variable                  *
//*-*             NV      - number of values for given cell               *
//*-*             AB(2,4) - coordinates of the cell corners               *
//*-*             V(NV)   - cell values                                   *
//*-*             TT(4,*) - additional function                           *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*               ICODES(3) - IV                                        *
//*-*               ICODES(4) - side: 1,2,3,4 - ordinary sides            *
//*-*                                 5-bottom,6-top                      *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT       - options: 'BF' - from BACK to FRONT          *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t iphi, jphi, kphi, incr, nphi, ivis[6], iopt, iphi1, iphi2, iface[4], i, j;
    Double_t tface[4], v[20], costh[4];
    Double_t sinth[4];
    Int_t k1, k2, ia, ib, incrth, ith, jth, kth, nth, mth, ith1, ith2, nv;
    Double_t ab[8]	/* was [2][4] */;
    Double_t th;
    Int_t iv, icodes[4];
    Double_t tt[80]	/* was [4][20] */, zn, cosphi[4];
    Double_t sinphi[4], th1, th2, phi;
    Double_t xyz[24]	/* was [3][8] */, phi1, phi2;
    TView *view = 0;
    ia = ib = 0;	
	
	if(gPad) {
		view = gPad->GetView();
		if(!view) {
			Error("LegoSpherical", "no TView in current pad");
			return;
		}
	}

    if (iordr == 0) {
	jth  = 1;
	jphi = 2;
	nth  = na;
	nphi = nb;
    } else {
	jth  = 2;
	jphi = 1;
	nth  = nb;
	nphi = na;
    }
    if (nth > 180) {
       Error("LegoSpherical", "too many THETA sectors (%d)", nth);
       return;
    }
    if (nphi > 180) {
       Error("LegoSpherical", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-       P R E P A R E   P H I   A R R A Y
//*-*-       F I N D    C R I T I C A L   P H I   S E C T O R S

    nv  = 0;
    kphi = nphi;
    mth = nth / 2;
    if (mth == 0)    mth = 1;
    if (iordr == 0) ia = mth;
    if (iordr != 0) ib = mth;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (i == 1)  fAphi[0] = ab[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + ab[jphi - 1]) / (float)2.;
	fAphi[i] = ab[jphi + 3];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-       P R E P A R E   T H E T A   A R R A Y

    if (iordr == 0) ib = 1;
    if (iordr != 0) ia = 1;
    for (i = 1; i <= nth; ++i) {
	if (iordr == 0) ia = i;
	if (iordr != 0) ib = i;
	(this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
	if (i == 1) fAphi[0] = ab[jth - 1];
	fAphi[i - 1] = (fAphi[i - 1] + ab[jth - 1]) / (float)2.;
	fAphi[i] = ab[jth + 3];
    }

//*-*-       D R A W   S T A C K   O F   L E G O - P L O T S

    kth = nth;
//printf("nth=%d nv=%d iordr=%d\n",nth,nv,iordr);

    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L500;

//*-*-      F I N D    C R I T I C A L   T H E T A   S E C T O R S
    if (!iordr) {ia = mth;	ib = iphi; }
    else        {ia = iphi;ib = mth;  }
    (this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
    phi = (ab[jphi - 1] + ab[jphi + 3]) / (float)2.;
    view->FindThetaSectors(iopt, phi, kth, fAphi, ith1, ith2);
    incrth = 1;
    ith = ith1;
L200:
    if (ith > nth)   goto L400;
    if (iordr == 0) ia = ith;
    if (iordr != 0) ib = ith;
//printf("na=%d nb=%d ith=%d iphi=%d kth=%d iphi1=%d iphi2=%d ith1=%d ith2=%d\n",
//     na,nb,ith,iphi,kth,iphi1,iphi2,ith1,ith2);
    (this->*fLegoFunction)(ia, ib, nv, ab, v, tt);
    if (nv < 2 || nv > 20) goto L400;

//*-*-      D E F I N E   V I S I B I L I T Y   O F   S I D E S
    for (i = 1; i <= 6; ++i) ivis[i - 1] = 0;

    phi1 = kRad*ab[jphi - 1];
    phi2 = kRad*ab[jphi + 3];
    th1  = kRad*ab[jth - 1];
    th2  = kRad*ab[jth + 3];
    view->FindNormal(TMath::Sin(phi1), -TMath::Cos(phi1), 0, zn);
    if (zn > 0) ivis[1] = 1;
    view->FindNormal(-TMath::Sin(phi2), TMath::Cos(phi2), 0, zn);
    if (zn > 0) ivis[3] = 1;
    phi = (phi1 + phi2) / (float)2.;
    view->FindNormal(-TMath::Cos(phi)*TMath::Cos(th1), -TMath::Sin(phi)*TMath::Cos(th1), TMath::Sin(th1), zn);
    if (zn > 0) ivis[0] = 1;
    view->FindNormal(TMath::Cos(phi)*TMath::Cos(th2), TMath::Sin(phi)*TMath::Cos(th2), -TMath::Sin(th2), zn);
    if (zn > 0) ivis[2] = 1;
    th = (th1 + th2) / (float)2.;
    if (ipsdr == 1) th = kRad*90;
    view->FindNormal(TMath::Cos(phi)*TMath::Sin(th), TMath::Sin(phi)*TMath::Sin(th), TMath::Cos(th), zn);
    if (zn < 0) ivis[4] = 1;
    if (zn > 0) ivis[5] = 1;

//*-*-      D R A W   S T A C K
    icodes[0] = ia;
    icodes[1] = ib;
    for (i = 1; i <= 4; ++i) {
	j = i;
	if (iordr != 0 && i == 2) j = 4;
	if (iordr != 0 && i == 4) j = 2;
	costh[j - 1]  = TMath::Cos(kRad*ab[jth + 2*i - 3]);
	sinth[j - 1]  = TMath::Sin(kRad*ab[jth + 2*i - 3]);
	cosphi[j - 1] = TMath::Cos(kRad*ab[jphi + 2*i - 3]);
	sinphi[j - 1] = TMath::Sin(kRad*ab[jphi + 2*i - 3]);
    }
    for (iv = 1; iv < nv; ++iv) {
	if (ipsdr == 1) {
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = v[iv - 1]*cosphi[i - 1];
		xyz[i*3 - 2] = v[iv - 1]*sinphi[i - 1];
		xyz[i*3 - 1] = v[iv - 1]*costh[i - 1] / sinth[i - 1];
		xyz[(i + 4)*3 - 3] = v[iv]*cosphi[i - 1];
		xyz[(i + 4)*3 - 2] = v[iv]*sinphi[i - 1];
		xyz[(i + 4)*3 - 1] = v[iv]*costh[i - 1] / sinth[i - 1];
	    }
	} else {
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = v[iv - 1]*sinth[i - 1]*cosphi[i - 1];
		xyz[i*3 - 2] = v[iv - 1]*sinth[i - 1]*sinphi[i - 1];
		xyz[i*3 - 1] = v[iv - 1]*costh[i - 1];
		xyz[(i + 4)*3 - 3] = v[iv]*sinth[i - 1]*cosphi[i - 1];
		xyz[(i + 4)*3 - 2] = v[iv]*sinth[i - 1]*sinphi[i - 1];
		xyz[(i + 4)*3 - 1] = v[iv]*costh[i - 1];
	    }
	}
	if (v[iv - 1] >= v[iv]) continue;
	icodes[2] = iv;
	for (i = 1; i <= 4; ++i) {
	    if (ivis[i - 1] == 0) continue;
	    k1 = i - 1;
	    if (i == 1) k1 = 4;
	    k2 = i;
	    iface[0] = k1;
	    iface[1] = k2;
	    iface[2] = k2 + 4;
	    iface[3] = k1 + 4;
	    tface[0] = tt[k1 + (iv << 2) - 5];
	    tface[1] = tt[k2 + (iv << 2) - 5];
	    tface[2] = tt[k2 + (iv + 1 << 2) - 5];
	    tface[3] = tt[k1 + (iv + 1 << 2) - 5];
	    icodes[3] = i;
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tface);
	}
    }
//*-*-      D R A W   B O T T O M   F A C E
    if (ivis[4] != 0 && v[0] > 0) {
	icodes[2] = 1;
	icodes[3] = 5;
	for (i = 1; i <= 4; ++i) {
	    if (ipsdr == 1) {
		xyz[i*3 - 3] = v[0]*cosphi[i - 1];
		xyz[i*3 - 2] = v[0]*sinphi[i - 1];
		xyz[i*3 - 1] = v[0]*costh[i - 1] / sinth[i - 1];
	    } else {
		xyz[i*3 - 3] = v[0]*sinth[i - 1]*cosphi[i - 1];
		xyz[i*3 - 2] = v[0]*sinth[i - 1]*sinphi[i - 1];
		xyz[i*3 - 1] = v[0]*costh[i - 1];
	    }
	    iface[i - 1] = 5 - i;
	    tface[i - 1] = tt[5 - i - 1];
	}
	(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
    }
//*-*-      D R A W   T O P   F A C E
    if (ivis[5] != 0 && v[nv - 1] > 0) {
	icodes[2] = nv - 1;
	icodes[3] = 6;
	for (i = 1; i <= 4; ++i) {
	    iface[i - 1] = i + 4;
//	    tface[i - 1] = tt[i + 4 + 4*nv - 5];
	    tface[i - 1] = tt[i + 4 + 2*nv - 5];
	}
	(this->*fDrawFace)(icodes, xyz, 4, iface, tface);
    }
//*-*-      N E X T   T H E T A
L400:
    ith += incrth;
    if (ith == 0)    ith = kth;
    if (ith > kth)   ith = 1;
    if (ith != ith2) goto L200;
    if (incrth == 0) goto L500;
    if (incrth < 0) {
       incrth = 0;
       goto L200;
    }
    incrth = -1;
    ith = ith1;
    goto L400;
//*-*-      N E X T   P H I
L500:
    iphi += incr;
    if (iphi == 0)     iphi = kphi;
    if (iphi > kphi)   iphi = 1;
    if (iphi != iphi2) goto L100;
    if (incr == 0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L500;
}

//______________________________________________________________________________
void TLego::LightSource(Int_t nl, Double_t yl, Double_t xscr, Double_t yscr, Double_t zscr, Int_t &irep)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Set light source-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ================                               *
//*-*                                                                     *
//*-*    Input: NL   - source number: -1 off all light sources            *
//*-*                                  0 set diffused light               *
//*-*           YL   - intensity of the light source                      *
//*-*           XSCR \                                                    *
//*-*           YSCR  - direction of the light (in respect of the screen) *
//*-*           ZSCR /                                                    *
//*-*                                                                     *
//*-*    Output: IREP   - reply : 0 - O.K.                                *
//*-*                            -1 - error in light sources definition:  *
//*-*                                 negative intensity                  *
//*-*                                 source number greater than max      *
//*-*                                 light source is placed at origin    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Int_t i;
    Double_t s;

    irep = 0;
    if (nl < 0)       goto L100;
    else if (nl == 0) goto L200;
    else              goto L300;

//*-*-          S W I T C H   O F F   L I G H T S
L100:
    fLoff = 1;
    fYdl = 0;
    for (i = 1; i <= 4; ++i) {
	fYls[i - 1] = 0;
    }
    return;
//*-*-          S E T   D I F F U S E D   L I G H T
L200:
    if (yl < 0) {
       Error("LightSource", "negative light intensity");
       irep = -1;
       return;
    }
    fYdl = yl;
    goto L400;
//*-*-          S E T   L I G H T   S O U R C E
L300:
    if (nl > 4 || yl < 0) {
       Error("LightSource", "illegal light source number (nl=%d, yl=%f)", nl, yl);
       irep = -1;
       return;
    }
    s = TMath::Sqrt(xscr*xscr + yscr*yscr + zscr*zscr);
    if (s == 0) {
       Error("LightSource", "light source is placed at origin");
       irep = -1;
       return;
    }
    fYls[nl - 1] = yl;
    fVls[nl*3 - 3] = xscr / s;
    fVls[nl*3 - 2] = yscr / s;
    fVls[nl*3 - 1] = zscr / s;
//*-*-         C H E C K   L I G H T S
L400:
    fLoff = 0;
    if (fYdl != 0) return;
    for (i = 1; i <= 4; ++i) {
	if (fYls[i - 1] != 0) return;
    }
    fLoff = 1;
}

//______________________________________________________________________________
void TLego::Luminosity(Double_t *anorm, Double_t &flum)
{
//*-*-*-*-*-*-*-*-*-*Find surface luminosity at given point *-*-*-*-*-*-*-*
//*-*                ======================================               *
//*-*                                                                     *
//*-*                                         --                          *
//*-*    Lightness model formula: Y = YD*QA + > YLi*(QD*cosNi+QS*cosRi)   *
//*-*                                         --                          *
//*-*                                                                     *
//*-*            B1     = VN(3)*VL(2) - VN(2)*VL(3)                       *
//*-*            B2     = VN(1)*VL(3) - VN(3)*VL(1)                       *
//*-*            B3     = VN(2)*VL(1) - VN(1)*VL(2)                       *
//*-*            B4     = VN(1)*VL(1) + VN(2)*VL(2) + VN(3)*VL(3)         *
//*-*            VR(1)  = VN(3)*B2 - VN(2)*B3 + VN(1)*B4                  *
//*-*            VR(2)  =-VN(3)*B1 + VN(1)*B3 + VN(2)*B4                  *
//*-*            VR(3)  = VN(2)*B1 - VN(1)*B2 + VN(3)*B4                  *
//*-*            S      = SQRT(VR(1)*VR(1)+VR(2)*VR(2)+VR(3)*VR(3))       *
//*-*            VR(1)  = VR(1)/S                                         *
//*-*            VR(2)  = VR(2)/S                                         *
//*-*            VR(3)  = VR(3)/S                                         *
//*-*            COSR   = VR(1)*0. + VR(2)*0. + VR(3)*1.                  *
//*-*                                                                     *
//*-*    References: WCtoNDC                                              *
//*-*                                                                     *
//*-*    Input: ANORM(3) - surface normal at given point                  *
//*-*                                                                     *
//*-*    Output: FLUM - luminosity                                        *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Double_t cosn, cosr;
    Int_t i;
    Double_t s, vl[3], vn[3];


	TView *view = gPad->GetView();   //Get current view
	if(!view) return;                //Check if `view` is valid!


    /* Parameter adjustments */
    --anorm;

    flum = 0;
    if (fLoff != 0) return;

//*-*-          T R A N S F E R   N O R M A L  T O   SCREEN COORDINATES

    view->NormalWCtoNDC(&anorm[1], vn);
    s = TMath::Sqrt(vn[0]*vn[0] + vn[1]*vn[1] + vn[2]*vn[2]);
    if (vn[2] < 0) s = -(Double_t)s;
    vn[0] /= s;
    vn[1] /= s;
    vn[2] /= s;

//*-*-          F I N D   L U M I N O S I T Y

    flum = fYdl*fQA;
    for (i = 1; i <= 4; ++i) {
	if (fYls[i - 1] <= 0) continue;
	vl[0] = fVls[i*3 - 3];
	vl[1] = fVls[i*3 - 2];
	vl[2] = fVls[i*3 - 1];
	cosn = vl[0]*vn[0] + vl[1]*vn[1] + vl[2]*vn[2];
	if (cosn < 0) continue;
	cosr = vn[1]*(vn[2]*vl[1] - vn[1]*vl[2]) - vn[0]*(vn[0]*vl[2]
		- vn[2]*vl[0]) + vn[2]*cosn;
	if (cosr <= 0) cosr = 0;
	flum += fYls[i - 1]*(fQD*cosn + fQS*TMath::Power(cosr, fNqs));
    }
}

//______________________________________________________________________________
void TLego::ModifyScreen(Double_t *r1, Double_t *r2)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*Modify SCREEN*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        =============                                *
//*-*                                                                     *
//*-*    Input: R1(3) - 1-st point of the line                            *
//*-*           R2(3) - 2-nd point of the line                            *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Int_t i, i1, i2;
    Double_t x1, x2, y1, y2, dy, ww, yy1, yy2, *tn;

    /* Parameter adjustments */
    --r2;
    --r1;

    if(gPad->GetView()) {
    	tn = gPad->GetView()->GetTN();
    	
    	x1 = tn[0]*r1[1] + tn[1]*r1[2] + tn[2]*r1[3] + tn[3];
    	x2 = tn[0]*r2[1] + tn[1]*r2[2] + tn[2]*r2[3] + tn[3];
    	y1 = tn[4]*r1[1] + tn[5]*r1[2] + tn[6]*r1[3] + tn[7];
    	y2 = tn[4]*r2[1] + tn[5]*r2[2] + tn[6]*r2[3] + tn[7];
    }
    else {
    	Error("ModifyScreen", "no TView in current pad");
    	return;
    }

    if (x1 >= x2) {
       ww = x1;
       x1 = x2;
       x2 = ww;
       ww = y1;
       y1 = y2;
       y2 = ww;
    }
    i1 = Int_t((x1 - fX0) / fDX) + 15;
    i2 = Int_t((x2 - fX0) / fDX) + 15;
    if (i1 == i2) return;

//*-*-          M O D I F Y   B O U N D A R I E S   OF THE SCREEN

    dy = (y2 - y1) / (i2 - i1);
    for (i = i1; i <= i2 - 1; ++i) {
	yy1 = y1 + dy*(i - i1);
	yy2 = yy1 + dy;
	if (fD[2*i - 2] > yy1) fD[2*i - 2] = yy1;
	if (fD[2*i - 1] > yy2) fD[2*i - 1] = yy2;
	if (fU[2*i - 2] < yy1) fU[2*i - 2] = yy1;
	if (fU[2*i - 1] < yy2) fU[2*i - 1] = yy2;
    }
}


//______________________________________________________________________________
void TLego::SetDrawFace(DrawFaceFunc_t drface)
{
//*-*-*-*-*-*-*-*-*Store pointer to current algorithm to draw faces *-*-*-*
//*-*              ================================================       *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fDrawFace = drface;
}

//______________________________________________________________________________
void TLego::SetLegoFunction(LegoFunc_t fun)
{
//*-*-*-*-*-*-*-*-*Store pointer to current lego function *-*-*-*-*-*-*-*-*
//*-*              ======================================                 *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fLegoFunction = fun;
}

//______________________________________________________________________________
void TLego::SetSurfaceFunction(SurfaceFunc_t fun)
{
//*-*-*-*-*-*-*-*-*Store pointer to current surface function*-*-*-*-*-*-*-*
//*-*              =========================================              *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fSurfaceFunction = fun;
}


//______________________________________________________________________________
void TLego::SetColorDark(Color_t color, Int_t n)
{
//*-*-*-*-*-*-*-*-*Store dark color for stack number n-*-*-**-*-*-*-*-*-*-*
//*-*              ===================================                    *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (n < 0 ) {fColorBottom = color; return;}
   if (n > 9 ) {fColorTop    = color; return;}
   fColorDark[n] = color;
}


//______________________________________________________________________________
void TLego::SetColorMain(Color_t color, Int_t n)
{
//*-*-*-*-*-*-*-*-*Store color for stack number n*-*-*-*-*-**-*-*-*-*-*-*-*
//*-*              ==============================                         *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (n < 0 ) {fColorBottom = color; return;}
   if (n > 9 ) {fColorTop    = color; return;}
   fColorMain[n] = color;
}


//______________________________________________________________________________
void TLego::SideVisibilityDecode(Double_t val, Int_t &iv1, Int_t &iv2, Int_t &iv3, Int_t &iv4, Int_t &iv5, Int_t &iv6, Int_t &ir)
{
//*-*-*-*-*-*-*Decode side visibilities and order along R for sector*-*-*-*
//*-*          =====================================================      *
//*-*                                                                     *
//*-*    Input: VAL - encoded value                                       *
//*-*                                                                     *
//*-*    Output: IV1 ... IV6  - visibility of the sides                   *
//*-*            IR           - increment along R                         *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t ivis[6], i, k, num;

    k = Int_t(val);
    num = 128;
    for (i = 1; i <= 6; ++i) {
	ivis[i - 1] = 0;
	num /= 2;
	if (k < num) continue;
	k -= num;
	ivis[i - 1] = 1;
    }
    ir = 1;
    if (k == 1) ir = -1;
    iv1 = ivis[5];
    iv2 = ivis[4];
    iv3 = ivis[3];
    iv4 = ivis[2];
    iv5 = ivis[1];
    iv6 = ivis[0];
}


//______________________________________________________________________________
void TLego::SideVisibilityEncode(Int_t iopt, Double_t phi1, Double_t phi2, Double_t &val)
{
//*-*-*-*-*-*-*Encode side visibilities and order along R for sector*-*-*-*
//*-*          =====================================================      *
//*-*                                                                     *
//*-*    Input: IOPT - options: 1 - from BACK to FRONT 'BF'               *
//*-*                           2 - from FRONT to BACK 'FB'               *
//*-*           PHI1 - 1st phi of sector                                  *
//*-*           PHI2 - 2nd phi of sector                                  *
//*-*                                                                     *
//*-*    Output: VAL - encoded value                                      *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Local variables */
    Double_t zn, phi;
    Int_t k = 0;
    TView *view = 0;

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
    		Error("SideVisibilityEncode", "no TView in current pad");
    		return;
    	}
    }

    view->FindNormal(0, 0, 1, zn);
    if (zn > 0) k += 64;
    if (zn < 0) k += 32;
    view->FindNormal(-TMath::Sin(phi2), TMath::Cos(phi2), 0, zn);
    if (zn > 0) k += 16;
    view->FindNormal(TMath::Sin(phi1), -TMath::Cos(phi1), 0, zn);
    if (zn > 0) k += 4;
    phi = (phi1 + phi2) / (float)2.;
    view->FindNormal(TMath::Cos(phi), TMath::Sin(phi), 0, zn);
    if (zn > 0) k += 8;
    if (zn < 0) k += 2;
    if (zn <= 0 && iopt == 1 || zn > 0 && iopt == 2) ++k;
    val = Double_t(k);
}


//______________________________________________________________________________
void TLego::Spectrum(Int_t nl, Double_t fmin, Double_t fmax, Int_t ic, Int_t idc, Int_t &irep)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Set Spectrum-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============                                  *
//*-*                                                                     *
//*-*    Input: NL   - number of levels                                   *
//*-*           FMIN - MIN fuction value                                  *
//*-*           FMAX - MAX fuction value                                  *
//*-*           IC   - initial color index (for 1st level)                *
//*-*           IDC  - color index increment                              *
//*-*                                                                     *
//*-*    Output: IREP - reply: 0 O.K.                                     *
//*-*                         -1 error in parameters                      *
//*-*                            F_max less than F_min                    *
//*-*                            illegal number of levels                 *
//*-*                            initial color index is negative          *
//*-*                            color index increment must be positive   *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    static const char *where = "Spectrum";

    /* Local variables */
    Double_t delf;
    Int_t i;

    irep = 0;
    if (nl == 0) {fNlevel = 0; return; }

//*-*-          C H E C K   P A R A M E T E R S

    if (fmax <= fmin) {
       Error(where, "fmax (%f) less than fmin (%f)", fmax, fmin);
       irep = -1;
       return;
    }
    if (nl < 0 || nl > 256) {
       Error(where, "illegal number of levels (%d)", nl);
       irep = -1;
       return;
    }
    if (ic < 0) {
       Error(where, "initial color index is negative");
       irep = -1;
       return;
    }
    if (idc < 0) {
       Error(where, "color index increment must be positive");
       irep = -1;
    }

//*-*-          S E T  S P E C T R

    const Int_t MAXCOL = 50;
    delf    = (fmax - fmin) / nl;
    fNlevel = -(nl + 1);
    for (i = 1; i <= nl+1; ++i) {
	fFunLevel[i - 1] = fmin + (i - 1)*delf;
	fColorLevel[i] = ic + (i - 1)*idc;
	if (ic <= MAXCOL && fColorLevel[i] > MAXCOL) fColorLevel[i] -= MAXCOL;
    }
    fColorLevel[0] = fColorLevel[1];
    fColorLevel[nl + 1] = fColorLevel[nl];
}

//______________________________________________________________________________
void TLego::SurfaceCartesian(Double_t ang, Int_t nx, Int_t ny, const char *chopt)
{
//*-*-*-*-*-*-*-*-*Draw surface in cartesian coordinate system*-*-*-*-*-*-*
//*-*              ===========================================            *
//*-*                                                                     *
//*-*    Input: ANG      - angle between X ang Y                          *
//*-*           NX       - number of steps along X                        *
//*-*           NY       - number of steps along Y                        *
//*-*                                                                     *
//*-*           FUN(IX,IY,F,T) - external routine                         *
//*-*             IX     - X number of the cell                           *
//*-*             IY     - Y number of the cell                           *
//*-*             F(3,4) - face which corresponds to the cell             *
//*-*             T(4)   - additional function (for example: temperature) *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IX                                        *
//*-*               ICODES(2) - IY                                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT - options: 'BF' - from BACK to FRONT                *
//*-*                            'FB' - from FRONT to BACK                *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */

    Int_t iface[4] = { 1,2,3,4 };

    /* Local variables */
    Double_t cosa, sina, f[12]	/* was [3][4] */;
    Int_t i, incrx, incry, i1, ix, iy;
    Double_t tt[4];
    Int_t icodes[2], ix1, iy1, ix2, iy2;
    Double_t xyz[12]	/* was [3][4] */;
    Double_t *tn;

    sina = TMath::Sin(ang*kRad);
    cosa = TMath::Cos(ang*kRad);

//*-*-          F I N D   T H E   M O S T   L E F T   P O I N T

	if(gPad->GetView())
		tn = gPad->GetView()->GetTN();
	else {
		Error("SurfaceCartesian", "no TView in current pad");
		return;
	}

    i1 = 1;
    if (tn[0] < 0) i1 = 2;
    if (tn[0]*cosa + tn[1]*sina < 0) i1 = 5 - i1;

//*-*-          D E F I N E   O R D E R   O F   D R A W I N G

    if (*chopt == 'B' || *chopt == 'b') {incrx = -1; incry = -1;}
    else                                {incrx = 1;  incry = 1;}
    if (i1 == 1 || i1 == 2) incrx = -incrx;
    if (i1 == 2 || i1 == 3) incry = -incry;
    ix1 = 1;
    iy1 = 1;
    if (incrx < 0) ix1 = nx;
    if (incry < 0) iy1 = ny;
    ix2 = nx - ix1 + 1;
    iy2 = ny - iy1 + 1;

//*-*-          D R A W   S U R F A C E

    THistPainter *painter = (THistPainter*)gCurrentHist->GetPainter();
    for (iy = iy1; incry < 0 ? iy >= iy2 : iy <= iy2; iy += incry) {
	for (ix = ix1; incrx < 0 ? ix >= ix2 : ix <= ix2; ix += incrx) {
	    if (!painter->IsInside(ix,iy)) continue;
	    (this->*fSurfaceFunction)(ix, iy, f, tt);
	    for (i = 1; i <= 4; ++i) {
		xyz[i*3 - 3] = f[i*3 - 3] + f[i*3 - 2]*cosa;
		xyz[i*3 - 2] = f[i*3 - 2]*sina;
		xyz[i*3 - 1] = f[i*3 - 1];
	    }
	    icodes[0] = ix;
	    icodes[1] = iy;
	    (this->*fDrawFace)(icodes, xyz, 4, iface, tt);
	}
    }
}

//______________________________________________________________________________
void TLego::SurfaceFunction(Int_t ia, Int_t ib, Double_t *f, Double_t *t)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Service function for Surfaces*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    /* Initialized data */

    static Int_t ixadd[4] = { 0,1,1,0 };
    static Int_t iyadd[4] = { 0,0,1,1 };

    const Double_t kHMAX = 1.05;
    Double_t rinrad = gStyle->GetLegoInnerR();
    Double_t dangle = 10; //Delta angle for Rapidity option
    Double_t xval1l, xval2l, yval1l, yval2l;
    Int_t i, ixa, iya, icx, ixt, iyt;

    /* Parameter adjustments */
    --t;
    f -= 4;

    ixt = ia + Hparam.xfirst - 1;
    iyt = ib + Hparam.yfirst - 1;

	xval1l = Hparam.xmin;
	xval2l = Hparam.xmax;
	yval1l = Hparam.ymin;
	yval2l = Hparam.ymax;

    for (i = 1; i <= 4; ++i) {
	ixa = ixadd[i - 1];
	iya = iyadd[i - 1];
        Double_t xwid = gCurrentHist->GetXaxis()->GetBinWidth(ixt+ixa);
        Double_t ywid = gCurrentHist->GetYaxis()->GetBinWidth(iyt+iya);

//*-*-          Compute the cell position in cartesian coordinates
//*-*-          and compute the LOG if necessary

	f[i*3 + 1] = gCurrentHist->GetXaxis()->GetBinLowEdge(ixt+ixa) + 0.5*xwid;
	f[i*3 + 2] = gCurrentHist->GetYaxis()->GetBinLowEdge(iyt+iya) + 0.5*ywid;
        if (Hoption.Logx) f[i*3 + 1] = TMath::Log10(f[i*3 + 1]);
        if (Hoption.Logy) f[i*3 + 2] = TMath::Log10(f[i*3 + 2]);

//*-*-     Transform the cell position in the required coordinate system

        if (Hoption.System == kPOLAR) {
	    f[i*3 + 1] = 360*(f[i*3 + 1] - xval1l) / (xval2l - xval1l);
	    f[i*3 + 2] = (f[i*3 + 2] - yval1l) / (yval2l - yval1l);
        } else if (Hoption.System == kCYLINDRICAL) {
	    f[i*3 + 1] = 360*(f[i*3 + 1] - xval1l) / (xval2l - xval1l);
        } else if (Hoption.System == kSPHERICAL) {
	    f[i*3 + 1] = 360*(f[i*3 + 1] - xval1l) / (xval2l - xval1l);
	    f[i*3 + 2] = 360*(f[i*3 + 2] - yval1l) / (yval2l - yval1l);
        } else if (Hoption.System == kRAPIDITY) {
	    f[i*3 + 1] = 360*(f[i*3 + 1] - xval1l) / (xval2l - xval1l);
	    f[i*3 + 2] = (180 - dangle*2)*(f[i*3 + 2] - yval1l) / (yval2l - yval1l) + dangle;
	}

//*-*-          Get the content of the table. If the X index (ICX) is
//*-*-          greater than the X size of the table (NCX), that's mean
//*-*-          IGTABL tried to close the surface and in this case the
//*-*-          first channel should be used. */

	icx = ixt + ixa;
	if (icx > Hparam.xlast) icx = 1;
        f[i*3+3] = gCurrentHist->GetCellContent(icx, iyt + iya);
        if (Hoption.Logz) {
           if (f[i*3+3] > 0) f[i*3+3] = TMath::Log10(f[i*3+3]);
           else              f[i*3+3] = Hparam.zmin;
           if (f[i*3+3] < Hparam.zmin) f[i*3+3] = Hparam.zmin;
           if (f[i*3+3] > Hparam.zmax) f[i*3+3] = Hparam.zmax;
        } else {
           f[i*3+3] = TMath::Max(Hparam.zmin, f[i*3+3]);
           f[i*3+3] = TMath::Min(Hparam.zmax, f[i*3+3]);
        }

//*-*-          The colors on the surface can represent the content or the errors.

//	if (fSumw2.fN) t[i] = gCurrentHist->GetCellError(icx, iyt + iya);
//	else           t[i] = f[i * 3 + 3];
        t[i] = f[i * 3 + 3];
    }

//*-*-          LOGZ is required...

    if (Hoption.Surf == 23) {
	for (i = 1; i <= 4; ++i) {
	    if (Hoption.Logz && Hparam.zmax > 0) {
		f[i * 3 + 3] = TMath::Log10(kHMAX*Hparam.zmax);
	    } else {
		f[i * 3 + 3] = kHMAX*Hparam.zmax;
	    }
	}
    }

    if (Hoption.System == kCYLINDRICAL || Hoption.System == kSPHERICAL || Hoption.System == kRAPIDITY) {
	for (i = 1; i <= 4; ++i) {
	    f[i*3 + 3] = (1 - rinrad)*((f[i*3 + 3] - Hparam.zmin) /
		    (Hparam.zmax - Hparam.zmin)) + rinrad;
	}
    }
}

//______________________________________________________________________________
void TLego::SurfacePolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw surface in polar coordinates*-*-*-*-*-*-*-*-*
//*-*                    =================================                *
//*-*                                                                     *
//*-*    Input: IORDR - order of variables (0 - R,PHI, 1 - PHI,R)         *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NB    - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,F,T) - external routine                         *
//*-*             IA     - cell number for 1st variable                   *
//*-*             IB     - cell number for 2nd variable                   *
//*-*             F(3,4) - face which corresponds to the cell             *
//*-*               F(1,*) - A                                            *
//*-*               F(2,*) - B                                            *
//*-*               F(3,*) - Z                                            *
//*-*             T(4)   - additional function (for example: temperature) *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT       - options: 'BF' - from BACK to FRONT          *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */

    static Int_t iface[4] = { 1,2,3,4 };
    TView *view = 0;

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
    		Error("SurfacePolar", "no TView in current pad");
    		return;
    	}
    }


    Int_t iphi, jphi, kphi, incr, nphi, iopt, iphi1, iphi2;
    Double_t f[12]	/* was [3][4] */;
    Int_t i, j, incrr, ir1, ir2;
    Double_t z;
    Int_t ia, ib, ir, jr, nr, icodes[2];
    Double_t tt[4];
    Double_t phi, ttt[4], xyz[12]	/* was [3][4] */;
    ia = ib = 0;	

    if (iordr == 0) {
	jr   = 1;
	jphi = 2;
	nr   = na;
	nphi = nb;
    } else {
	jr   = 2;
	jphi = 1;
	nr   = nb;
	nphi = na;
    }
    if (nphi > 180) {
       Error("SurfacePolar", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-       P R E P A R E   P H I   A R R A Y
//*-*-      F I N D    C R I T I C A L   S E C T O R S

    kphi = nphi;
    if (iordr == 0) ia = nr;
    if (iordr != 0) ib = nr;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fSurfaceFunction)(ia, ib, f, tt);
	if (i == 1) fAphi[0] = f[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + f[jphi - 1]) / (float)2.;
	fAphi[i] = f[jphi + 5];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-       D R A W   S U R F A C E

    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L300;

//*-*-      F I N D   O R D E R   A L O N G   R
    if (iordr == 0) {ia = nr; 	ib = iphi;}
    else            {ia = iphi;ib = nr;}

    (this->*fSurfaceFunction)(ia, ib, f, tt);
    phi = kRad*((f[jphi - 1] + f[jphi + 5]) / 2);
    view->FindNormal(TMath::Cos(phi), TMath::Sin(phi), 0, z);
    incrr = 1;
    ir1 = 1;
    if (z <= 0 && iopt == 1 || z > 0 && iopt == 2) {
	incrr = -1;
	ir1 = nr;
    }
    ir2 = nr - ir1 + 1;
//*-*-      D R A W   S U R F A C E   F O R   S E C T O R
    for (ir = ir1; incrr < 0 ? ir >= ir2 : ir <= ir2; ir += incrr) {
	if (iordr == 0) ia = ir;
	if (iordr != 0) ib = ir;

	(this->*fSurfaceFunction)(ia, ib, f, tt);
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr != 0 && i == 2) j = 4;
	    if (iordr != 0 && i == 4) j = 2;
	    xyz[j*3 - 3] = f[jr + i*3 - 4]*TMath::Cos(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 2] = f[jr + i*3 - 4]*TMath::Sin(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 1] = f[i*3 - 1];
	    ttt[j - 1] = tt[i - 1];
	}
	icodes[0] = ia;
	icodes[1] = ib;
	(this->*fDrawFace)(icodes, xyz, 4, iface, ttt);
    }
//*-*-      N E X T   P H I
L300:
    iphi += incr;
    if (iphi == 0)     iphi = kphi;
    if (iphi > kphi)   iphi = 1;
    if (iphi != iphi2) goto L100;
    if (incr == 0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L300;
}

//______________________________________________________________________________
void TLego::SurfaceCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*-*-*-*Draw surface in cylindrical coordinates*-*-*-*-*-*-*-*-*
//*-*              =======================================                *
//*-*                                                                     *
//*-*    Input: IORDR - order of variables (0 - Z,PHI, 1 - PHI,Z)         *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NB    - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,F,T) - external routine                         *
//*-*             IA     - cell number for 1st variable                   *
//*-*             IB     - cell number for 2nd variable                   *
//*-*             F(3,4) - face which corresponds to the cell             *
//*-*               F(1,*) - A                                            *
//*-*               F(2,*) - B                                            *
//*-*               F(3,*) - R                                            *
//*-*             T(4)   - additional function (for example: temperature) *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT       - options: 'BF' - from BACK to FRONT          *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//Begin_Html
/*
<img src="gif/Surface1Cylindrical.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */

    static Int_t iface[4] = { 1,2,3,4 };

    Int_t iphi, jphi, kphi, incr, nphi, iopt, iphi1, iphi2;
    Int_t i, j, incrz, nz, iz1, iz2;
    Int_t ia, ib, iz, jz, icodes[2];
    Double_t f[12]	/* was [3][4] */;
    Double_t z;
    Double_t tt[4];
    Double_t ttt[4], xyz[12]	/* was [3][4] */;
    TView *view = 0;
    ia = ib = 0;	

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
    		Error("SurfaceCylindrical", "no TView in current pad");
    		return;
    	}
    }


    if (iordr == 0) {
	jz   = 1;
	jphi = 2;
	nz   = na;
	nphi = nb;
    } else {
	jz   = 2;
	jphi = 1;
	nz   = nb;
	nphi = na;
    }
    if (nphi > 180) {
       Error("SurfaceCylindrical", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-       P R E P A R E   P H I   A R R A Y
//*-*-       F I N D    C R I T I C A L   S E C T O R S

    kphi = nphi;
    if (iordr == 0) ia = nz;
    if (iordr != 0) ib = nz;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fSurfaceFunction)(ia, ib, f, tt);
	if (i == 1) fAphi[0] = f[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + f[jphi - 1]) / (float)2.;
	fAphi[i] = f[jphi + 5];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-       F I N D   O R D E R   A L O N G   Z

    incrz = 1;
    iz1 = 1;
    view->FindNormal(0, 0, 1, z);
    if (z <= 0 && iopt == 1 || z > 0 && iopt == 2) {
	incrz = -1;
	iz1 = nz;
    }
    iz2 = nz - iz1 + 1;

//*-*-       D R A W   S U R F A C E

    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L400;
    for (iz = iz1; incrz < 0 ? iz >= iz2 : iz <= iz2; iz += incrz) {
	if (iordr == 0) {ia = iz;   ib = iphi;}
	else             {ia = iphi; ib = iz;}
	(this->*fSurfaceFunction)(ia, ib, f, tt);
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr == 0 && i == 2) j = 4;
	    if (iordr == 0 && i == 4) j = 2;
	    xyz[j*3 - 3] = f[i*3 - 1]*TMath::Cos(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 2] = f[i*3 - 1]*TMath::Sin(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 1] = f[jz + i*3 - 4];
	    ttt[j - 1] = tt[i - 1];
	}
	icodes[0] = ia;
	icodes[1] = ib;
	(this->*fDrawFace)(icodes, xyz, 4, iface, ttt);
    }
//*-*-      N E X T   P H I
L400:
    iphi += incr;
    if (iphi == 0)     iphi = kphi;
    if (iphi > kphi)   iphi = 1;
    if (iphi != iphi2) goto L100;
    if (incr ==  0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L400;
}

//______________________________________________________________________________
void TLego::SurfaceSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt)
{
//*-*-*-*-*-*-*-*-*-*-*Draw surface in spheric coordinates*-*-*-*-*-*-*-*-*
//*-*                  ===================================                *
//*-*                                                                     *
//*-*    Input: IPSDR - pseudo-rapidity flag                              *
//*-*           IORDR - order of variables (0 - THETA,PHI; 1 - PHI,THETA) *
//*-*           NA    - number of steps along 1st variable                *
//*-*           NB    - number of steps along 2nd variable                *
//*-*                                                                     *
//*-*           FUN(IA,IB,F,T) - external routine                         *
//*-*             IA     - cell number for 1st variable                   *
//*-*             IB     - cell number for 2nd variable                   *
//*-*             F(3,4) - face which corresponds to the cell             *
//*-*               F(1,*) - A                                            *
//*-*               F(2,*) - B                                            *
//*-*               F(3,*) - R                                            *
//*-*             T(4)   - additional function (for example: temperature) *
//*-*                                                                     *
//*-*           DRFACE(ICODES,XYZ,NP,IFACE,T) - routine for face drawing  *
//*-*             ICODES(*) - set of codes for this face                  *
//*-*               ICODES(1) - IA                                        *
//*-*               ICODES(2) - IB                                        *
//*-*             XYZ(3,*)  - coordinates of nodes                        *
//*-*             NP        - number of nodes in face                     *
//*-*             IFACE(NP) - face                                        *
//*-*             T(NP)     - additional function                         *
//*-*                                                                     *
//*-*           CHOPT       - options: 'BF' - from BACK to FRONT          *
//*-*                                  'FB' - from FRONT to BACK          *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    /* Initialized data */
    static Int_t iface[4] = { 1,2,3,4 };

    Int_t iphi, jphi, kphi, incr, nphi, iopt, iphi1, iphi2;
    Int_t i, j, incrth, ith, jth, kth, nth, mth, ith1, ith2;
    Int_t ia, ib, icodes[2];
    Double_t f[12]	/* was [3][4] */;
    Double_t tt[4];
    Double_t phi;
    Double_t ttt[4], xyz[12]	/* was [3][4] */;
    TView *view = 0;
    ia = ib = 0;	

    if(gPad) {
    	view = gPad->GetView();
    	if(!view) {
    		Error("SurfaceSpherical", "no TView in current pad");
    		return;
    	}
    }


    if (iordr == 0) {
	jth  = 1;
	jphi = 2;
	nth  = na;
	nphi = nb;
    } else {
	jth  = 2;
	jphi = 1;
	nth  = nb;
	nphi = na;
    }
    if (nth > 180)  {
       Error("SurfaceSpherical", "too many THETA sectors (%d)", nth);
       return;
    }
    if (nphi > 180) {
       Error("SurfaceSpherical", "too many PHI sectors (%d)", nphi);
       return;
    }
    iopt = 2;
    if (*chopt == 'B' || *chopt == 'b') iopt = 1;

//*-*-       P R E P A R E   P H I   A R R A Y
//*-*-       F I N D    C R I T I C A L   P H I   S E C T O R S

    kphi = nphi;
    mth = nth / 2;
    if (mth == 0)    mth = 1;
    if (iordr == 0) ia = mth;
    if (iordr != 0) ib = mth;
    for (i = 1; i <= nphi; ++i) {
	if (iordr == 0) ib = i;
	if (iordr != 0) ia = i;
	(this->*fSurfaceFunction)(ia, ib, f, tt);
	if (i == 1) fAphi[0] = f[jphi - 1];
	fAphi[i - 1] = (fAphi[i - 1] + f[jphi - 1]) / (float)2.;
	fAphi[i] = f[jphi + 5];
    }
    view->FindPhiSectors(iopt, kphi, fAphi, iphi1, iphi2);

//*-*-       P R E P A R E   T H E T A   A R R A Y

    if (iordr == 0) ib = 1;
    if (iordr != 0) ia = 1;
    for (i = 1; i <= nth; ++i) {
	if (iordr == 0) ia = i;
	if (iordr != 0) ib = i;

	(this->*fSurfaceFunction)(ia, ib, f, tt);
	if (i == 1) fAphi[0] = f[jth - 1];
	fAphi[i - 1] = (fAphi[i - 1] + f[jth - 1]) / (float)2.;
	fAphi[i] = f[jth + 5];
    }

//*-*-       D R A W   S U R F A C E

    kth  = nth;
    incr = 1;
    iphi = iphi1;
L100:
    if (iphi > nphi) goto L500;

//*-*-      F I N D    C R I T I C A L   T H E T A   S E C T O R S
    if (iordr == 0) {ia = mth; ib = iphi;}
    else             {ia = iphi;ib = mth;}

    (this->*fSurfaceFunction)(ia, ib, f, tt);
    phi = (f[jphi - 1] + f[jphi + 5]) / (float)2.;
    view->FindThetaSectors(iopt, phi, kth, fAphi, ith1, ith2);
    incrth = 1;
    ith = ith1;
L200:
    if (ith > nth)   goto L400;
    if (iordr == 0) ia = ith;
    if (iordr != 0) ib = ith;

    (this->*fSurfaceFunction)(ia, ib, f, tt);
    if (ipsdr == 1) {
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr != 0 && i == 2) j = 4;
	    if (iordr != 0 && i == 4) j = 2;
	    xyz[j * 3 - 3] = f[i*3 - 1]*TMath::Cos(f[jphi + i*3 - 4]*kRad);
	    xyz[j * 3 - 2] = f[i*3 - 1]*TMath::Sin(f[jphi + i*3 - 4]*kRad);
	    xyz[j * 3 - 1] = f[i*3 - 1]*TMath::Cos(f[jth  + i*3 - 4]*kRad) /
		    TMath::Sin(f[jth + i*3 - 4]*kRad);
	    ttt[j - 1] = tt[i - 1];
	}
    } else {
	for (i = 1; i <= 4; ++i) {
	    j = i;
	    if (iordr != 0 && i == 2) j = 4;
	    if (iordr != 0 && i == 4) j = 2;
	    xyz[j*3 - 3] = f[i*3 - 1]*TMath::Sin(f[jth + i*3 - 4]*kRad)*TMath::Cos(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 2] = f[i*3 - 1]*TMath::Sin(f[jth + i*3 - 4]*kRad)*TMath::Sin(f[jphi + i*3 - 4]*kRad);
	    xyz[j*3 - 1] = f[i*3 - 1]*TMath::Cos(f[jth + i*3 - 4]*kRad);
	    ttt[j - 1] = tt[i - 1];
	}
    }
    icodes[0] = ia;
    icodes[1] = ib;
    (this->*fDrawFace)(icodes, xyz, 4, iface, ttt);
//*-*-      N E X T   T H E T A
L400:
    ith += incrth;
    if (ith == 0)    ith = kth;
    if (ith > kth)   ith = 1;
    if (ith != ith2) goto L200;
    if (incrth == 0)  goto L500;
    if (incrth < 0) {
       incrth = 0;
       goto L200;
    }
    incrth = -1;
    ith = ith1;
    goto L400;
//*-*-      N E X T   P H I
L500:
    iphi += incr;
    if (iphi == 0)     iphi = kphi;
    if (iphi > kphi)   iphi = 1;
    if (iphi != iphi2) goto L100;
    if (incr == 0) return;
    if (incr < 0) {
       incr = 0;
       goto L100;
    }
    incr = -1;
    iphi = iphi1;
    goto L500;
}

//______________________________________________________________________________
void TLego::SurfaceProperty(Double_t qqa, Double_t qqd, Double_t qqs, Int_t nnqs, Int_t &irep)
{
//*-*-*-*-*-*-*-*-*-*-*Set surface property coefficients*-*-*-*-*-*-*-*-*-*
//*-*                  =================================                  *
//*-*                                                                     *
//*-*    Input: QQA  - diffusion coefficient for diffused light  [0.,1.]  *
//*-*           QQD  - diffusion coefficient for direct light    [0.,1.]  *
//*-*           QQS  - diffusion coefficient for reflected light [0.,1.]  *
//*-*           NNCS - power coefficient for reflected light     (.GE.1)  *
//*-*                                                                     *
//*-*                                         --                          *
//*-*    Lightness model formula: Y = YD*QA + > YLi*(QD*cosNi+QS*cosRi)   *
//*-*                                         --                          *
//*-*                                                                     *
//*-*    Output: IREP   - reply : 0 - O.K.                                *
//*-*                            -1 - error in cooefficients              *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    irep = 0;
    if (qqa < 0 || qqa > 1 || qqd < 0 || qqd > 1 || qqs < 0 || qqs > 1 || nnqs < 1) {
       Error("SurfaceProperty", "error in coefficients");
       irep = -1;
       return;
    }
    fQA  = qqa;
    fQD  = qqd;
    fQS  = qqs;
    fNqs = nnqs;
}

