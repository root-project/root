// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   19/01/96

#include "TGWin32Object.h"

#ifndef ROOT_TGWin32Pen
#include "TGWin32Pen.h"
#endif

// ClassImp(TGWin32Pen)

//______________________________________________________________________________
TGWin32Pen::TGWin32Pen(){

    fPen.lopnStyle   = BS_SOLID;
    fPen.lopnWidth.x = 0;
    fPen.lopnColor   = 0;

    fBrush.lbStyle = BS_SOLID;
    fBrush.lbColor = 0;
    fBrush.lbHatch = 0;

    flUserDash     = 0;
    fUserDash      = NULL;

    CreatePen();
}

//______________________________________________________________________________
TGWin32Pen::~TGWin32Pen(){
  Delete();
}

//______________________________________________________________________________
void TGWin32Pen::Delete(){
  if (fhdPen) DeleteObject(fhdPen);
}
//______________________________________________________________________________
HPEN TGWin32Pen::CreatePen(){


//*-*  Set width of the line

  DWORD dwWidth = fPen.lopnWidth.x;
  if (dwWidth <= 1) dwWidth = 1;

//*-* Set style of the line

  DWORD dwStyle = fPen.lopnStyle |
                   ((dwWidth == 1) ? PS_COSMETIC
                                   : (PS_GEOMETRIC | PS_INSIDEFRAME) );

 Delete();
// return fhdPen = ExtCreatePen(dwStyle, dwWidth, &fBrush, flUserDash, (CONST DWORD *)fUserDash);
  if (fPen.lopnStyle & PS_USERSTYLE)
     fhdPen = ExtCreatePen(dwStyle, dwWidth, &fBrush, flUserDash, (CONST DWORD *)fUserDash);
  else
     fhdPen = CreatePenIndirect(&fPen);
  return fhdPen;
}

//______________________________________________________________________________
void TGWin32Pen::SetWidth(Width_t width){
    if (fPen.lopnWidth.x != width) {
        fPen.lopnWidth.x = width;
        CreatePen();
    }
}

//______________________________________________________________________________
void TGWin32Pen::SetType(int n, int *dash){
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
//*-*    SUBROUTINE IXSETLS(N,DASH)
//*-*    INTEGER N       : length of dash list
//*-*    INTEGER DASH(N) : dash segment lengths
//*-*
//*-*    Set line style:
//*-*    if N.LE.0 use pre-defined Windows style:
//*-*         0 - solid lines
//*-*        -1 - solid lines
//*-*        -2 - dash line
//*-*        -3 - dot  line
//*-*        -4 - dash-dot line
//*-*        -5 - dash-dot-dot line
//*-*    .LE.-6 - solid line
//*-*
//*-*    if N.GT.0 use dashed lines described by DASH(N)
//*-*    e.g. N=4,DASH=(6,3,1,3) gives a dashed-dotted line with dash length 6
//*-*    and a gap of 7 between dashes
//*-*

  UINT style;
  if( n <= 0 ) {
    flUserDash = 0;
    fUserDash = NULL;
    switch (n)
     {
       case  0:  style = PS_SOLID;
                 break;
       case -1:  style = PS_SOLID;
                 break;
       case -2:  style = PS_DASH;
                 break;
       case -3:  style = PS_DOT;
                 break;
       case -4:  style = PS_DASHDOT;
                 break;
       case -5:  style = PS_DASHDOTDOT;
                 break;
       default:  style = PS_SOLID;
                break;
     }
   if (fPen.lopnStyle == style ) return;
   fPen.lopnStyle = style;
  }
  else {
    fPen.lopnStyle = PS_USERSTYLE;   // This style is defined for Windows NT only
    flUserDash = n;                  // not for Windows 95
    fUserDash  = dash;
  }
  CreatePen();
}

//______________________________________________________________________________
void TGWin32Pen::SetColor(COLORREF color){
  if (fPen.lopnColor != color) {
    fPen.lopnColor = color;
//    fBrush.lbColor = ROOTColorIndex(cindex);
    fBrush.lbColor = color;
    CreatePen();
 }
}

//______________________________________________________________________________
HGDIOBJ TGWin32Pen::GetWin32Pen() {
    return fhdPen;
}

