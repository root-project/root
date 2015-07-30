// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id$
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
**
*****************************************************************************/


/////////////////////////////////////////////////////////////////////////////////
//
// TQtPen class is Qt QPen with ROOT TAttLine interface
//
/////////////////////////////////////////////////////////////////////////////////

#include "TQtPen.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TMath.h"
#include "TStyle.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TAttLine.h"

#include <QtGui/QFontMetrics>
#include <QDebug>

////////////////////////////////////////////////////////////////////////////////
/// TQtPen default ctor

TQtPen::TQtPen(): QPen(),TAttLine()
{
}
////////////////////////////////////////////////////////////////////////////////
/// Copy ctor to copy ROOT TAttLine object

TQtPen::TQtPen(const TAttLine &line) : QPen()
{
   SetLineAttributes(line);
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the  Qt pen attributes from ROOT TAttLine object

TQtPen &TQtPen::operator=(const TAttLine &line)
{
   SetLineAttributes(line);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Maps the ROOT TAttLine attributes to QPen attributes

void  TQtPen::SetLineAttributes(const TAttLine &lineAttributes)
{
   SetLineColor(lineAttributes.GetLineColor());
   SetLineStyle(lineAttributes.GetLineStyle());
   SetLineWidth(lineAttributes.GetLineWidth());
}

////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Set color index for lines*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
///*-*                  =========================
///*-*  cindex    : color index
///*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

void  TQtPen::SetLineColor(Color_t cindex)
{
  if (fLineColor != cindex) {
    fLineColor = cindex;
    if (fLineColor >= 0)  setColor(gQt->ColorIndex(gQt->UpdateColor(fLineColor)));
  }
}
////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Set line style-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
///*-*                  ==============
///*-*    Set line style:
///*-*    if n < 0 use pre-defined Windows style:
///*-*         0 - solid lines
///*-*        -1 - solid lines
///*-*        -2 - dash line
///*-*        -3 - dot  line
///*-*        -4 - dash-dot line
///*-*        -5 - dash-dot-dot line
///*-*     < -6 - solid line
///*-*
///*-*    if n > 0 use dashed lines described by DASH(N)
///*-*    e.g. n=4,DASH=(6,3,1,3) gives a dashed-dotted line with dash length 6
///*-*    and a gap of 7 between dashes
///*-*

void  TQtPen::SetLineType(int n, int*dash)
{
/*
   SetLineStyleString(1," ");
   SetLineStyleString(2,"12 12");
   SetLineStyleString(3,"4 8");
   SetLineStyleString(4,"12 16 4 16");
   SetLineStyleString(5,"20 12 4 12");
   SetLineStyleString(6,"20 12 4 12 4 12 4 12");
   SetLineStyleString(7,"20 20");
   SetLineStyleString(8,"20 12 4 12 4 12");
   SetLineStyleString(9,"80 20");
   SetLineStyleString(10,"80 40 4 40");
*/
  static  Qt::PenStyle styles[] = {
      Qt::NoPen          // - no line at all.
     ,Qt::SolidLine      // - a simple line.
     ,Qt::DashLine       // - dashes separated by a few pixels.
     ,Qt::DotLine        // - dots separated by a few pixels.
     ,Qt::DashDotLine    // - alternate dots and dashes.
     ,Qt::DashDotDotLine // - one dash, two dots, one dash, two dotsQt::NoPen
    };

  if (n == 0 ) n = -1; // solid lines
  if (n < 0) {
    int l = -n;
    if (l >= int(sizeof(styles)/sizeof(Qt::PenStyle)) ) l = 1; // Solid line "by default"
    setStyle(styles[l]);
  }
  else if (dash) {
     // - A custom pattern defined using QPainterPathStroker::setDashPattern().
     QVector<qreal> dashes;
     int i;
     for (i=0;i<n;i++) dashes << dash[i];
     setDashPattern(dashes);
  }
}
////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Set line style-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
///*-*                  ==============
///*-*    Use pre-defined Windows style:
///*-*    linestyle =
///*-*         0 - solid lines
///*-*        -1 - solid lines
///*-*        -2 - dash line
///*-*        -3 - dot  line
///*-*        -4 - dash-dot line
///*-*        -5 - dash-dot-dot line
///*-*      < -6 - solid line
///*-*
///*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
/// Copy/Paste from TGX11::SetLineStyle (it is called "subclassing" ;-)
/// Set line style.

void  TQtPen::SetLineStyle(Style_t linestyle)
{
   if (fLineStyle != linestyle) { //set style index only if different
      fLineStyle = linestyle;
      if (linestyle > 0 && linestyle <= 5 ) {
         SetLineType(-linestyle,NULL);
      } else {
         TString st = (TString)gStyle->GetLineStyleString(linestyle);
         TObjArray *tokens = st.Tokenize(" ");
         Int_t nt;
         nt = tokens->GetEntries();
         Int_t *lstyle = new Int_t[nt];
         for (Int_t j = 0; j<nt; j++) {
            Int_t it;
            sscanf(((TObjString*)tokens->At(j))->GetName(), "%d", &it);
            lstyle[j] = (Int_t)(it/4);
         }
         SetLineType(nt,lstyle);
         delete [] lstyle;
         delete tokens;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///*-*-*-*-*-*-*-*-*-*-*Set line width*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
///*-*                  ==============
///*-*  w   : line width in pixels
///*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

void  TQtPen::SetLineWidth(Width_t w)
{
   if (w==1) w =0;
   if (fLineWidth != w) {
      fLineWidth = w;
      if (fLineWidth >= 0 )  setWidth(fLineWidth);
   }
}
