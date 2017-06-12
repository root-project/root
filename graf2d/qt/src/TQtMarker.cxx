// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQtRConfig.h"
#include "TQtMarker.h"
#include "TAttMarker.h"
#include "TGQt.h"
#include <QtGui/QPolygon>
#include <QtGui/QPainter>
#include <QtCore/QDebug>


ClassImp(TQtMarker);

////////////////////////////////////////////////////////////////////////
//
// TQtMarker - class-utility to convert the ROOT TMarker object shape
//             into the Qt QPointArray.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

TQtMarker::TQtMarker(int n, TPoint *xy, int type) : fNumNode(n),
fChain(0), fCindex(0), fMarkerType(0),fLineWidth(0),fLineOption(false)
{
   SetPenAttributes(type);
   if (GetType() != kDot) {
      fChain.resize(n);
      TPoint *rootPoint = xy;
      for (int i=0;i<n;i++,rootPoint++)
         fChain.setPoint(i,rootPoint->fX,rootPoint->fY);
   }
}
//______________________________________________________________________________
TQtMarker::~TQtMarker(){}

////////////////////////////////////////////////////////////////////////////////
/// Assign the TQtMarker from ROOT TAttMarker

TQtMarker &TQtMarker::operator=(const TAttMarker&markerAttributes)
{
   SetMarkerAttributes(markerAttributes);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the TQtMarker from ROOT TAttMarker

TQtMarker::TQtMarker(const TAttMarker &markerAttributes)
{
   SetMarkerAttributes(markerAttributes);
}

////////////////////////////////////////////////////////////////////////////////
/// Map Qt marker  attributes to ROOT TAttMaker parameters

void TQtMarker::SetMarkerAttributes(const TAttMarker& markerAttributes)
{
   fCindex     = markerAttributes.GetMarkerColor();
   SetPenAttributes(markerAttributes.GetMarkerStyle());
   fNumNode    = Int_t(markerAttributes.GetMarkerSize());
}

////////////////////////////////////////////////////////////////////////////////
/// Pen attrbutes is 100000*LineFlag + 1000*width + "marker style"

void  TQtMarker::SetPenAttributes(int type)
{
   static const int packFactor = 1000;
   static const int lineFactor = 10000;
   fMarkerType = type%packFactor;
   fLineWidth = (type - fMarkerType)/packFactor;
   if (type >= lineFactor) {
      // Set line style
      fLineWidth -= lineFactor/packFactor;
      fLineOption = true;
   }
}
//______________________________________________________________________________
int   TQtMarker::GetNumber() const {return fNumNode;}
//______________________________________________________________________________
const QPolygon &TQtMarker::GetNodes() const {return fChain;}
//______________________________________________________________________________
int  TQtMarker::GetType()  const {return fMarkerType;}

//______________________________________________________________________________
int  TQtMarker::GetWidth() const { return fLineWidth;}

////////////////////////////////////////////////////////////////////////////////
///*-* Did we have a chain ?

void TQtMarker::SetMarker(int n, TPoint *xy, int type)
{
   fNumNode = n;
   SetPenAttributes(type);
   if (GetType() != kDot) {
      fChain.resize(n);
      TPoint *rootPoint = xy;
      for (int i=0;i<n;i++,rootPoint++)
         fChain.setPoint(i,rootPoint->fX,rootPoint->fY);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw n markers with the current attributes at positions xy.
/// p    : the external QPainter
/// n    : number of markers to draw
/// xy   : x,y coordinates of markers

void  TQtMarker::DrawPolyMarker(QPainter &p, int n, TPoint *xy)
{
   /* Set marker Color */
   const QColor &mColor  = gQt->ColorIndex(fCindex);

   p.save();
   if (this->GetWidth()>0) p.setPen(QPen(mColor,this->GetWidth()));
   else                    p.setPen(mColor);

   if( this->GetNumber() <= 0  || fLineOption )
   {
      QPolygon qtPoints(n);
      TPoint *rootPoint = xy;
      for (int i=0;i<n;i++,rootPoint++)
         qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
      if (fLineOption) p.drawPolyline(qtPoints);
      else             p.drawPoints(qtPoints);
   }
   if ( this->GetNumber() >0 ) {
      int r = this->GetNumber()/2;
      switch (this -> GetType())
      {
         case 1:
         case 3:
         default:
            p.setBrush(mColor);
            break;
         case 0:
         case 2:
            p.setBrush(Qt::NoBrush);
            break;
         case 4:
            break;
      }

      for( int m = 0; m < n; m++ )
      {
         int i;
         switch( this->GetType() )
         {
            case 0:        /* hollow circle */
            case 1:        /* filled circle */
               p.drawEllipse(xy[m].fX-r, xy[m].fY-r, 2*r, 2*r);
               break;
            case 2:        /* hollow polygon */
            case 3:        /* filled polygon */
            {
               QPolygon mxy = this->GetNodes();
               mxy.translate(xy[m].fX,xy[m].fY);
               p.drawPolygon(mxy);
               break;
            }
            case 4:        /* segmented line */
            {
               QPolygon mxy = this->GetNodes();
               mxy.translate(xy[m].fX,xy[m].fY);
               QVector<QLine> lines(this->GetNumber());
               for( i = 0; i < this->GetNumber(); i+=2 )
                  lines.push_back(QLine(mxy.point(i),mxy.point(i+1)));
               p.drawLines(lines);
               break;
            }
         }
      }
   }
   p.restore();
}
