// @(#)root/graf:$Id$
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEllipse.h"

#include <cstdlib>
#include <iostream>

#include "TROOT.h"
#include "TBuffer.h"
#include "TAttMarker.h"
#include "TVirtualPad.h"
#include "TBoxInteractive.h"
#include "TMath.h"
#include "TPoint.h"


constexpr Double_t kPI = TMath::Pi();


/** \class TEllipse
\ingroup BasicGraphics

Draw Ellipses.

The ellipse can be truncated and rotated. It is defined by its center `(x1,y1)`
and two radius `r1` and `r2`.

A minimum and maximum angle may be specified `(phimin, phimax)`.
The ellipse may be rotated with an angle `theta`. All these
angles are in degrees.
The attributes of the outline line are given via `TAttLine`.
The attributes of the fill area are given via `TAttFill`.
The picture below illustrates different types of ellipses.

When an ellipse sector only is drawn, the lines connecting the center
of the ellipse to the edges are drawn by default. One can specify
the drawing option "only" to not draw these lines or alternatively
call the function `SetNoEdges()`. To remove completely the ellipse
outline it is enough to specify 0 as line style.

Begin_Macro(source)
../../../tutorials/visualisation/graphics/ellipse.C
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// Ellipse default constructor.

TEllipse::TEllipse(): TObject(), TAttLine(), TAttFill()
{
   fX1 = 0;
   fY1 = 0;
   fR1 = 1;
   fR2 = 1;
   fPhimin = 0;
   fPhimax = 360;
   fTheta  = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Ellipse normal constructor.

TEllipse::TEllipse(Double_t x1, Double_t y1,Double_t r1,Double_t r2,Double_t phimin,Double_t phimax,Double_t theta)
      :TObject(), TAttLine(), TAttFill(0,1001)
{
   fX1     = x1;
   fY1     = y1;
   fR1     = r1;
   fR2     = r2;
   fPhimin = phimin;
   fPhimax = phimax;
   fTheta  = theta;
   if (r2 <= 0) fR2 = fR1;
}

////////////////////////////////////////////////////////////////////////////////
/// Ellipse default destructor.

TEllipse::~TEllipse()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TEllipse::TEllipse(const TEllipse &ellipse) : TObject(ellipse), TAttLine(ellipse), TAttFill(ellipse), TAttBBox2D(ellipse)
{
   fX1 = 0;
   fY1 = 0;
   fR1 = 1;
   fR2 = 1;
   fPhimin = 0;
   fPhimax = 360;
   fTheta  = 0;

   ellipse.TEllipse::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this ellipse to ellipse.

void TEllipse::Copy(TObject &obj) const
{
   TObject::Copy(obj);
   TAttLine::Copy(((TEllipse&)obj));
   TAttFill::Copy(((TEllipse&)obj));
   ((TEllipse&)obj).fX1 = fX1;
   ((TEllipse&)obj).fY1 = fY1;
   ((TEllipse&)obj).fR1 = fR1;
   ((TEllipse&)obj).fR2 = fR2;
   ((TEllipse&)obj).fPhimin = fPhimin;
   ((TEllipse&)obj).fPhimax = fPhimax;
   ((TEllipse&)obj).fTheta  = fTheta;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to an ellipse.
///
/// Compute the closest distance of approach from point px,py to this
/// ellipse. The distance is computed in pixels units.
///
/// In case of a filled ellipse the distance returned is 0 if the point
/// (px,py) is inside the ellipse, and is huge if the point is outside.

Int_t TEllipse::DistancetoPrimitive(Int_t px, Int_t py)
{
   if (!gPad) return 9999;
   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y = gPad->PadtoY(gPad->AbsPixeltoY(py));

   Double_t dxnr = x - fX1;
   Double_t dynr = y - fY1;

   Double_t ct = TMath::Cos(kPI*GetTheta()/180.0);
   Double_t st = TMath::Sin(kPI*GetTheta()/180.0);

   Double_t dx =  dxnr*ct + dynr*st;
   Double_t dy = -dxnr*st + dynr*ct;

   Double_t r1 = fR1;
   Double_t r2 = fR2;

   if (dx == 0 || r1 == 0 || r2 == 0) return 9999;
   Double_t distp = TMath::Sqrt(dx*dx + dy*dy);

   Double_t tana = dy/dx;
   tana *= tana;
   Double_t distr = TMath::Sqrt((1+tana)/(1.0/(r1*r1) + tana/(r2*r2)));
   Int_t dist = 9999;
   if (GetFillColor() && GetFillStyle()) {
      if (distr > distp) dist = 0;
   } else {
      if (TMath::Abs(distr-distp)/(r1+r2) < 0.01) dist = 0;
   }
   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this ellipse with its current attributes.

void TEllipse::Draw(Option_t *option)
{
   AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this ellipse with new coordinates.

TEllipse *TEllipse::DrawEllipse(Double_t x1, Double_t y1,Double_t r1,Double_t r2,Double_t phimin,Double_t phimax,Double_t theta,Option_t *option)
{
   TEllipse *newellipse = new TEllipse(x1, y1, r1, r2, phimin, phimax,theta);
   TAttLine::Copy(*newellipse);
   TAttFill::Copy(*newellipse);
   newellipse->SetBit(kCanDelete);
   newellipse->AppendPad(option);
   if (TestBit(kNoEdges)) newellipse->SetBit(kNoEdges);
   return newellipse;
}


class TEllipseInteractive : public TBoxInteractive {
   public:
      using TBoxInteractive::TBoxInteractive;

      void PaintOutline(TVirtualPad &parent) override
      {
         auto e = static_cast<TEllipse *> (GetObject());

         std::vector<Double_t> x, y;
         Bool_t fullcircle = e->FillPoints(parent, x, y, (newX1 + newX2) / 2, (newY1 + newY2) / 2, (newX2 - newX1) / 2, (newY2 - newY1) / 2, e->GetPhimin(), e->GetPhimax(), e->GetTheta());

         // "i" is interactive painting, "diamond" is id
         parent.PaintPolyLine(x.size(), x.data(), y.data(), "iellipse");

         if(fullcircle && !e->GetTheta())
            return;

         // draw corner markers for better visualisation
         // repeat logic of classical ROOT
         Double_t xd[4] = { (newX1+newX2)/2, newX1, (newX1+newX2)/2, newX2 };
         Double_t yd[4] = { newY2, (newY1+newY2)/2, newY1, (newY1+newY2)/2 };
         // area around corner with 6 pixels
         Double_t dx = (parent.GetX2() - parent.GetX1()) / parent.GetPadWidth() * 6;
         Double_t dy = (parent.GetY2() - parent.GetY1()) / parent.GetPadHeight() * 6;

         for (Int_t n = 0; n < 4; n++) {
            Double_t xx[5] = { xd[n] - dx, xd[n] + dx, xd[n] + dx, xd[n] - dx, xd[n] - dx };
            Double_t yy[5] = { yd[n] - dy, yd[n] - dy, yd[n] + dy, yd[n] + dy, yd[n] - dy };
            parent.PaintPolyLine(5, xx, yy, TString::Format("iellipse%d",n).Data());
         }
      }
};


////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
/// This member function is called when a line is clicked with the locator
///
/// If Left button clicked on one of the line end points, this point
/// follows the cursor until button is released.
///
/// if Middle button clicked, the line is moved parallel to itself
/// until the button is released.
///
/// NOTE that support for log scale is not implemented

void TEllipse::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad || !gPad->IsEditable())
      return;

   auto &parent = *gPad;

   auto inter = dynamic_cast<TEllipseInteractive *>(parent.Interactive(this));

   auto setNewValues = [&inter, this]() {
      SetX1((inter->newX1 + inter->newX2) / 2);
      SetR1((inter->newX2 - inter->newX1) / 2);
      SetY1((inter->newY1 + inter->newY2) / 2);
      SetR2((inter->newY2 - inter->newY1) / 2);
   };

   switch (event) {

   case kArrowKeyPress:
   case kButton1Down:
      inter = new TEllipseInteractive(kFALSE, GetX1() - GetR1(), GetY1() - GetR2(), GetX1() + GetR1(), GetY1() + GetR2());
      parent.Interactive(this, inter);

      // No break !!!

   case kMouseMotion: {
      TEllipseInteractive dummy(kFALSE);
      if (!inter) inter = &dummy;
      inter->CalcPixelCoord(parent, GetX1() - GetR1(), GetY1() - GetR2(), GetX1() + GetR1(), GetY1() + GetR2());

      if (!inter->SelectDiamondCorner(px, py, kFALSE)) {
         // refuse interactive changes
         parent.Interactive();
      } else {
         inter->SetCursor(parent, event == kButton1Down);
      }
      break;
   }

   case kArrowKeyRelease:
   case kButton1Motion:
      if (!inter)
         return;

      if (!inter->ProcessMouseMove(parent, px, py))
         return;

      inter->ApplyChanges(parent);

      if (inter->IsOpaque(parent)) {
         setNewValues();
         parent.ShowGuidelines(this, event, inter->GetGuideChar(), true);
         parent.Modified(kTRUE);
      }

      break;

   case kButton1Up:
      if (inter && inter->IsOpaque(parent))
         parent.ShowGuidelines(this, event);

      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (inter && inter->IsOpaque(parent)) {
            SetX1((inter->oldX1 + inter->oldX2) / 2);
            SetR1((inter->oldX2 - inter->oldX1) / 2);
            SetY1((inter->oldY1 + inter->oldY2) / 2);
            SetR2((inter->oldY2 - inter->oldY1) / 2);
         }
      } else if (inter && !inter->IsOpaque(parent) && (inter->newX1 != inter->newX2)) {
         setNewValues();
      }

      parent.Modified();
      parent.Interactive(); // delete interactive object
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return 1 if the point (x,y) is inside the polygon defined by
/// the ellipse 0 otherwise.
/// Author: Ole Hansen (ole@jlab.org)
Int_t TEllipse::IsInside(Double_t x, Double_t y) const
{
   x -= fX1;
   y -= fY1;
   Double_t th = fTheta * TMath::DegToRad();
   Double_t st = TMath::Sin(th);
   Double_t ct = TMath::Cos(th);
   Double_t xx =  ct * x + st * y;
   Double_t yy = -st * x + ct * y;

   if (TMath::Abs(xx) > fR1 || TMath::Abs(yy) > fR2)
      return 0;
   Double_t xn = xx / fR1;
   Double_t yn = yy / fR2;
   if (xn * xn + yn * yn > 1.)
      return 0;
   if (fPhimax - fPhimin >= 360.)
      return 1;
   Double_t phimin = std::fmod(fPhimin, 360.);
   Double_t phimax = std::fmod(fPhimax, 360.);
   Double_t phi = TMath::RadToDeg()*(TMath::Pi() + TMath::ATan2(-yy * fR1 / fR2, -xx));
   if (phi < phimin || phi > phimax)
      return 0;

   return 1;
}


////////////////////////////////////////////////////////////////////////////////
/// List this ellipse with its attributes.

void TEllipse::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("%s:  X1= %f Y1=%f R1=%f R2=%f\n",GetName(),fX1,fY1,fR1,fR2);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this ellipse with its current attributes.

void TEllipse::Paint(Option_t *option)
{
   PaintEllipse(fX1, fY1, fR1, fR2, fPhimin, fPhimax, fTheta, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Fill points which can be used for the painting
/// Return true if full 360 ellipse is created

Bool_t TEllipse::FillPoints(TVirtualPad &pad, std::vector<Double_t> &x, std::vector<Double_t> &y,
                           Double_t x1, Double_t y1, Double_t r1, Double_t r2, Double_t phimin, Double_t phimax, Double_t theta)
{
   const Int_t np = 200;

   Double_t phi1 = TMath::Min(phimin, phimax);
   Double_t phi2 = TMath::Max(phimin, phimax);

   //set number of points approximatively proportional to the ellipse circumference
   Double_t circ = kPI*(r1+r2)*(phi2-phi1)/360;
   Int_t n = (Int_t)(np*circ/((pad.GetX2() - pad.GetX1())+(pad.GetY2()-pad.GetY1())));
   Bool_t full_circle = phi2-phi1 >= 360;
   n = TMath::Min(np, TMath::Max(n, (Int_t) (full_circle ? 36 : 8)));

   x.resize(n + (full_circle ? 1 : 3));
   y.resize(n + (full_circle ? 1 : 3));

   Double_t dphi = (phi2-phi1)*kPI/(180*n);
   Double_t ct   = TMath::Cos(kPI*theta/180);
   Double_t st   = TMath::Sin(kPI*theta/180);
   for (Int_t i = 0; i <= n; i++) {
      Double_t angle = phi1*kPI/180 + i*dphi;
      Double_t dx    = r1*TMath::Cos(angle);
      Double_t dy    = r2*TMath::Sin(angle);
      x[i]  = pad.XtoPad(x1 + dx*ct - dy*st);
      y[i]  = pad.YtoPad(y1 + dx*st + dy*ct);
   }
   if (!full_circle) {
      x[n+1] = pad.XtoPad(x1);
      y[n+1] = pad.YtoPad(y1);
      x[n+2] = x[0];
      y[n+2] = y[0];
   }

   return full_circle;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this ellipse with new coordinates.

void TEllipse::PaintEllipse(Double_t x1, Double_t y1, Double_t r1, Double_t r2,
                            Double_t phimin, Double_t phimax, Double_t theta,
                            Option_t *option)
{
   if (!gPad) return;

   std::vector<Double_t> x, y;
   Bool_t full_circle = FillPoints(*gPad, x, y, x1, y1, r1, r2, phimin, phimax, theta);

   TAttFill::ModifyOn(*gPad);  //Change fill attributes only if necessary
   TAttLine::ModifyOn(*gPad);  //Change line attributes only if necessary

   if (GetFillStyle() > 0)
      gPad->PaintFillArea(x.size() - 1, x.data(), y.data());

   if (GetLineStyle() > 0) {
      TString opt = option;
      opt.ToLower();
      Bool_t less_points = !full_circle && (TestBit(kNoEdges) || opt.Contains("only"));
      gPad->PaintPolyLine(x.size() - (less_points ? 2 : 0), x.data(), y.data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this ellipse with its attributes.

void TEllipse::Print(Option_t *) const
{
   printf("Ellipse:  X1=%f Y1=%f R1=%f R2=%f",fX1,fY1,fR1,fR2);
   if (GetLineColor() != 1) printf(" Color=%d",GetLineColor());
   if (GetLineStyle() != 1) printf(" Style=%d",GetLineStyle());
   if (GetLineWidth() != 1) printf(" Width=%d",GetLineWidth());
   printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TEllipse::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(
      out, Class(), "ellipse",
      TString::Format("%g, %g, %g, %g, %g, %g, %g", fX1, fY1, fR1, fR2, fPhimin, fPhimax, fTheta));

   SaveFillAttributes(out, "ellipse", 0, 1001);
   SaveLineAttributes(out, "ellipse", 1, 1, 1);

   if (GetNoEdges())
      out << "   ellipse->SetNoEdges();\n";

   SavePrimitiveDraw(out, "ellipse", option);
}

////////////////////////////////////////////////////////////////////////////////
/// Return kTRUE if kNoEdges bit is set, kFALSE otherwise.

Bool_t TEllipse::GetNoEdges() const
{
   return TestBit(kNoEdges) ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// if  noEdges = kTRUE the lines connecting the center to the edges
/// will not be drawn.
/// default is to draw the edges.

void TEllipse::SetNoEdges(Bool_t noEdges)
{
   if (noEdges) SetBit(kNoEdges);
   else         ResetBit(kNoEdges);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TEllipse.

void TEllipse::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TEllipse::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      TAttFill::Streamer(R__b);
      Float_t x1,y1,r1,r2,phimin,phimax,theta;
      R__b >> x1;     fX1 = x1;
      R__b >> y1;     fY1 = y1;
      R__b >> r1;     fR1 = r1;
      R__b >> r2;     fR2 = r2;
      R__b >> phimin; fPhimin = phimin;
      R__b >> phimax; fPhimax = phimax;
      R__b >> theta;  fTheta  = theta;
      R__b.CheckByteCount(R__s, R__c, TEllipse::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TEllipse::Class(),this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the bounding Box of the Ellipse, currently not taking into
/// account the rotating angle.

Rectangle_t TEllipse::GetBBox()
{
   Rectangle_t BBox{0, 0, 0, 0};
   if (gPad) {
      BBox.fX = gPad->XtoPixel(fX1 - fR1);
      BBox.fY = gPad->YtoPixel(fY1 + fR2);
      BBox.fWidth = gPad->XtoPixel(fX1 + fR1) - gPad->XtoPixel(fX1 - fR1);
      BBox.fHeight = gPad->YtoPixel(fY1 - fR2) - gPad->YtoPixel(fY1 + fR2);
   }
   return BBox;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the center of the Ellipse as TPoint in pixels

TPoint TEllipse::GetBBoxCenter()
{
   TPoint p(0, 0);
   if (gPad) {
      p.SetX(gPad->XtoPixel(fX1));
      p.SetY(gPad->YtoPixel(fY1));
   }
   return p;
}

////////////////////////////////////////////////////////////////////////////////
/// Set X coordinate of the center of the Ellipse

void TEllipse::SetBBoxCenterX(const Int_t x)
{
   SetX1(GetXCoord(x));
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y coordinate of the center of the Ellipse

void TEllipse::SetBBoxCenterY(const Int_t y)
{
   SetY1(GetYCoord(y));
}

////////////////////////////////////////////////////////////////////////////////
/// Set left hand side of BoundingBox to a value
/// (resize in x direction on left)

void TEllipse::SetBBoxX1(const Int_t x)
{
   Double_t x1 = GetXCoord(x);
   if (x1 > fX1+fR1) return;

   SetR1((fX1+fR1-x1)*0.5);
   SetX1(x1 + fR1);
}

////////////////////////////////////////////////////////////////////////////////
/// Set right hand side of BoundingBox to a value
/// (resize in x direction on right)

void TEllipse::SetBBoxX2(const Int_t x)
{
   Double_t x2 = GetXCoord(x);
   if (x2 < fX1-fR1) return;

   SetR1((x2-fX1+fR1)*0.5);
   SetX1(x2 - fR1);
}

////////////////////////////////////////////////////////////////////////////////
/// Set top of BoundingBox to a value (resize in y direction on top)

void TEllipse::SetBBoxY1(const Int_t y)
{
   Double_t y1 = GetYCoord(y);
   if (y1 < fY1-fR2) return;

   SetR2((y1-fY1+fR2)*0.5);
   SetY1(y1 - fR2);
}

////////////////////////////////////////////////////////////////////////////////
/// Set bottom of BoundingBox to a value
/// (resize in y direction on bottom)

void TEllipse::SetBBoxY2(const Int_t y)
{
   Double_t y2 = GetYCoord(y);
   if (y2 > fY1+fR2) return;

   SetR2((fY1+fR2-y2)*0.5);
   SetY1(y2 + fR2);
}
