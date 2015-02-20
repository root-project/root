// @(#)root/graf:$Id$
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TMath.h"
#include "TArrow.h"
#include "TVirtualPad.h"

Float_t TArrow::fgDefaultAngle      = 60;
Float_t TArrow::fgDefaultArrowSize  = 0.05;
TString TArrow::fgDefaultOption     = ">";

ClassImp(TArrow)

//______________________________________________________________________________
/* Begin_Html
<center><h2>TArrow : to draw all kinds of arrows</h2></center>
The different arrow's formats are explained in TArrow::TArrow.
The picture below gives some examples.
<P>
Once an arrow is drawn on the screen:
<ul>
<li> One can click on one of the edges and move this edge.</li>
<li> One can click on any other arrow part to move the entire arrow.</li>
</ul>
End_Html
Begin_Macro(source)
../../../tutorials/graphics/arrow.C
End_Macro */

//______________________________________________________________________________
TArrow::TArrow(): TLine(),TAttFill()
{
   // Arrow default constructor.

   fAngle     = fgDefaultAngle;
   fArrowSize = 0.;
}

//______________________________________________________________________________
TArrow::TArrow(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
             Float_t arrowsize ,Option_t *option)
       :TLine(x1,y1,x2,y2), TAttFill(0,1001)
{
   /* Begin_Html
   Arrow normal constructor.
   <p>
   Define an arrow between points x1,y1 and x2,y2
   the arrowsize is in percentage of the pad height
   Opening angle between the two sides of the arrow is fAngle (60 degrees)
   <ul><tt>
   <li> option = ">"      -------->
   <li> option = "|->"    |------->
   <li> option = "<"      <--------
   <li> option = "<-|"    <-------|
   <li> option = "->-"    ---->----
   <li> option = "-<-"    ----<----
   <li> option = "-|>-"   ---|>----
   <li> option = "<>"     <------->
   <li> option = "<|>"    <|-----|>  arrow defined by a triangle
   </tt></ul>
   Note:
   <p>
   <ul>
   <li> If FillColor == 0 an open triangle is drawn, otherwise a full triangle
   is drawn with the fill color. The default is filled with LineColor
   <li> The "Begin" and "end" bars options can be combined with any other options.
   </ul>
   End_Html */

   fAngle       = fgDefaultAngle;
   fArrowSize   = arrowsize;
   fOption      = option;
   SetFillColor(this->GetLineColor());
}

//______________________________________________________________________________
TArrow::~TArrow()
{
   /* Begin_Html
   Arrow default destructor.
   End_Html */
}

//______________________________________________________________________________
TArrow::TArrow(const TArrow &arrow) : TLine(), TAttFill()
{
   /* Begin_Html
   Copy constructor.
   End_Html */

   fAngle     = fgDefaultAngle;
   fArrowSize = 0.;
   ((TArrow&)arrow).Copy(*this);
}

//______________________________________________________________________________
void TArrow::Copy(TObject &obj) const
{
   /* Begin_Html
   Copy this arrow to arrow.
   End_Html */

   TLine::Copy(obj);
   TAttFill::Copy(((TArrow&)obj));
   ((TArrow&)obj).fAngle      = fAngle;
   ((TArrow&)obj).fArrowSize  = fArrowSize;
   ((TArrow&)obj).fOption     = fOption;
}

//______________________________________________________________________________
void TArrow::Draw(Option_t *option)
{
   /* Begin_Html
   Draw this arrow with its current attributes.
   End_Html */

   Option_t *opt;
   if (option && strlen(option)) opt = option;
   else                          opt = (char*)GetOption();

   AppendPad(opt);

}

//______________________________________________________________________________
void TArrow::DrawArrow(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                     Float_t arrowsize ,Option_t *option)
{
   /* Begin_Html
   Draw this arrow with new coordinates.

   if arrowsize is <= 0, arrowsize will be the current arrow size
   if option="", option will be the current arrow option
   End_Html */

   Float_t size = arrowsize;
   if (size <= 0) size = fArrowSize;
   if (size <= 0) size = 0.05;
   const char* opt = option;
   if (!opt || !opt[0]) opt = fOption.Data();
   if (!opt || !opt[0]) opt = "|>";
   TArrow *newarrow = new TArrow(x1,y1,x2,y2,size,opt);
   newarrow->SetAngle(fAngle);
   TAttLine::Copy(*newarrow);
   TAttFill::Copy(*newarrow);
   newarrow->SetBit(kCanDelete);
   newarrow->AppendPad(opt);
}

//______________________________________________________________________________
void TArrow::Paint(Option_t *option)
{
   /* Begin_Html
   Paint this arrow with its current attributes.
   End_Html */

   Option_t *opt;
   if (option && strlen(option)) opt = option;
   else                          opt = (char*)GetOption();
   PaintArrow(gPad->XtoPad(fX1),gPad->YtoPad(fY1),gPad->XtoPad(fX2),gPad->YtoPad(fY2), fArrowSize, opt);
}


//______________________________________________________________________________
void TArrow::PaintArrow(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
                        Float_t arrowsize, Option_t *option)
{
   /* Begin_Html
   Draw this arrow
   End_Html */

   Int_t i;

   // Option and attributes
   TString opt = option;
   opt.ToLower();
   TAttLine::Modify();
   TAttFill::Modify();

   // Compute the gPad coordinates in TRUE normalized space (NDC)
   Int_t ix1,iy1,ix2,iy2;
   Int_t iw = gPad->GetWw();
   Int_t ih = gPad->GetWh();
   Double_t x1p,y1p,x2p,y2p;
   gPad->GetPadPar(x1p,y1p,x2p,y2p);
   ix1 = (Int_t)(iw*x1p);
   iy1 = (Int_t)(ih*y1p);
   ix2 = (Int_t)(iw*x2p);
   iy2 = (Int_t)(ih*y2p);
   Double_t wndc  = TMath::Min(1.,(Double_t)iw/(Double_t)ih);
   Double_t hndc  = TMath::Min(1.,(Double_t)ih/(Double_t)iw);
   Double_t rh    = hndc/(Double_t)ih;
   Double_t rw    = wndc/(Double_t)iw;
   Double_t x1ndc = (Double_t)ix1*rw;
   Double_t y1ndc = (Double_t)iy1*rh;
   Double_t x2ndc = (Double_t)ix2*rw;
   Double_t y2ndc = (Double_t)iy2*rh;

   // Ratios to convert user space in TRUE normalized space (NDC)
   Double_t rx1,ry1,rx2,ry2;
   gPad->GetRange(rx1,ry1,rx2,ry2);
   Double_t rx = (x2ndc-x1ndc)/(rx2-rx1);
   Double_t ry = (y2ndc-y1ndc)/(ry2-ry1);

   // Arrow position and arrow's middle in NDC space
   Double_t x1n, y1n, x2n, y2n, xm, ym;
   x1n = rx*(x1-rx1)+x1ndc;
   x2n = rx*(x2-rx1)+x1ndc;
   y1n = ry*(y1-ry1)+y1ndc;
   y2n = ry*(y2-ry1)+y1ndc;
   xm  = (x1n+x2n)/2;
   ym  = (y1n+y2n)/2;

   // Arrow heads size
   Double_t length = TMath::Sqrt(Double_t((x2n-x1n)*(x2n-x1n)+(y2n-y1n)*(y2n-y1n)));
   Double_t rSize  = 0.7*arrowsize;
   Double_t dSize  = rSize*TMath::Tan(TMath::Pi()*fAngle/360);
   Double_t cosT   = 1;
   Double_t sinT   = 0;
   if (length > 0) {
      cosT   = (x2n-x1n)/length;
      sinT   = (y2n-y1n)/length;
   }
   // Arrays holding the arrows coordinates
   Double_t x1ar[4], y1ar[4];
   Double_t x2ar[4], y2ar[4];

   // Draw the start and end bars if needed
   if (opt.BeginsWith("|-")) {
      x1ar[0] = x1n-sinT*dSize;
      y1ar[0] = y1n+cosT*dSize;
      x1ar[1] = x1n+sinT*dSize;
      y1ar[1] = y1n-cosT*dSize;
      // NDC to user coordinates
      for (i=0; i<2; i++) {
         x1ar[i] = (1/rx)*(x1ar[i]-x1ndc)+rx1;
         y1ar[i] = (1/ry)*(y1ar[i]-y1ndc)+ry1;
      }
      gPad->PaintLine(x1ar[0],y1ar[0],x1ar[1],y1ar[1]);
      opt(0) = ' ';
   }
   if (opt.EndsWith("-|")) {
      x2ar[0] = x2n-sinT*dSize;
      y2ar[0] = y2n+cosT*dSize;
      x2ar[1] = x2n+sinT*dSize;
      y2ar[1] = y2n-cosT*dSize;
      // NDC to user coordinates
      for (i=0; i<2; i++) {
         x2ar[i] = (1/rx)*(x2ar[i]-x1ndc)+rx1;
         y2ar[i] = (1/ry)*(y2ar[i]-y1ndc)+ry1;
      }
      gPad->PaintLine(x2ar[0],y2ar[0],x2ar[1],y2ar[1]);
      opt(opt.Length()-1) = ' ';
   }

   // Move arrow head's position if needed
   Double_t x1h = x1n;
   Double_t y1h = y1n;
   Double_t x2h = x2n;
   Double_t y2h = y2n;
   if (opt.Contains("->-") || opt.Contains("-|>-")) {
      x2h = xm + cosT*rSize/2;
      y2h = ym + sinT*rSize/2;
   }
   if (opt.Contains("-<-") || opt.Contains("-<|-")) {
      x1h = xm - cosT*rSize/2;
      y1h = ym - sinT*rSize/2;
   }

   // Define the arrow's head coordinates
   if (opt.Contains(">")) {
      x2ar[0] = x2h - rSize*cosT - sinT*dSize;
      y2ar[0] = y2h - rSize*sinT + cosT*dSize;
      x2ar[1] = x2h;
      y2ar[1] = y2h;
      x2ar[2] = x2h - rSize*cosT + sinT*dSize;
      y2ar[2] = y2h - rSize*sinT - cosT*dSize;
      x2ar[3] = x2ar[0];
      y2ar[3] = y2ar[0];
   }

   if (opt.Contains("<")) {
      x1ar[0] = x1h + rSize*cosT + sinT*dSize;
      y1ar[0] = y1h + rSize*sinT - cosT*dSize;
      x1ar[1] = x1h;
      y1ar[1] = y1h;
      x1ar[2] = x1h + rSize*cosT - sinT*dSize;
      y1ar[2] = y1h + rSize*sinT + cosT*dSize;
      x1ar[3] = x1ar[0];
      y1ar[3] = y1ar[0];
   }

   // Paint Arrow body
   if (opt.Contains("|>") && !opt.Contains("-|>-")) {
      x2n = x2n-cosT*rSize;
      y2n = y2n-sinT*rSize;
   }
   if (opt.Contains("<|") && !opt.Contains("-<|-")) {
      x1n = x1n+cosT*rSize;
      y1n = y1n+sinT*rSize;
   }
   x1n = (1/rx)*(x1n-x1ndc)+rx1;
   y1n = (1/ry)*(y1n-y1ndc)+ry1;
   x2n = (1/rx)*(x2n-x1ndc)+rx1;
   y2n = (1/ry)*(y2n-y1ndc)+ry1;
   gPad->PaintLine(x1n,y1n,x2n,y2n);

   // Draw the arrow's head(s)
   if (opt.Contains(">")) {
      // NDC to user coordinates
      for (i=0; i<4; i++) {
         x2ar[i] = (1/rx)*(x2ar[i]-x1ndc)+rx1;
         y2ar[i] = (1/ry)*(y2ar[i]-y1ndc)+ry1;
      }
      if (opt.Contains("|>")) {
         if (GetFillColor()) {
            gPad->PaintFillArea(3,x2ar,y2ar);
            gPad->PaintPolyLine(4,x2ar,y2ar);
         } else {
            gPad->PaintPolyLine(4,x2ar,y2ar);
         }
      } else {
         gPad->PaintPolyLine(3,x2ar,y2ar);
      }
   }
   if (opt.Contains("<")) {
      // NDC to user coordinates
      for (i=0; i<4; i++) {
         x1ar[i] = (1/rx)*(x1ar[i]-x1ndc)+rx1;
         y1ar[i] = (1/ry)*(y1ar[i]-y1ndc)+ry1;
      }
      if (opt.Contains("<|")) {
         if (GetFillColor()) {
            gPad->PaintFillArea(3,x1ar,y1ar);
            gPad->PaintPolyLine(4,x1ar,y1ar);
         } else {
            gPad->PaintPolyLine(4,x1ar,y1ar);
         }
      } else {
         gPad->PaintPolyLine(3,x1ar,y1ar);
      }
   }
}

//______________________________________________________________________________
void TArrow::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   /* Begin_Html
    // Save primitive as a C++ statement(s) on output stream out
    End_Html */

   char quote = '"';
   if (gROOT->ClassSaved(TArrow::Class())) {
      out<<"   ";
   } else {
      out<<"   TArrow *";
   }
   out<<"arrow = new TArrow("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<fArrowSize<<","<<quote<<GetDrawOption()<<quote<<");"<<std::endl;

   SaveFillAttributes(out,"arrow",0,1);
   SaveLineAttributes(out,"arrow",1,1,1);

   if (fAngle !=60) {
      out << "   arrow->SetAngle(" << GetAngle() << ");" << std::endl;
   }

   out<<"   arrow->Draw();"<<std::endl;
}


//______________________________________________________________________________
void TArrow::SetDefaultAngle(Float_t Angle)
{
   /* Begin_Html
   Set default angle.
   End_Html */

   fgDefaultAngle = Angle;
}


//______________________________________________________________________________
void TArrow::SetDefaultArrowSize (Float_t ArrowSize)
{
   /* Begin_Html
   Set default arrow sive.
   End_Html */

   fgDefaultArrowSize = ArrowSize;
}


//______________________________________________________________________________
void TArrow::SetDefaultOption(Option_t *Option)
{
   /* Begin_Html
   Set default option.
   End_Html */

   fgDefaultOption = Option;
}


//______________________________________________________________________________
Float_t TArrow::GetDefaultAngle()
{
   /* Begin_Html
   Get default angle.
   End_Html */

   return fgDefaultAngle;
}


//______________________________________________________________________________
Float_t TArrow::GetDefaultArrowSize()
{
   /* Begin_Html
   Get default arrow size.
   End_Html */

   return fgDefaultArrowSize;
}


//______________________________________________________________________________
Option_t *TArrow::GetDefaultOption()
{
   /* Begin_Html
   Get default option.
   End_Html */

   return fgDefaultOption.Data();
}
