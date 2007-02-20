// @(#)root/base:$Name:  $:$Id: TAttMarker.cxx,v 1.11 2007/02/16 17:23:33 couet Exp $
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "Strlen.h"
#include "TAttMarker.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TVirtualPadEditor.h"
#include "TColor.h"

ClassImp(TAttMarker)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Marker Attributes class</h2></center>

This class is used (in general by secondary inheritance)
by many other classes (graphics, histograms). It holds all the markers
attributes.

<h3>Marker attributes</h3>
The marker attributes are:
<ul>
<li><a href="#M1">Marker color.</a></li>
<li><a href="#M2">Marker style.</a></li>
<li><a href="#M3">Marker size.</a></li>
</ul>

<a name="M1"></a><h3>Marker color</h3>
The marker color is a color index (integer) pointing in the ROOT color
table. The following table shows the first 50 default colors.
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Marker colors",0,0,500,200);
   c.DrawColorTable();
   return c;
}
End_Macro

Begin_Html
<a name="M2"></a><h3>Marker style</h3>
The Marker style defines the markers' shape.
The following table gives the list of the currently supported markers (screen 
and PostScript) style. Each marker style is identified by an integer number 
(first column) corresponding to a marker shape (second column) and can be also
accessed via a global name (third column).
<center><table>
<tr><td>  1 </td><td> dot                 </td><td> kDot              </td></tr>
<tr><td>  2 </td><td> +                   </td><td> kPlus             </td></tr>
<tr><td>  3 </td><td> *                   </td><td> kStar             </td></tr>
<tr><td>  4 </td><td> o                   </td><td> kCircle           </td></tr>
<tr><td>  5 </td><td> x                   </td><td> kMultiply         </td></tr>
<tr><td>  6 </td><td> small scalable down </td><td> kFullDotSmall     </td></tr>
<tr><td>  7 </td><td> medium scalable     </td><td> kFullDotMedium    </td></tr>
<tr><td>  8 </td><td> large scalable down </td><td> kFullDotLarge     </td></tr>
<tr><td>  9 -->19 </td><td> dot                                       </td></tr>
<tr><td> 20 </td><td> full circle         </td><td> kFullCircle       </td></tr>
<tr><td> 21 </td><td> full square         </td><td> kFullSquare       </td></tr>
<tr><td> 22 </td><td> full triangle up    </td><td> kFullTriangleUp   </td></tr>
<tr><td> 23 </td><td> full triangle down  </td><td> kFullTriangleDown </td></tr>
<tr><td> 24 </td><td> open circle         </td><td> kOpenCircle       </td></tr>
<tr><td> 25 </td><td> open square         </td><td> kOpenSquare       </td></tr>
<tr><td> 26 </td><td> open triangle up    </td><td> kOpenTriangleUp   </td></tr>
<tr><td> 27 </td><td> open diamond        </td><td> kOpenDiamond      </td></tr>
<tr><td> 28 </td><td> open cross          </td><td> kOpenCross        </td></tr>
<tr><td> 29 </td><td> open star           </td><td> kOpenStar         </td></tr>
<tr><td> 30 </td><td> full star           </td><td> kFullStar         </td></tr>
</table></center>
End_Html
Begin_Macro(source)
{
   TCanvas *c = new TCanvas("c","Marker types",0,0,500,200);
   TMarker marker;
   marker.DisplayMarkerTypes();
   return c;
}
End_Macro

Begin_Html
<a name="M3"></a><h3>Marker size</h3>
Various marker sizes are shown in the figure below. The default marker size=1
is shown in the top left corner. Marker sizes smaller than 1 can be
specified. The marker size does not refer to any coordinate systems, it is an
absolute value. Therefore the marker size is not affected by any change 
in TPad's scale.
End_Html
Begin_Macro(source)
{
   c = new TCanvas("c","Marker sizes",0,0,500,200);
   TMarker marker;
   marker.SetMarkerStyle(3);
   Double_t x = 0;
   Double_t dx = 1/6.0;
   for (Int_t i=1; i<6; i++) {
      x += dx;
      marker.SetMarkerSize(i*0.2); marker.DrawMarker(x,.165);
      marker.SetMarkerSize(i*0.8); marker.DrawMarker(x,.495);
      marker.SetMarkerSize(i*1.0); marker.DrawMarker(x,.835);
   }
   return c;
}
End_Macro */


//______________________________________________________________________________
TAttMarker::TAttMarker()
{
   // TAttMarker default constructor.
   //
   // Default text attributes are taking from the current style.

   if (!gStyle) return;
   fMarkerColor = gStyle->GetMarkerColor();
   fMarkerStyle = gStyle->GetMarkerStyle();
   fMarkerSize  = gStyle->GetMarkerSize();
}


//______________________________________________________________________________
TAttMarker::TAttMarker(Color_t color, Style_t style, Size_t msize)
{
   // TAttMarker normal constructor.
   //
   // Text attributes are taking from the argument list
   //    color : Marker Color Index
   //    style : Marker style (from 1 to 30)
   //    size  : marker size (float)

   fMarkerColor = color;
   fMarkerSize  = msize;
   fMarkerStyle = style;
}


//______________________________________________________________________________
TAttMarker::~TAttMarker()
{
   // TAttMarker destructor.
}


//______________________________________________________________________________
void TAttMarker::Copy(TAttMarker &attmarker) const
{
   // Copy this marker attributes to a new TAttMarker.

   attmarker.fMarkerColor  = fMarkerColor;
   attmarker.fMarkerStyle  = fMarkerStyle;
   attmarker.fMarkerSize   = fMarkerSize;
}


//______________________________________________________________________________
void TAttMarker::Modify()
{
   // Change current marker attributes if necessary.

   if (!gPad) return;
   if (!gPad->IsBatch()) {
      gVirtualX->SetMarkerColor(fMarkerColor);
      gVirtualX->SetMarkerSize (fMarkerSize);
      gVirtualX->SetMarkerStyle(fMarkerStyle);
   }

   gPad->SetAttMarkerPS(fMarkerColor,fMarkerStyle,fMarkerSize);
}


//______________________________________________________________________________
void TAttMarker::ResetAttMarker(Option_t *)
{
   // Reset this marker attributes to the default values.

   fMarkerColor  = 1;
   fMarkerStyle  = 1;
   fMarkerSize   = 1;
}


//______________________________________________________________________________
void TAttMarker::SaveMarkerAttributes(ostream &out, const char *name, Int_t coldef, Int_t stydef, Int_t sizdef)
{
   // Save line attributes as C++ statement(s) on output stream out.

   if (fMarkerColor != coldef) {
      if (fMarkerColor > 228) {
         TColor::SaveColor(out, fMarkerColor);
         out<<"   "<<name<<"->SetMarkerColor(ci);" << endl;
      } else
         out<<"   "<<name<<"->SetMarkerColor("<<fMarkerColor<<");"<<endl;
   }
   if (fMarkerStyle != stydef) {
      out<<"   "<<name<<"->SetMarkerStyle("<<fMarkerStyle<<");"<<endl;
   }
   if (fMarkerSize != sizdef) {
      out<<"   "<<name<<"->SetMarkerSize("<<fMarkerSize<<");"<<endl;
   }
}


//______________________________________________________________________________
void TAttMarker::SetMarkerAttributes()
{
   // Invoke the DialogCanvas Marker attributes.

   TVirtualPadEditor::UpdateMarkerAttributes(fMarkerColor,fMarkerStyle,fMarkerSize);
}
