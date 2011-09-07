// @(#)root/graf:$Id$
// Author: Matthew.Adam.Dobbs   06/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>

#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include "TMarker.h"
#include "TLegend.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TROOT.h"
#include "TLegendEntry.h"
#include "Riostream.h"
#include "TMultiGraph.h"
#include "THStack.h"


ClassImp(TLegend)


//______________________________________________________________________________
/* Begin_Html
<center><h2>Legend class</h2></center>
This class displays a legend box (TPaveText) containing several legend entries.
Each legend entry is made of a reference to a ROOT object, a text label and an
option specifying which graphical attributes (marker/line/fill) should be
displayed.
<p>
The following example shows how to create a legend. In this example the legend
contains a histogram, a function and a graph. The histogram is put in the legend
using its reference pointer whereas the graph and the function are added
using their names. Note that, because <tt>TGraph</tt> contructors do not have the
<tt>TGraph</tt> name as parameter, the graph name should be specified using the
<tt>SetName</tt> method.
<p>
When an object is added by name, a scan is performed on the list of objects
contained in the current pad (<tt>gPad</tt>) and also in the possible
<tt>TMultiGraph</tt> and <tt>THStack</tt> present in the pad. If a matching
name is found, the coresponding object is added in the legend using its pointer.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,500);
   gStyle->SetOptStat(0);

   TH1F *h1 = new TH1F("h1","TLegend Example",200,-10,10);
   h1->FillRandom("gaus",30000);
   h1->SetFillColor(kGreen);
   h1->SetFillStyle(3003);
   h1->Draw();

   TF1 *f1=new TF1("f1","1000*TMath::Abs(sin(x)/x)",-10,10);
   f1->SetLineColor(kBlue);
   f1->SetLineWidth(4);
   f1->Draw("same");

   const Int_t n = 20;
   Double_t x[n], y[n], ex[n], ey[n];
   for (Int_t i=0;i<n;i++) {
      x[i]  = i*0.1;
      y[i]  = 1000*sin(x[i]+0.2);
      x[i]  = 17.8*x[i]-8.9;
      ex[i] = 1.0;
      ey[i] = 10.*i;
   }
   TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);
   gr->SetName("gr");
   gr->SetLineColor(kRed);
   gr->SetLineWidth(2);
   gr->SetMarkerStyle(21);
   gr->SetMarkerSize(1.3);
   gr->SetMarkerColor(7);
   gr->Draw("P");

   leg = new TLegend(0.1,0.7,0.48,0.9);
   leg->SetHeader("The Legend Title");
   leg->AddEntry(h1,"Histogram filled with random numbers","f");
   leg->AddEntry("f1","Function abs(#frac{sin(x)}{x})","l");
   leg->AddEntry("gr","Graph with error bars","lep");
   leg->Draw();

   return c1;
}
End_Macro
Begin_Html

Note that the <tt>TPad</tt> class has a method to build automatically a legend
for all objects in the pad. It is called <tt>TPad::BuildLegend()</tt>.
<p>
Each item in the legend is added using the <tt>AddEntry</tt> method. This
method defines the object to be added (by reference or name), the label
associated to this object and an option which a combination of:
<ul>
<li> L: draw line associated with TAttLine if obj inherits from TAttLine
<li> P: draw polymarker associated with TAttMarker if obj inherits from TAttMarker
<li> F: draw a box with fill associated wit TAttFill if obj inherits TAttFill
<li> E: draw vertical error bar if option "L" is also specified
</ul>
<p>
As shown in the following example, passing a NULL pointer as first parameter in
<tt>AddEntry</tt> is also valid. This allows to add text or blank lines in a
legend.

End_Html
Begin_Macro(source)
{
   TCanvas *c2 = new TCanvas("c2","c2",500,300);

   TLegend* leg = new TLegend(0.2, 0.2, .8, .8);
   TH1* h = new TH1F("", "", 1, 0, 1);

   leg->AddEntry(h, "Histogram \"h\"", "l"); 
   leg->AddEntry((TObject*)0, "", "");  
   leg->AddEntry((TObject*)0, "Some text", "");
   leg->AddEntry((TObject*)0, "", "");  
   leg->AddEntry(h, "Histogram \"h\" again", "l");

   leg->Draw();
   return c2;
}
End_Macro
Begin_Html
It is possible to draw the legend entries over several columns using
the method <tt>SetNColumns()</tt> like in the following example.

End_Html
Begin_Macro(source)
{
   TCanvas *c3 = new TCanvas("c2","c2",500,300);
 
   TLegend* leg = new TLegend(0.2, 0.2, .8, .8);
   TH1* h = new TH1F("", "", 1, 0, 1);
 
   leg-> SetNColumns(2);
 
   leg->AddEntry(h, "Column 1 line 1", "l");
   leg->AddEntry(h, "Column 2 line 1", "l");
   leg->AddEntry(h, "Column 1 line 2", "l");
   leg->AddEntry(h, "Column 2 line 2", "l");
 
   leg->Draw();
   return c3;
}
End_Macro
*/


//______________________________________________________________________________
TLegend::TLegend(): TPave(), TAttText()
{
   /* Begin_Html
   Default constructor.
   End_Html */

   fPrimitives = 0;
   SetDefaults();
}


//______________________________________________________________________________
TLegend::TLegend( Double_t x1, Double_t y1,Double_t x2, Double_t y2,
                  const char *header, Option_t *option)
        :TPave(x1,y1,x2,y2,4,option), TAttText(12,0,1,gStyle->GetTextFont(),0)
{
   /* Begin_Html
   Normal Contructor.
   <p>
   A TLegend is a Pave with several TLegendEntry(s).
   x1,y1,x2,y2 are the coordinates of the Legend in the current pad
   (in normalized coordinates by default)
   "header" is the title that will be displayed at the top of the legend
   it is treated like a regular entry and supports TLatex. The default
   is no header (header = 0).
   The options are the same as for TPave Default = "brNDC"
   End_Html */

   fPrimitives = new TList;
   if ( header && strlen(header) > 0) {
      TLegendEntry *headerEntry = new TLegendEntry( 0, header, "h" );
      headerEntry->SetTextAlign(0);
      headerEntry->SetTextAngle(0);
      headerEntry->SetTextColor(0);
      headerEntry->SetTextFont(62); // default font is 62 for the header
      headerEntry->SetTextSize(0);
      fPrimitives->AddFirst(headerEntry);
   }
   SetDefaults();
   SetBorderSize(gStyle->GetLegendBorderSize());
}


//______________________________________________________________________________
TLegend::TLegend( const TLegend &legend ) : TPave(legend), TAttText(legend),
                                            fPrimitives(0)
{
   /* Begin_Html
   Copy constuctor.
   End_Html */

  if (legend.fPrimitives) {
      fPrimitives = new TList();
      TListIter it(legend.fPrimitives);
      while (TLegendEntry *e = (TLegendEntry *)it.Next()) {
         TLegendEntry *newentry = new TLegendEntry(*e);
         fPrimitives->Add(newentry);
      }
   }
   ((TLegend&)legend).Copy(*this);
}


//______________________________________________________________________________
TLegend& TLegend::operator=(const TLegend &lg)
{
   /* Begin_Html
   Assignment operator.
   End_Html */

   if(this!=&lg) {
      TPave::operator=(lg);
      TAttText::operator=(lg);
      fPrimitives=lg.fPrimitives;
      fEntrySeparation=lg.fEntrySeparation;
      fMargin=lg.fMargin;
      fNColumns=lg.fNColumns;
   }
   return *this;
}


//______________________________________________________________________________
TLegend::~TLegend()
{
   /* Begin_Html
   Default destructor.
   End_Html */

   if (fPrimitives) fPrimitives->Delete();
   delete fPrimitives;
   fPrimitives = 0;
}


//______________________________________________________________________________
TLegendEntry *TLegend::AddEntry(const TObject *obj, const char *label, Option_t *option)
{
   /* Begin_Html
   Add a new entry to this legend. "obj" is the object to be represented.
   "label" is the text you wish to associate with obj in the legend.
   If "label" is null or empty, the title of the object will be used.
   <p>
   Options are:
   <ul>
   <li> L: draw line associated with TAttLine if obj inherits from TAttLine
   <li> P: draw polymarker associated with TAttMarker if obj inherits from TAttMarker
   <li> F: draw a box with fill associated wit TAttFill if obj inherits TAttFill
   <li> E: draw vertical error bar if option "L" is also specified
   </ul>
   End_Html */

   const char *lab = label;

   if (obj && (!label || strlen(label)==0)) lab = obj->GetTitle();
   TLegendEntry *newentry = new TLegendEntry( obj, lab, option );
   if ( !fPrimitives ) fPrimitives = new TList;
   fPrimitives->Add(newentry);
   return newentry;
}


//______________________________________________________________________________
TLegendEntry *TLegend::AddEntry(const char *name, const char *label, Option_t *option)
{
   /* Begin_Html
   Add a new entry to this legend. "name" is the name of an object in the pad to
   be represented label is the text you wish to associate with obj in the legend
   if label is null or empty, the title of the object will be used.
   <p>
   Options are:
   <ul>
   <li> L: draw line associated with TAttLine if obj inherits from TAttLine
   <li> P: draw polymarker associated with TAttMarker if obj inherits from TAttMarker
   <li> F: draw a box with fill associated wit TAttFill if obj inherits TAttFill
   <li> E: draw vertical error bar if option "L" is also specified
   </ul>
   End_Html */

   TObject *obj = gPad->FindObject(name);

   // If the object "name" has not been found, the following code tries to
   // find it in TMultiGraph or THStack possibly present in the current pad.
   if (!obj) {
      TList *lop = gPad->GetListOfPrimitives();
      if (lop) {
         TObject *o=0;
         TIter next(lop);
         while( (o=next()) ) {
            if ( o->InheritsFrom(TMultiGraph::Class() ) ) {
               TList * grlist = ((TMultiGraph *)o)->GetListOfGraphs();
               obj = grlist->FindObject(name);
               if (obj) continue;
            }
            if ( o->InheritsFrom(THStack::Class() ) ) {
               TList * hlist = ((THStack *)o)->GetHists();
               obj = hlist->FindObject(name);
               if (obj) continue;
            }
         }
      }
   }

   return AddEntry( obj, label, option );
}


//______________________________________________________________________________
void TLegend::Clear( Option_t *)
{
   /* Begin_Html
   Clear all entries in this legend, including the header.
   End_Html */

   if (!fPrimitives) return;
   fPrimitives->Delete();
}


//______________________________________________________________________________
void TLegend::Copy( TObject &obj ) const
{
   /* Begin_Html
   Copy this legend into "obj".
   End_Html */

   TPave::Copy(obj);
   TAttText::Copy((TLegend&)obj);
   ((TLegend&)obj).fEntrySeparation = fEntrySeparation;
   ((TLegend&)obj).fMargin = fMargin;
   ((TLegend&)obj).fNColumns = fNColumns;
}


//______________________________________________________________________________
void TLegend::DeleteEntry()
{
   /* Begin_Html
   Delete entry at the mouse position.
   End_Html */

   if ( !fPrimitives ) return;
   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( !entry ) return;
   fPrimitives->Remove(entry);
   delete entry;
}


//______________________________________________________________________________
void TLegend::Draw( Option_t *option )
{
   /* Begin_Html
   Draw this legend with its current attributes.
   End_Html */

   AppendPad(option);
}


//______________________________________________________________________________
void TLegend::EditEntryAttFill()
{
   /* Begin_Html
   Edit the fill attributes for the entry pointed by the mouse.
   End_Html */

   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( !entry ) return;
   gROOT->SetSelectedPrimitive( entry );
   entry->SetFillAttributes();
}


//______________________________________________________________________________
void TLegend::EditEntryAttLine()
{
   /* Begin_Html
   Edit the line attributes for the entry pointed by the mouse.
   End_Html */

   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( !entry ) return;
   gROOT->SetSelectedPrimitive( entry );
   entry->SetLineAttributes();
}


//______________________________________________________________________________
void TLegend::EditEntryAttMarker()
{
   /* Begin_Html
   Edit the marker attributes for the entry pointed by the mouse.
   End_Html */

   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( !entry ) return;
   gROOT->SetSelectedPrimitive( entry );
   entry->SetMarkerAttributes();
}


//______________________________________________________________________________
void TLegend::EditEntryAttText()
{
   /* Begin_Html
   Edit the text attributes for the entry pointed by the mouse.
   End_Html */

   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( !entry ) return;
   gROOT->SetSelectedPrimitive( entry );
   entry->SetTextAttributes();
}


//______________________________________________________________________________
TLegendEntry *TLegend::GetEntry() const
{
   /* Begin_Html
   Get entry pointed to by the mouse.
   This method is mostly a tool for other methods inside this class.
   End_Html */

   Int_t nRows = GetNRows();
   if ( nRows == 0 ) return 0;

   Double_t ymouse = gPad->AbsPixeltoY(gPad->GetEventY())-fY1;
   Double_t yspace = (fY2 - fY1)/nRows;
   
   Int_t nColumns = GetNColumns();
   Double_t xmouse = gPad->AbsPixeltoX(gPad->GetEventX())-fX1;
   Double_t xspace = 0.;
   if (nColumns > 0) xspace = (fX2 - fX1)/nColumns;

   Int_t ix = 1;
   if (xspace > 0.) ix = (Int_t)(xmouse/xspace)+1;
   if (ix > nColumns) ix = nColumns;
   if (ix < 1)        ix = 1;
   
   Int_t iy = nRows-(Int_t)(ymouse/yspace);
   if (iy > nRows) iy = nRows;
   if (iy < 1)     iy = 1;
   
   Int_t nloops = TMath::Min(ix+(nColumns*(iy-1)), fPrimitives->GetSize());

   TIter next(fPrimitives);
   TLegendEntry *entry = 0;

   for (Int_t i=1; i<= nloops; i++) entry = (TLegendEntry *)next();

   return entry;
}


//______________________________________________________________________________
const char *TLegend::GetHeader() const
{
   /* Begin_Html
   Returns the header, which is the title that appears at the top
   of the legend.
   End_Html */

   if ( !fPrimitives ) return 0;
      TIter next(fPrimitives);
   TLegendEntry *first;   // header is always the first entry
   if ((  first = (TLegendEntry*)next()  )) {
      TString opt = first->GetOption();
      opt.ToLower();
      if ( opt.Contains("h") ) return first->GetLabel();
   }
   return 0;
}


//______________________________________________________________________________
void TLegend::InsertEntry( const char* objectName, const char* label, Option_t* option)
{
   /* Begin_Html
   Add a new entry before the entry at the mouse position.
   End_Html */

   TLegendEntry* beforeEntry = GetEntry();   // get entry pointed by the mouse
   TObject *obj = gPad->FindObject( objectName );

   // note either obj OR beforeEntry may be zero at this point

   TLegendEntry *newentry = new TLegendEntry( obj, label, option );

   if ( !fPrimitives ) fPrimitives = new TList;
   if ( beforeEntry ) {
      fPrimitives->AddBefore( (TObject*)beforeEntry, (TObject*)newentry );
   } else {
      fPrimitives->Add((TObject*)newentry);
   }
}


//______________________________________________________________________________
void TLegend::Paint( Option_t* option )
{
   /* Begin_Html
   Paint this legend with its current attributes.
   End_Html */

   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);
   PaintPrimitives();
}


//______________________________________________________________________________
Int_t TLegend::GetNRows() const
{
   /* Begin_Html
   Get the number of rows.
   End_Html */

   Int_t nEntries = 0;
   if ( fPrimitives ) nEntries = fPrimitives->GetSize();
   if ( nEntries == 0 ) return 0;

   Int_t nRows;
   if(GetHeader() != NULL) nRows = 1 + (Int_t) TMath::Ceil((Double_t) (nEntries-1)/fNColumns);
   else  nRows = (Int_t) TMath::Ceil((Double_t) nEntries/fNColumns);

   return nRows;
}


//______________________________________________________________________________
void TLegend::SetNColumns(Int_t nColumns)
{
   /* Begin_Html
   Set the number of columns for the legend. The header, if set, is given
   its own row. After that, every nColumns entries are inserted into the
   same row. For example, if one calls legend.SetNColumns(2), and there
   is no header, then the first two TObjects added to the legend will be
   in the first row, the next two will appear in the second row, and so on.
   End_Html */

   if(nColumns < 1) {
      Warning("TLegend::SetNColumns", "illegal value nColumns = %d; keeping fNColumns = %d", nColumns, fNColumns);
      return;
   }
   fNColumns = nColumns;
}


//______________________________________________________________________________
void TLegend::PaintPrimitives()
{
   /* Begin_Html
   Paint the entries (list of primitives) for this legend.
   End_Html */

   Int_t nRows = GetNRows();
   if ( nRows == 0 ) return;

   // Evaluate text size as a function of the number of entries
   //  taking into account their real size after drawing latex
   // Note: in pixel coords y1 > y2=0, but x2 > x1=0
   //       in NDC          y2 > y1,   and x2 > x1

   Double_t x1 = fX1NDC;
   Double_t y1 = fY1NDC;
   Double_t x2 = fX2NDC;
   Double_t y2 = fY2NDC;
   Double_t margin = fMargin*( x2-x1 )/fNColumns;
   Double_t boxwidth = margin;
   Double_t boxw = boxwidth*0.35;
   Double_t yspace = (y2-y1)/nRows;
   Double_t textsize = GetTextSize();
   Double_t save_textsize = textsize;
   Double_t* columnWidths = new Double_t[fNColumns];
   memset(columnWidths, 0, fNColumns*sizeof(Double_t));

   if ( textsize == 0 ) {
      textsize = ( 1. - fEntrySeparation ) * yspace;

      // find the max width and height (in pad coords) of one latex entry label
      Double_t maxentrywidth = 0, maxentryheight = 0;
      TIter nextsize(fPrimitives);
      TLegendEntry *entrysize;
      Int_t iColumn = 0;
      while (( entrysize = (TLegendEntry *)nextsize() )) {
         TLatex entrytex( 0, 0, entrysize->GetLabel() );
         entrytex.SetNDC();
         Style_t tfont = entrysize->GetTextFont();
         if (tfont == 0) tfont = GetTextFont();
         entrytex.SetTextFont(tfont);
         entrytex.SetTextSize(textsize);
         if ( entrytex.GetYsize() > maxentryheight ) {
            maxentryheight = entrytex.GetYsize();
         }
         TString opt = entrysize->GetOption();
         opt.ToLower();
         if ( opt.Contains("h") ) {
            if ( entrytex.GetXsize() > maxentrywidth ) {
               maxentrywidth = entrytex.GetXsize();
            }
         } else {
            if ( entrytex.GetXsize() > columnWidths[iColumn] ) {
               columnWidths[iColumn] = entrytex.GetXsize();
            }
            iColumn++;
            iColumn %= fNColumns;
         }
         Double_t tmpMaxWidth = 0.0;
         for(int i=0; i<fNColumns; i++) tmpMaxWidth += columnWidths[i];
         if ( tmpMaxWidth > maxentrywidth) maxentrywidth = tmpMaxWidth;
      }
      // make sure all labels fit in the allotted space
      Double_t tmpsize_h = maxentryheight /(gPad->GetY2() - gPad->GetY1());
      textsize = TMath::Min( textsize, tmpsize_h );
      Double_t tmpsize_w = textsize*(fX2-fX1)*(1.0-fMargin)/maxentrywidth;
      if(fNColumns > 1) tmpsize_w = textsize*(fX2-fX1)*(1.0-fMargin-fColumnSeparation)/maxentrywidth;
      textsize = TMath::Min( textsize, tmpsize_w );
      SetTextSize( textsize );
   }

   // Update column widths, put into NDC units
   // block off this section of code to make sure all variables are local:
   // don't want to ruin initialization of these variables later on
   {
      TIter next(fPrimitives);
      TLegendEntry *entry;
      Int_t iColumn = 0;
      memset(columnWidths, 0, fNColumns*sizeof(Double_t));
      while (( entry = (TLegendEntry *)next() )) {
         TLatex entrytex( 0, 0, entry->GetLabel() );
         entrytex.SetNDC();
         Style_t tfont = entry->GetTextFont();
         if (tfont == 0) tfont = GetTextFont();
         entrytex.SetTextFont(tfont);
         if(entry->GetTextSize() == 0) entrytex.SetTextSize(textsize);
         TString opt = entry->GetOption();
         opt.ToLower();
         if (!opt.Contains("h")) {
            if ( entrytex.GetXsize() > columnWidths[iColumn] ) {
               columnWidths[iColumn] = entrytex.GetXsize();
            }
            iColumn++;
            iColumn %= fNColumns;
         }
      }
      double totalWidth = 0.0;
      for(int i=0; i<fNColumns; i++) totalWidth += columnWidths[i];
      if(fNColumns > 1) totalWidth /= (1.0-fMargin-fColumnSeparation);
      else totalWidth /= (1.0 - fMargin);
      for(int i=0; i<fNColumns; i++) {
         columnWidths[i] = columnWidths[i]/totalWidth*(x2-x1) + margin;
      }
   }

   Double_t ytext = y2 + 0.5*yspace;  // y-location of 0th entry

   // iterate over and paint all the TLegendEntries
   TIter next(fPrimitives);
   TLegendEntry *entry;
   Int_t iColumn = 0;
   while (( entry = (TLegendEntry *)next() )) {
      if(iColumn == 0) ytext -= yspace;

      // Draw Label in Latexmargin

      Short_t talign = entry->GetTextAlign();
      Float_t tangle = entry->GetTextAngle();
      Color_t tcolor = entry->GetTextColor();
      Style_t tfont  = entry->GetTextFont();
      Size_t  tsize  = entry->GetTextSize();
      // if the user hasn't set a parameter, then set it to the TLegend value
      if (talign == 0) entry->SetTextAlign(GetTextAlign());
      if (tangle == 0) entry->SetTextAngle(GetTextAngle());
      if (tcolor == 0) entry->SetTextColor(GetTextColor());
      if (tfont  == 0) entry->SetTextFont(GetTextFont());
      if (tsize  == 0) entry->SetTextSize(GetTextSize());
      // set x,y according to the requested alignment
      Double_t x=0,y=0;
      Int_t halign = entry->GetTextAlign()/10;
      Double_t entrymargin = margin;
      // for the header the margin is near zero
      TString opt = entry->GetOption();
      opt.ToLower();
      x1 = fX1NDC;
      x2 = fX2NDC;
      if ( opt.Contains("h") ) entrymargin = margin/10.;
      else if (fNColumns > 1) {
         for(int i=0; i<iColumn; i++) x1 += columnWidths[i] + fColumnSeparation*(fX2NDC-fX1NDC)/(fNColumns-1);
         x2 = x1 + columnWidths[iColumn];
         iColumn++;
         iColumn %= fNColumns;
      }
      if (halign == 1) x = x1 + entrymargin;
      if (halign == 2) x = 0.5*( (x1+entrymargin) + x2 );
      if (halign == 3) x = x2 - entrymargin/10.;
      Int_t valign = entry->GetTextAlign()%10;
      if (valign == 1) y = ytext - (1. - fEntrySeparation)* yspace/2.;
      if (valign == 2) y = ytext;
      if (valign == 3) y = ytext + (1. - fEntrySeparation)* yspace/2.;
      //
      TLatex entrytex( x, y, entry->GetLabel() );
      entrytex.SetNDC();
      entry->TAttText::Copy(entrytex);
      entrytex.Paint();
      // reset attributes back to their original values
      entry->SetTextAlign(talign);
      entry->SetTextAngle(tangle);
      entry->SetTextColor(tcolor);
      entry->SetTextFont(tfont);
      entry->SetTextSize(tsize);

      // define x,y as the center of the symbol for this entry
      Double_t xsym = x1 + margin/2.;
      Double_t ysym = ytext;

      TObject *eobj = entry->GetObject();

      // Draw fill pattern (in a box)

      if ( opt.Contains("f")) {
         if (eobj && eobj->InheritsFrom(TAttFill::Class())) {
            dynamic_cast<TAttFill*>(eobj)->Copy(*entry);
         }

         // box total height is yspace*0.7
         entry->TAttFill::Modify();
         Double_t xf[4],yf[4];
         xf[0] = xsym - boxw;
         yf[0] = ysym - yspace*0.35;
         xf[1] = xsym + boxw;
         yf[1] = yf[0];
         xf[2] = xf[1];
         yf[2] = ysym + yspace*0.35;
         xf[3] = xf[0];
         yf[3] = yf[2];
         for (Int_t i=0;i<4;i++) {
            xf[i] = gPad->GetX1() + xf[i]*(gPad->GetX2()-gPad->GetX1());
            yf[i] = gPad->GetY1() + yf[i]*(gPad->GetY2()-gPad->GetY1());
         }
         gPad->PaintFillArea(4,xf,yf);
      }

      // Draw line

      if ( opt.Contains("l") || opt.Contains("f")) {

         if (eobj && eobj->InheritsFrom(TAttLine::Class())) {
            dynamic_cast<TAttLine*>(eobj)->Copy(*entry);
         }
         // line total length (in x) is margin*0.8
         TLine entryline( xsym - boxw, ysym, xsym + boxw, ysym );
         entryline.SetBit(TLine::kLineNDC);
         entry->TAttLine::Copy(entryline);
         // if the entry is filled, then surround the box with the line instead
         if ( opt.Contains("f") && !opt.Contains("l")) {
            // box total height is yspace*0.7
            boxwidth = yspace*
               (gPad->GetX2()-gPad->GetX1())/(gPad->GetY2()-gPad->GetY1());
            if ( boxwidth > margin ) boxwidth = margin;

            entryline.PaintLineNDC( xsym - boxw, ysym + yspace*0.35,
                                 xsym + boxw, ysym + yspace*0.35);
            entryline.PaintLineNDC( xsym - boxw, ysym - yspace*0.35,
                                 xsym + boxw, ysym - yspace*0.35);
            entryline.PaintLineNDC( xsym + boxw, ysym - yspace*0.35,
                                 xsym + boxw, ysym + yspace*0.35);
            entryline.PaintLineNDC( xsym - boxw, ysym - yspace*0.35,
                                 xsym - boxw, ysym + yspace*0.35);
         } else {
            entryline.Paint();
            if (opt.Contains("e")) {
               entryline.PaintLineNDC( xsym, ysym - yspace*0.30,
                                       xsym, ysym + yspace*0.30);
            }
         }
      }

      // Draw Polymarker

      if ( opt.Contains("p")) {

         if (eobj && eobj->InheritsFrom(TAttMarker::Class())) {
            dynamic_cast<TAttMarker*>(eobj)->Copy(*entry);
         }
         TMarker entrymarker( xsym, ysym, 0 );
         entrymarker.SetNDC();
         entry->TAttMarker::Copy(entrymarker);
         entrymarker.Paint();
      }
   }

   SetTextSize(save_textsize);
   delete [] columnWidths;
}


//______________________________________________________________________________
void TLegend::Print( Option_t* option ) const
{
   /* Begin_Html
   Dump this TLegend and its contents.
   End_Html */

   TPave::Print( option );
   if (fPrimitives) fPrimitives->Print();
}


//______________________________________________________________________________
void TLegend::RecursiveRemove(TObject *obj)
{
   /* Begin_Html
   Reset the legend entries pointing to "obj".
   End_Html */

   TIter next(fPrimitives);
   TLegendEntry *entry;
   while (( entry = (TLegendEntry *)next() )) {
      if (entry->GetObject() == obj) entry->SetObject((TObject*)0);
   }
}


//______________________________________________________________________________
void TLegend::SavePrimitive(ostream &out, Option_t* )
{
   /* Begin_Html
   Save this legend as C++ statements on output stream out
   to be used with the SaveAs .C option.
   End_Html */

   out << "   " << endl;
   char quote = '"';
   if ( gROOT->ClassSaved( TLegend::Class() ) ) {
      out << "   ";
   } else {
      out << "   TLegend *";
   }
   // note, we can always use NULL header, since its included in primitives
   out << "leg = new TLegend("<<GetX1NDC()<<","<<GetY1NDC()<<","
       <<GetX2NDC()<<","<<GetY2NDC()<<","
       << "NULL" << "," <<quote<< fOption <<quote<<");" << endl;
   if (fBorderSize != 4) {
      out<<"   leg->SetBorderSize("<<fBorderSize<<");"<<endl;
   }
   SaveTextAttributes(out,"leg",12,0,1,42,0);
   SaveLineAttributes(out,"leg",-1,-1,-1);
   SaveFillAttributes(out,"leg",-1,-1);
   if ( fPrimitives ) {
      TIter next(fPrimitives);
      TLegendEntry *entry;
      while (( entry = (TLegendEntry *)next() )) entry->SaveEntry(out,"leg");
   }
   out << "   leg->Draw();"<<endl;
}


//______________________________________________________________________________
void TLegend::SetEntryLabel( const char* label )
{
   /* Begin_Html
   Edit the label of the entry pointed to by the mouse.
   End_Html */
   
   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( entry ) entry->SetLabel( label );
}


//______________________________________________________________________________
void TLegend::SetEntryOption( Option_t* option )
{
   /* Begin_Html
   Edit the option of the entry pointed to by the mouse.
   End_Html */

   TLegendEntry* entry = GetEntry();   // get entry pointed by the mouse
   if ( entry ) entry->SetOption( option );
}


//______________________________________________________________________________
void TLegend::SetHeader( const char *header )
{
   /* Begin_Html
   Sets the header, which is the "title" that appears at the top of the legend.
   End_Html */

   if ( !fPrimitives ) fPrimitives = new TList;
   TIter next(fPrimitives);
   TLegendEntry *first;   // header is always the first entry
   if ((  first = (TLegendEntry*)next() )) {
      TString opt = first->GetOption();
      opt.ToLower();
      if ( opt.Contains("h") ) {
         first->SetLabel(header);
         return;
      }
   }
   first = new TLegendEntry( 0, header, "h" );
   first->SetTextAlign(0);
   first->SetTextAngle(0);
   first->SetTextColor(0);
   first->SetTextFont(GetTextFont()); // default font is TLegend font for the header
   first->SetTextSize(0);
   fPrimitives->AddFirst((TObject*)first);
}
