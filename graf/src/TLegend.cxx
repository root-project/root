// @(#)root/graf:$Name:  $:$Id: TLegend.cxx,v 1.5 2000/09/08 16:05:21 rdm Exp $
// Author: Matthew.Adam.Dobbs   06/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include <fstream.h>
#include <stdio.h>
#include <iostream.h>

ClassImp(TLegend)

//____________________________________________________________________________
// TLegend   Matthew.Adam.Dobbs@Cern.CH, September 1999
// Legend of markers/lines/boxes to represent objects with marker/line/fill
//   attributes
//   (the methods employed are very similar to those in TPaveText class)
//

//____________________________________________________________________________
TLegend::TLegend(): TPave(), TAttText()
{
  // TPadLegend do-nothing default constructor
  fPrimitives = 0;
  SetDefaults();
}

//____________________________________________________________________________
TLegend::TLegend( Double_t x1, Double_t y1,Double_t x2, Double_t y2, const char *header, Option_t *option)
        :TPave(x1,y1,x2,y2,4,option), TAttText(12,0,1,42,0)
{
  //___________________________________
  // TLegend normal Contructor
  // A TLegend is a Pave with several TLegendEntry(s)
  // The pave is defined with default coords, bordersize and option
  // x1,y1,x2,y2 are the coordinates of the Legend in the current pad
  // (in NDC by default!)
  // text is left adjusted and vertically centered = 12
  //      Angle=0 (degrees), color=1 (black)
  //      helvetica-medium-r-normal scalable font = 42
  //      (will use bold = 62 for header)
  //      size =0 (calculate this later when number of entries is known)
  // header is the "title" that will be displayed at the top of the legend
  //   it is treated like a regular entry and supports TLatex. The default
  //   is no header (header = 0).
  // Options are the same as for TPave Default = "brNDC"
  //
  // Here's an example of a Legend created with TLegend
  //Begin_Html
  /*
    <IMG SRC="gif/example_legend.gif">
  */
  //End_Html
  //
  // The Legend part of this plot was created as follows:
  //
  //    leg = new TLegend(0.4,0.6,0.89,0.89);
  //    leg->AddEntry(fun1,"One Theory","l");
  //    leg->AddEntry(fun3,"Another Theory","f");
  //    leg->AddEntry(gr,"The Data","p");
  //    leg->Draw();
  //    // oops we forgot the blue line... add it after
  //    leg->AddEntry(fun2,"#sqrt{2#pi} P_{T} (#gamma)  latex formula","f");
  //    // and add a header (or "title") for the legend
  //    leg->SetHeader("The Legend Title");
  //    leg->Draw();
  //
  // where fun1,fun2,fun3 and gr are pre-existing functions and graphs
  //
  // You can edit the TLegend by right-clicking on it.
  //
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
}

//____________________________________________________________________________
TLegend::TLegend( const TLegend &legend )
{
  // copy constuctor
  ((TLegend&)legend).Copy(*this);
}

//____________________________________________________________________________
TLegend::~TLegend()
{
  // TLegend default destructor
  if (fPrimitives) fPrimitives->Delete();
  delete fPrimitives;
  fPrimitives = 0;
}

//____________________________________________________________________________
TLegendEntry *TLegend::AddEntry(TObject *obj, const char *label, Option_t *option)
{
  // Add a new entry to this legend
  // obj is the object to be represented
  // label is the text you wish to associate with obj in the legend
  // Options are:
  //    L draw line associated w/ TAttLine if obj inherits from TAttLine
  //    P draw polymarker assoc. w/ TAttMarker if obj inherits from TAttMarker
  //    F draw a box with fill associated w/ TAttFill if obj inherits TAttFill
  //
  TLegendEntry *newentry = new TLegendEntry( obj, label, option );
  if ( !fPrimitives ) fPrimitives = new TList;
  fPrimitives->Add(newentry);
  return newentry;
}

//____________________________________________________________________________
TLegendEntry *TLegend::AddEntry(const char *name, const char *label, Option_t *option)
{
  // Add a new entry to this legend
  // name is the name of an object in the pad to be represented
  // label is the text you wish to associate with obj in the legend
  // Options are:
  //    L draw line associated w/ TAttLine if obj inherits from TAttLine
  //    P draw polymarker assoc. w/ TAttMarker if obj inherits from TAttMarker
  //    F draw a box with fill associated w/ TAttFill if obj inherits TAttFill
  //
  TObject *obj = gPad->FindObject(name);
  return AddEntry( obj, label, option );
}

//____________________________________________________________________________
void TLegend::Clear( Option_t *)
{
  // Clear all entries in this legend --- including the header!
  if (!fPrimitives) return;
  fPrimitives->Delete();
}

//____________________________________________________________________________
void TLegend::Copy( TObject &obj )
{
  // copy this legend into obj
  TPave::Copy(obj);
  TAttText::Copy((TLegend&)obj);
  ((TLegend&)obj).fEntrySeparation = fEntrySeparation;
  ((TLegend&)obj).fMargin = fMargin;
}

//____________________________________________________________________________
void TLegend::DeleteEntry()
{
  // Delete entry at the mouse position
  if ( !fPrimitives ) return;
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( !entry ) return;
  fPrimitives->Remove(entry);
  delete entry;
}

//____________________________________________________________________________
void TLegend::Draw( Option_t *option )
{
  // Draw this legend with its current attributes
  AppendPad(option);
}

//____________________________________________________________________________
void TLegend::EditEntryAttFill()
{
  // Edit the fill attributes for the entry pointed to be the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( !entry ) return;
  gROOT->SetSelectedPrimitive( entry );
  entry->SetFillAttributes();
}

//____________________________________________________________________________
void TLegend::EditEntryAttLine()
{
  // Edit the line attributes for the entry pointed to be the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( !entry ) return;
  gROOT->SetSelectedPrimitive( entry );
  entry->SetLineAttributes();
}

//____________________________________________________________________________
void TLegend::EditEntryAttMarker()
{
  // Edit the marker attributes for the entry pointed to be the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( !entry ) return;
  gROOT->SetSelectedPrimitive( entry );
  entry->SetMarkerAttributes();
}

//____________________________________________________________________________
void TLegend::EditEntryAttText()
{
  // Edit the text attributes for the entry pointed to be the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( !entry ) return;
  gROOT->SetSelectedPrimitive( entry );
  entry->SetTextAttributes();
}

//____________________________________________________________________________
TLegendEntry *TLegend::GetEntry() const
{
  // Get entry pointed to by the mouse
  // This method is mostly a tool for other methods inside this class
  Int_t nEntries = 0;
  if ( fPrimitives ) nEntries = fPrimitives->GetSize();
  if ( nEntries == 0 ) return 0;

  Double_t ymouse = gPad->AbsPixeltoY(gPad->GetEventY());
  Double_t yspace = (fY2 - fY1)/nEntries;

  Double_t ybottomOfEntry = fY2;  // y-location of bottom of 0th entry
  TIter next(fPrimitives);
  TLegendEntry *entry;
  while (( entry = (TLegendEntry *)next() )) {
    ybottomOfEntry -= yspace;
    if ( ybottomOfEntry < ymouse ) return entry;
  }
  return 0;
}

//____________________________________________________________________________
const char *TLegend::GetHeader() const
{
  // returns the header, which is the title that appears at the top
  //  of the legend
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

//____________________________________________________________________________
void TLegend::InsertEntry( const char* objectName, const char* label, Option_t* option)
{
  // Add a new entry before the entry at the mouse position

  TLegendEntry* beforeEntry = GetEntry();   // get entry pointed to be mouse
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

//____________________________________________________________________________
void TLegend::Paint( Option_t* option )
{
  // Paint this legend with its current attributes
  TPave::ConvertNDCtoPad();
  TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);
  PaintPrimitives();
}

//____________________________________________________________________________
void TLegend::PaintPrimitives()
{
  // Paint the entries (list of primitives) for this legend
  //
  // NOTE: if we want an     Int_t mode
  //       it can be added later... but I understand whyaas

  Int_t nEntries = 0;
  if ( fPrimitives ) nEntries = fPrimitives->GetSize();
  if ( nEntries == 0 ) return;

  // Evaluate text size as a function of the number of entries
  //  taking into account their real size after drawing latex
  // Note: in pixel coords y1 > y2=0, but x2 > x1=0
  //       in NDC          y2 > y1,   and x2 > x1
  //
  Double_t margin = fMargin*( fX2 - fX1 );
  Double_t yspace = (fY2 - fY1)/nEntries;
  Double_t textsize = GetTextSize();
  Double_t save_textsize = textsize;

  if ( textsize == 0 ) {
    textsize = ( 1. - fEntrySeparation ) * yspace;

    // find the max width and height (in pad coords) of one latex entry label
    Double_t maxentrywidth = 0, maxentryheight = 0;
    TIter nextsize(fPrimitives);
    TLegendEntry *entrysize;
    while (( entrysize = (TLegendEntry *)nextsize() )) {
      TLatex entrytex( 0, 0, entrysize->GetLabel() );
      entrytex.SetTextSize(textsize);
      if ( entrytex.GetYsize() > maxentryheight ) {
        maxentryheight = entrytex.GetYsize();
      }
      if ( entrytex.GetXsize() > maxentrywidth ) {
        maxentrywidth = entrytex.GetXsize();
      }
    }
    // make sure all labels fit in the allotted space
    Double_t tmpsize_h = textsize * ( textsize/maxentryheight );
    Double_t tmpsize_w = textsize * ( (fX2 - (fX1+margin))/maxentrywidth);
    textsize = TMath::Min( textsize, TMath::Min(tmpsize_h,tmpsize_w) );
    SetTextSize( textsize );
  }

  Double_t ytext = fY2 + 0.5*yspace;  // y-location of 0th entry

  // iterate over and paint all the TLegendEntries
  TIter next(fPrimitives);
  TLegendEntry *entry;
  while (( entry = (TLegendEntry *)next() )) {
    ytext -= yspace;

    // Draw Label in Latex

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
    if ( opt.Contains("h") ) entrymargin = margin/10.;
    if (halign == 1) x = fX1 + entrymargin;
    if (halign == 2) x = 0.5*( (fX1+entrymargin) + fX2 );
    if (halign == 3) x = fX2 - entrymargin/10.;
    Int_t valign = entry->GetTextAlign()%10;
    if (valign == 1) y = ytext - (1. - fEntrySeparation)* yspace/2.;
    if (valign == 2) y = ytext;
    if (valign == 3) y = ytext + (1. - fEntrySeparation)* yspace/2.;
    //
    TLatex entrytex( x, y, entry->GetLabel() );
    entry->TAttText::Copy(entrytex);
    entrytex.Paint();
    // reset attributes back to their original values
    entry->SetTextAlign(talign);
    entry->SetTextAngle(tangle);
    entry->SetTextColor(tcolor);
    entry->SetTextFont(tfont);
    entry->SetTextSize(tsize);

    // define x,y as the center of the symbol for this entry
    Double_t xsym = fX1 + margin/2.;
    Double_t ysym = ytext;

    if ( entry->GetObject() == 0 ) continue;

    // Draw fill pattern (in a box)

    if ( opt.Contains("f") && entry->GetObject()->InheritsFrom(TAttFill::Class())) {
      Color_t fcolor = entry->GetFillColor();
      Style_t fstyle = entry->GetFillStyle();
      char cmd[50];
//      if ( fcolor == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetFillColor();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetFillColor",cmd);
//      }
//      if ( fstyle == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetFillStyle();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetFillStyle",cmd);
//      }

      // box total height is yspace*0.7
      Double_t boxwidth = yspace*
        (gPad->GetX2()-gPad->GetX1())/(gPad->GetY2()-gPad->GetY1());
      if ( boxwidth > margin ) boxwidth = margin;
      TBox entrybox(xsym - boxwidth*0.35, ysym - yspace*0.35,
                    xsym + boxwidth*0.35, ysym + yspace*0.35);
      entry->TAttFill::Copy(entrybox);
      entrybox.Paint();
      entry->SetFillColor(fcolor);
      entry->SetFillStyle(fstyle);
    }

    // Draw line

    if ( ( opt.Contains("l") || opt.Contains("f") ) && entry->GetObject()->InheritsFrom(TAttLine::Class())) {

      Color_t lcolor = entry->GetLineColor();
      Style_t lstyle = entry->GetLineStyle();
      Width_t lwidth = entry->GetLineWidth();
      char cmd[50];
//      if ( lcolor == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetLineColor();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetLineColor",cmd);
//      }
//      if ( lstyle == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetLineStyle();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetLineStyle",cmd);
//      }
//      if ( lwidth == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetLineWidth();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetLineWidth",cmd);
//      }

      // line total length (in x) is margin*0.8
      TLine entryline( xsym - margin*0.4, ysym, xsym + margin*0.4, ysym );
      entry->TAttLine::Copy(entryline);
      // if the entry is filled, then surround the box with the line instead
      if ( opt.Contains("f") && !opt.Contains("l") && entry->GetObject()->InheritsFrom(TAttFill::Class())) {
        // box total height is yspace*0.7
        Double_t boxwidth = yspace*
          (gPad->GetX2()-gPad->GetX1())/(gPad->GetY2()-gPad->GetY1());

        entryline.PaintLine( xsym - boxwidth*0.35, ysym + yspace*0.35,
                             xsym + boxwidth*0.35, ysym + yspace*0.35);
        entryline.PaintLine( xsym - boxwidth*0.35, ysym - yspace*0.35,
                             xsym + boxwidth*0.35, ysym - yspace*0.35);
        entryline.PaintLine( xsym + boxwidth*0.35, ysym - yspace*0.35,
                             xsym + boxwidth*0.35, ysym + yspace*0.35);
        entryline.PaintLine( xsym - boxwidth*0.35, ysym - yspace*0.35,
                             xsym - boxwidth*0.35, ysym + yspace*0.35);
      } else { entryline.Paint(); }

      entry->SetLineColor(lcolor);
      entry->SetLineStyle(lstyle);
      entry->SetLineWidth(lwidth);
    }

    // Draw Polymarker

    if ( opt.Contains("p") && entry->GetObject()->InheritsFrom(TAttMarker::Class())) {

      Color_t mcolor = entry->GetMarkerColor();
      Style_t mstyle = entry->GetMarkerStyle();
      Size_t msize = entry->GetMarkerSize();
      char cmd[50];
//      if ( mcolor == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetMarkerColor();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetMarkerColor",cmd);
//      }
//      if ( mstyle == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetMarkerStyle();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetMarkerStyle",cmd);
//      }
//      if ( msize == 0 ) {
        sprintf(cmd,"((%s*)0x%lx)->GetMarkerSize();",
                entry->GetObject()->ClassName(),(Long_t)entry->GetObject());
        entry->Execute("SetMarkerSize",cmd);
//      }

      TMarker entrymarker( xsym, ysym, 0 );
      entry->TAttMarker::Copy(entrymarker);
      entry->SetMarkerColor(mcolor);
      entry->SetMarkerStyle(mstyle);
      entry->SetMarkerSize(msize);
      entrymarker.Paint();
    }
  }

  SetTextSize(save_textsize);
}

//____________________________________________________________________________
void TLegend::Print( Option_t* option ) const
{
  // dump this TLegend and its contents
  TPave::Print( option );
  if (fPrimitives) fPrimitives->Print();
}

//____________________________________________________________________________
void TLegend::SavePrimitive( ofstream &out, Option_t* )
{
  // Save this legend as C++ statements on output stream out
  //  to be used with the SaveAs .C option
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
  SaveTextAttributes(out,"leg",12,0,1,42,0);
  SaveLineAttributes(out,"leg",0,0,0);
  SaveFillAttributes(out,"leg",0,0);
  if ( fPrimitives ) {
    TIter next(fPrimitives);
    TLegendEntry *entry;
    while (( entry = (TLegendEntry *)next() )) entry->SaveEntry(out,"leg");
  }
  out << "   leg->Draw();"<<endl;
}

//____________________________________________________________________________
void TLegend::SetEntryLabel( const char* label )
{
  // edit the label of the entry pointed to by the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( entry ) entry->SetLabel( label );
}

//____________________________________________________________________________
void TLegend::SetEntryOption( Option_t* option )
{
  // edit the option of the entry pointed to by the mouse
  TLegendEntry* entry = GetEntry();   // get entry pointed to be mouse
  if ( entry ) entry->SetOption( option );
}

//____________________________________________________________________________
void TLegend::SetHeader( const char *header )
{
  // Sets the header, which is the "title" that appears at the top of the
  //  TLegend
  if ( !fPrimitives ) new TList;
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
  first->SetTextFont(62); // default font is 62 for the header
  first->SetTextSize(0);
  fPrimitives->AddFirst((TObject*)first);
}



