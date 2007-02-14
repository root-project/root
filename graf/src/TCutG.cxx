// @(#)root/graf:$Name:  $:$Id: TCutG.cxx,v 1.21 2007/01/15 16:10:10 brun Exp $
// Author: Rene Brun   16/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCutG                                                                //
//                                                                      //
//  A Graphical cut.                                                    //
//  A TCutG object defines a closed polygon in a x,y plot.              //
//  It can be created via the graphics editor option "CutG"             //
//  or directly by invoking its constructor.                            //
//  To create a TCutG via the graphics editor, use the left button      //
//  to select the points building the contour of the cut. Click on      //
//  the right button to close the TCutG.                                //
//  When it is created via the graphics editor, the TCutG object        //
//  is named "CUTG". It is recommended to immediatly change the name    //
//  by using the context menu item "SetName".                           //                                                                      //
//   When the graphics editor is used, the names of the variables X,Y   //
//   are automatically taken from the current pad title.                //
//  Example:                                                            //
//  Assume a TTree object T and:                                        //
//     Root > T.Draw("abs(fMomemtum)%fEtot")                            //
//  the TCutG members fVarX, fVary will be set to:                      //
//     fVarx = fEtot                                                    //
//     fVary = abs(fMomemtum)                                           //
//                                                                      //
//  A graphical cut can be used in a TTree selection expression:        //
//    Root > T.Draw("fEtot","cutg1")                                    //
//    where "cutg1" is the name of an existing graphical cut.           //
//                                                                      //
//  Note that, as shown in the example above, a graphical cut may be    //
//  used in a selection expression when drawing TTrees expressions      //
//  of 1-d, 2-d or 3-dimensions.                                        //
//  The expressions used in TTree::Draw can reference the variables     //
//  in the fVarX, fVarY of the graphical cut plus other variables.      //
//                                                                      //
//  When the TCutG object is created, it is added to the list of special//
//  objects in the main TROOT object pointed by gROOT. To retrieve a    //
//  pointer to this object from the code or command line, do:           //
//      TCutG *mycutg;                                                  //
//      mycutg = (TCutG*)gROOT->GetListOfSpecials()->FindObject("CUTG") //
//      mycutg->SetName("mycutg");                                      //
//                                                                      //
//  Example of use of a TCutG in TTree::Draw:                           //
//       tree.Draw("x:y","mycutg && z>0 %% sqrt(x)>1")                  //
//                                                                      //
//  A Graphical cut may be drawn via TGraph::Draw.                      //
//  It can be edited like a normal TGraph.                              //
//                                                                      //
//  A Graphical cut may be saved to a file via TCutG::Write.            //                                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TCutG.h"
#include "TVirtualPad.h"
#include "TPaveText.h"
#include "TH2.h"
#include "TClass.h"
#include "TMath.h"

ClassImp(TCutG)

//______________________________________________________________________________
TCutG::TCutG() : TGraph()
{
   // TCutG default constructor.

   fObjectX  = 0;
   fObjectY  = 0;
}

//______________________________________________________________________________
TCutG::TCutG(const TCutG &cutg)
      :TGraph(cutg)
{
   // TCutG copy constructor
   
   fVarX    = cutg.fVarX;
   fVarY    = cutg.fVarY;
   fObjectX = cutg.fObjectX;
   fObjectY = cutg.fObjectY;
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n)
      :TGraph(n)
{
   // TCutG normal constructor.

   fObjectX  = 0;
   fObjectY  = 0;
   SetName(name);
   delete gROOT->GetListOfSpecials()->FindObject(name);
   gROOT->GetListOfSpecials()->Add(this);

// Take name of cut variables from pad title if title contains ":"
   if (gPad) {
      TPaveText *ptitle = (TPaveText*)gPad->FindObject("title");
      if (!ptitle) return;
      TText *ttitle = ptitle->GetLineWith(":");
      if (!ttitle) ttitle = ptitle->GetLineWith("{");
      if (!ttitle) ttitle = ptitle->GetLine(0);
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (col) {
         *col = 0;
         col++;
         char *brak = strstr(col," {");
         if (brak) *brak = 0;
         fVarY = vars;
         fVarX = col;
      } else {
         char *brak = strstr(vars," {");
         if (brak) *brak = 0;
         fVarX = vars;
      }        
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n, const Float_t *x, const Float_t *y)
      :TGraph(n,x,y)
{
   // TCutG normal constructor.

   fObjectX  = 0;
   fObjectY  = 0;
   SetName(name);
   delete gROOT->GetListOfSpecials()->FindObject(name);
   gROOT->GetListOfSpecials()->Add(this);

// Take name of cut variables from pad title if title contains ":"
   if (gPad) {
      TPaveText *ptitle = (TPaveText*)gPad->FindObject("title");
      if (!ptitle) return;
      TText *ttitle = ptitle->GetLineWith(":");
      if (!ttitle) ttitle = ptitle->GetLineWith("{");
      if (!ttitle) ttitle = ptitle->GetLine(0);
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (col) {
         *col = 0;
         col++;
         char *brak = strstr(col," {");
         if (brak) *brak = 0;
         fVarY = vars;
         fVarX = col;
      } else {
         char *brak = strstr(vars," {");
         if (brak) *brak = 0;
         fVarX = vars;
      }        
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n, const Double_t *x, const Double_t *y)
      :TGraph(n,x,y)
{
   // TCutG normal constructor.

   fObjectX  = 0;
   fObjectY  = 0;
   SetName(name);
   delete gROOT->GetListOfSpecials()->FindObject(name);
   gROOT->GetListOfSpecials()->Add(this);

// Take name of cut variables from pad title if title contains ":"
   if (gPad) {
      TPaveText *ptitle = (TPaveText*)gPad->FindObject("title");
      if (!ptitle) return;
      TText *ttitle = ptitle->GetLineWith(":");
      if (!ttitle) ttitle = ptitle->GetLineWith("{");
      if (!ttitle) ttitle = ptitle->GetLine(0);
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (col) {
         *col = 0;
         col++;
         char *brak = strstr(col," {");
         if (brak) *brak = 0;
         fVarY = vars;
         fVarX = col;
      } else {
         char *brak = strstr(vars," {");
         if (brak) *brak = 0;
         fVarX = vars;
      }        
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::~TCutG()
{
   // TCutG destructor.

   delete fObjectX;
   delete fObjectY;
   gROOT->GetListOfSpecials()->Remove(this);
}

//______________________________________________________________________________
Double_t TCutG::Integral(TH2 *h, Option_t *option) const
{
   // Compute the integral of 2-d histogram h for all bins inside the cut
   // if option "width" is specified, the integral is the sum of
   // the bin contents multiplied by the bin width in x and in y.

   if (!h) return 0;
   Int_t n = GetN();
   Double_t xmin= 1e200;
   Double_t xmax = -xmin;
   Double_t ymin = xmin;
   Double_t ymax = xmax;
   for (Int_t i=0;i<n;i++) {
      if (fX[i] < xmin) xmin = fX[i];
      if (fX[i] > xmax) xmax = fX[i];
      if (fY[i] < ymin) ymin = fY[i];
      if (fY[i] > ymax) ymax = fY[i];
   }
   TAxis *xaxis = h->GetXaxis();
   TAxis *yaxis = h->GetYaxis();
   Int_t binx1 = xaxis->FindBin(xmin);
   Int_t binx2 = xaxis->FindBin(xmax);
   Int_t biny1 = yaxis->FindBin(ymin);
   Int_t biny2 = yaxis->FindBin(ymax);
   Int_t nbinsx = h->GetNbinsX();
   Stat_t integral = 0;

   // Loop on bins for which the bin center is in the cut 
   TString opt = option;
   opt.ToLower();
   Bool_t width = kFALSE;
   if (opt.Contains("width")) width = kTRUE;
   Int_t bin, binx, biny;
   for (biny=biny1;biny<=biny2;biny++) {
      Double_t y = yaxis->GetBinCenter(biny);
      for (binx=binx1;binx<=binx2;binx++) {
         Double_t x = xaxis->GetBinCenter(binx);
         if (!IsInside(x,y)) continue;
         bin = binx +(nbinsx+2)*biny;
         if (width) integral += h->GetBinContent(bin)*xaxis->GetBinWidth(binx)*yaxis->GetBinWidth(biny);
         else       integral += h->GetBinContent(bin);
      }
   }
   return integral;
}

//______________________________________________________________________________
Int_t TCutG::IsInside(Double_t x, Double_t y) const
{
   //         Function which returns 1 if point x,y lies inside the
   //              polygon defined by the graph points
   //                                0 otherwise
   //
   //     The loop is executed with the end-point coordinates of a
   //     line segment (X1,Y1)-(X2,Y2) and the Y-coordinate of a
   //     horizontal line.
   //     The counter inter is incremented if the line (X1,Y1)-(X2,Y2)
   //     intersects the horizontal line.
   //     In this case XINT is set to the X-coordinate of the
   //     intersection point.
   //     If inter is an odd number, then the point x,y is within
   //     the polygon.
   //
   //         This routine is based on an original algorithm
   //         developed by R.Nierhaus.

   return (Int_t)TMath::IsInside(x,y,fNpoints,fX,fY);
}

//______________________________________________________________________________
void TCutG::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out.

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TCutG::Class())) {
      out<<"   ";
   } else {
      out<<"   TCutG *";
   }
   out<<"cutg = new TCutG("<<quote<<GetName()<<quote<<","<<fNpoints<<");"<<endl;
   out<<"   cutg->SetVarX("<<quote<<GetVarX()<<quote<<");"<<endl;
   out<<"   cutg->SetVarY("<<quote<<GetVarY()<<quote<<");"<<endl;
   out<<"   cutg->SetTitle("<<quote<<GetTitle()<<quote<<");"<<endl;

   SaveFillAttributes(out,"cutg",0,1001);
   SaveLineAttributes(out,"cutg",1,1,1);
   SaveMarkerAttributes(out,"cutg",1,1,1);

   for (Int_t i=0;i<fNpoints;i++) {
      out<<"   cutg->SetPoint("<<i<<","<<fX[i]<<","<<fY[i]<<");"<<endl;
   }
   out<<"   cutg->Draw("
      <<quote<<option<<quote<<");"<<endl;
}

//______________________________________________________________________________
void TCutG::SetVarX(const char *varx)
{
   // Set X variable.

   fVarX = varx;
   delete fObjectX;
   fObjectX = 0;
}

//______________________________________________________________________________
void TCutG::SetVarY(const char *vary)
{
   // Set Y variable.

   fVarY = vary;
   delete fObjectY;
   fObjectY = 0;
}


//______________________________________________________________________________
void TCutG::Streamer(TBuffer &R__b)
{
   // Stream an object of class TCutG.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TCutG::Class(), this);
      gROOT->GetListOfSpecials()->Add(this);
   } else {
      R__b.WriteClassBuffer(TCutG::Class(), this);
   }
}
