// @(#)root/graf:$Name:  $:$Id: TCutG.cxx,v 1.9 2002/01/23 17:52:48 rdm Exp $
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
//  The TTree expressions may or may not reference the same variables   //
//  than in the fVarX, fVarY of the graphical cut.                      //
//                                                                      //
//  When the TCutG object is created, it is added to the list of special//
//  objects in the main TROOT object pointed by gROOT. To retrieve a    //
//  pointer to this object from the code or command line, do:           //
//      TCutG *mycutg;                                                  //
//      mycutg = (TCutG*)gROOT->GetListOfSpecials()->FindObject("CUTG") //
//      mycutg->SetName("mycutg");                                      //
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

ClassImp(TCutG)

//______________________________________________________________________________
TCutG::TCutG() : TGraph()
{

   fObjectX  = 0;
   fObjectY  = 0;
   gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n)
      :TGraph(n)
{
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
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (!col) return;
      *col = 0;
      col++;
      char *brak = strstr(col," {");
      if (brak) *brak = 0;
      fVarY = vars;
      fVarX = col;
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n, const Float_t *x, const Float_t *y)
      :TGraph(n,x,y)
{
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
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (!col) return;
      *col = 0;
      col++;
      char *brak = strstr(col," {");
      if (brak) *brak = 0;
      fVarY = vars;
      fVarX = col;
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::TCutG(const char *name, Int_t n, const Double_t *x, const Double_t *y)
      :TGraph(n,x,y)
{
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
      if (!ttitle) return;
      const char *title = ttitle->GetTitle();
      Int_t nch = strlen(title);
      char *vars = new char[nch+1];
      strcpy(vars,title);
      char *col = strstr(vars,":");
      if (!col) return;
      *col = 0;
      col++;
      char *brak = strstr(col," {");
      if (brak) *brak = 0;
      fVarY = vars;
      fVarX = col;
      delete [] vars;
   }
}

//______________________________________________________________________________
TCutG::~TCutG()
{

   delete fObjectX;
   delete fObjectY;
   gROOT->GetListOfSpecials()->Remove(this);
}

//______________________________________________________________________________
Int_t TCutG::IsInside(Double_t x, Double_t y) const
{
//*.         Function which returns 1 if point x,y lies inside the
//*.              polygon defined by the graph points
//*.                                0 otherwise
//*.
//*.     The loop is executed with the end-point coordinates of a
//*.     line segment (X1,Y1)-(X2,Y2) and the Y-coordinate of a
//*.     horizontal line.
//*.     The counter inter is incremented if the line (X1,Y1)-(X2,Y2)
//*.     intersects the horizontal line.
//*.     In this case XINT is set to the X-coordinate of the
//*.     intersection point.
//*.     If inter is an odd number, then the point x,y is within
//*.     the polygon.
//*.
//*.         This routine is based on an original algorithm
//*.         developed by R.Nierhaus.
//*.

   Double_t xint;
   Int_t i;
   Int_t inter = 0;
   for (i=0;i<fNpoints-1;i++) {
      if (fY[i] == fY[i+1]) continue;
      if (y <= fY[i] && y <= fY[i+1]) continue;
      if (fY[i] < y && fY[i+1] < y) continue;
      xint = fX[i] + (y-fY[i])*(fX[i+1]-fX[i])/(fY[i+1]-fY[i]);
      if (x < xint) inter++;
   }
   if (inter%2) return 1;
   return 0;
}

//______________________________________________________________________________
void TCutG::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

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
   fVarX = varx;
   delete fObjectX;
   fObjectX = 0;
}

//______________________________________________________________________________
void TCutG::SetVarY(const char *vary)
{
   fVarY = vary;
   delete fObjectY;
   fObjectY = 0;
}


//______________________________________________________________________________
void TCutG::Streamer(TBuffer &R__b)
{
   // Stream an object of class TCutG.

   if (R__b.IsReading()) {
      TCutG::Class()->ReadBuffer(R__b, this);
      gROOT->GetListOfSpecials()->Add(this);
   } else {
      TCutG::Class()->WriteBuffer(R__b, this);
   }
}
