// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>

#include "TROOT.h"
#include "TArc.h"
#include "TVirtualPad.h"

ClassImp(TArc)

//______________________________________________________________________________
//
//  An arc is specified with the position of its centre, its radius
//  a minimum and maximum angle.
//  The attributes of the outline line are given via TAttLine
//  The attributes of the fill area are given via TAttFill
//

//______________________________________________________________________________
TArc::TArc(): TEllipse()
{
//*-*-*-*-*-*-*-*-*-*-*Arc  default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

}
//______________________________________________________________________________
TArc::TArc(Float_t x1, Float_t y1,Float_t r1,Float_t phimin,Float_t phimax)
      :TEllipse(x1,y1,r1,r1,phimin,phimax,0)
{
//*-*-*-*-*-*-*-*-*-*-*Arc  normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
//  x1,y1  : coordinates of centre of arc
//  r1     : arc radius
//  phimin : min and max angle in degrees (default is 0-->360)
//  phimax :
//
}

//______________________________________________________________________________
TArc::TArc(const TArc &arc)
{
   ((TArc&)arc).Copy(*this);
}

//______________________________________________________________________________
TArc::~TArc()
{
//*-*-*-*-*-*-*-*-*-*-*Arc  default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

}

//______________________________________________________________________________
void TArc::Copy(TObject &arc)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this arc to arc*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ====================
   TEllipse::Copy(arc);
}

//______________________________________________________________________________
void TArc::DrawArc(Float_t x1, Float_t y1,Float_t r1,Float_t phimin,Float_t phimax)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this arc with new coordinates*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================================
   TArc *newarc = new TArc(x1, y1, r1, phimin, phimax);
   TAttLine::Copy(*newarc);
   TAttFill::Copy(*newarc);
   newarc->AppendPad();
}

//______________________________________________________________________________
void TArc::SavePrimitive(ofstream &out, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out

   out<<"   "<<endl;
   if (gROOT->ClassSaved(TArc::Class())) {
       out<<"   ";
   } else {
       out<<"   TArc *";
   }
   out<<"arc = new TArc("<<fX1<<","<<fY1<<","<<fR1
      <<","<<fPhimin<<","<<fPhimax<<");"<<endl;

   SaveFillAttributes(out,"arc",0,1001);
   SaveLineAttributes(out,"arc",1,1,1);

   out<<"   arc->Draw();"<<endl;
}
