// @(#)root/graf:$Name:  $:$Id: TPavesText.cxx,v 1.4 2002/01/23 17:52:49 rdm Exp $
// Author: Rene Brun   19/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TPavesText.h"
#include "TVirtualPad.h"
#include "TError.h"

ClassImp(TPavesText)

//______________________________________________________________________________
//  A PavesText is a PaveText (see TPaveText) with several stacked paves.
//Begin_Html
/*
<img src="gif/pavestext.gif">
*/
//End_Html
//

//______________________________________________________________________________
TPavesText::TPavesText(): TPaveText()
{
//*-*-*-*-*-*-*-*-*-*-*pavestext default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================

   fNpaves = 5;
}

//______________________________________________________________________________
TPavesText::TPavesText(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Int_t npaves,Option_t *option)
           :TPaveText(x1,y1,x2,y2,option)
{
//*-*-*-*-*-*-*-*-*-*-*pavestext normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
//
// The PavesText is by default defined bith bordersize=1 and option ="br".
//

   fNpaves = npaves;
   SetBorderSize(1);
}

//______________________________________________________________________________
TPavesText::~TPavesText()
{
//*-*-*-*-*-*-*-*-*-*-*pavestext default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

}

//______________________________________________________________________________
TPavesText::TPavesText(const TPavesText &pavestext)
{
   ((TPavesText&)pavestext).Copy(*this);
}

//______________________________________________________________________________
void TPavesText::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this pavestext to pavestext*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================

   TPaveText::Copy(obj);
   ((TPavesText&)obj).fNpaves = fNpaves;
}

//______________________________________________________________________________
void TPavesText::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this pavestext with its current attributes*-*-*-*-*
//*-*                  ===============================================

   AppendPad(option);

}

//______________________________________________________________________________
void TPavesText::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this pavestext with its current attributes*-*-*-*
//*-*                  ================================================


//*-* Draw the fNpaves-1 stacked paves
//*-* The spacing between paves is set to 3 times the bordersize
   Int_t bordersize = GetBorderSize();
   const char *opt = GetOption();
   Double_t signx, signy;
   if (strstr(opt,"l")) signx = -1;
   else                 signx =  1;
   if (strstr(opt,"b")) signy = -1;
   else                 signy =  1;
   Double_t dx = 3*signx*(gPad->PixeltoX(bordersize) - gPad->PixeltoX(0));
   Double_t dy = 3*signy*(gPad->PixeltoY(bordersize) - gPad->PixeltoY(0));

   TPave::ConvertNDCtoPad();

   for (Int_t ipave=fNpaves;ipave>1;ipave--) {
      Double_t x1 = fX1 + dx*Double_t(ipave-1);
      Double_t y1 = fY1 - dy*Double_t(ipave-1);
      Double_t x2 = fX2 + dx*Double_t(ipave-1);
      Double_t y2 = fY2 - dy*Double_t(ipave-1);
      TPave::PaintPave(x1,y1,x2,y2,bordersize,option);
   }

//*-*- Draw the top pavetext
   TPaveText::Paint(option);
}

//______________________________________________________________________________
void TPavesText::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   if (!strcmp(GetName(),"stats")) return;
   if (!strcmp(GetName(),"title")) return;
   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPavesText::Class())) {
       out<<"   ";
   } else {
       out<<"   TPavesText *";
   }
   out<<"pst = new TPavesText("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<fNpaves<<","<<quote<<fOption<<quote<<");"<<endl;

   if (strcmp(GetName(),"TPave")) {
      out<<"   pst->SetName("<<quote<<GetName()<<quote<<");"<<endl;
   }
   if (fLabel.Length() > 0) {
      out<<"   pst->SetLabel("<<quote<<fLabel<<quote<<");"<<endl;
   }
   if (fBorderSize != 4) {
      out<<"   pst->SetBorderSize("<<fBorderSize<<");"<<endl;
   }
   SaveFillAttributes(out,"pst",0,1001);
   SaveLineAttributes(out,"pst",1,1,1);
   SaveTextAttributes(out,"pst",22,0,1,62,0);
   TPaveText::SaveLines(out,"pst");
   out<<"   pst->Draw();"<<endl;
}
