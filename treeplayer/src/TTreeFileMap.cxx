// @(#)root/treeplayer:$Name:  $:$Id: TTreeFileMap.cxx,v 1.117 2003/01/11 14:21:29 brun Exp $
// Author: Rene Brun   15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFileMap                                                         //
//                                                                      //
//Begin_Html
/*
<img src="gif/ttree_filemap.gif">
*/
//End_Html
//
//  =============================================================================

#include "TTreeFileMap.h"
#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TH1.h"
#include "TBox.h"
#include "TRegexp.h"

ClassImp(TTreeFileMap)

//______________________________________________________________________________
TTreeFileMap::TTreeFileMap()
{
// Default TreeFileMap constructor

   fTree   = 0;
   fFrame  = 0;
}

//______________________________________________________________________________
TTreeFileMap::TTreeFileMap(TTree *tree, const char *branches, Option_t *option)
{
// TreeFileMap normal constructor

   fTree     = tree;
   fBranches = branches;
   fOption   = option;
   fOption.ToLower();
   SetBit(kCanDelete);
   
   //create histogram used to draw the map frame
   TFile *file = fTree->GetDirectory()->GetFile();

   if (file->GetEND() > 1000000) {
      fXsize = 1000000;
   } else {
      fXsize = 1000;
   }
   fFrame = new TH1D("hmapframe","",1000,0,fXsize);
   fFrame->SetDirectory(0);
   fFrame->SetBit(TH1::kNoStats);
   fFrame->SetBit(kCanDelete);
   fFrame->SetMinimum(0);
   if (fXsize > 1000) {
      fFrame->GetYaxis()->SetTitle("MBytes");
   } else {
      fFrame->GetYaxis()->SetTitle("KBytes");
   }
   fFrame->GetXaxis()->SetTitle("Bytes");
   fYsize = 1 + Int_t(file->GetEND()/fXsize);
   fFrame->SetMaximum(fYsize+1);
   fFrame->GetYaxis()->SetLimits(0,fYsize+1);
}

//______________________________________________________________________________
TTreeFileMap::~TTreeFileMap()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   delete fFrame;
}

//______________________________________________________________________________
Int_t TTreeFileMap::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute distance from point px,py to this TreeFileMap

   Int_t pxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t pxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t pymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t pymax = gPad->YtoAbsPixel(gPad->GetUymax());
   if (px > pxmin && px < pxmax && py > pymax && py < pymin) return 0;
   return fFrame->DistancetoPrimitive(px,py);
}


//______________________________________________________________________________
void TTreeFileMap::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute action corresponding to one event
   
   fFrame->ExecuteEvent(event,px,py);
}

//______________________________________________________________________________
char *TTreeFileMap::GetObjectInfo(Int_t px, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the branch/basket info
//   corresponding to cursor position px,py
//
   static char info[512];
   Double_t x = gPad->AbsPixeltoX(px);
   Double_t y = gPad->AbsPixeltoY(py);
   Int_t iy   = (Int_t)y;
   Seek_t pbyte = (Seek_t)(fXsize*iy+x);
   Int_t entry = 1;
   
   TIter next(fTree->GetListOfLeaves());
   TLeaf *leaf;
   while ((leaf = (TLeaf*)next())) {
      TBranch *branch = leaf->GetBranch();
      Int_t offsets = branch->GetEntryOffsetLen();
      Int_t len = leaf->GetLen();
      Int_t nbaskets = branch->GetMaxBaskets();
      for (Int_t i=0;i<nbaskets;i++) {
         Seek_t bseek = branch->GetBasketSeek(i);
         if (!bseek) break;
         Int_t nbytes = branch->GetBasketBytes()[i];
         Int_t entry0 = branch->GetBasketEntry()[i];
         if (pbyte >= bseek && pbyte < bseek+nbytes) {
            entry = entry0;
            if (!offsets) entry += (pbyte-bseek)/len;
            sprintf(info,"(byte=%d, branch=%s, basket=%d, entry=%d)",pbyte,branch->GetName(),i,entry);
            return info;            
         }
      }
   }
   sprintf(info,"(byte=%d)",pbyte);
   return info;
}

//______________________________________________________________________________
void TTreeFileMap::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this line with its current attributes*-*-*-*-*-*-*
//*-*                  ===========================================

   // draw map frame
   if (!fOption.Contains("same")) {
      gPad->Clear();
      //just in case axis Y has been unzoomed
      if (fFrame->GetMaximumStored() < -1000) {
         fFrame->SetMaximum(fYsize+1);
         fFrame->SetMinimum(0);
         fFrame->GetYaxis()->SetLimits(0,fYsize+1);
      }
      fFrame->Paint("a");
   }
   
   //draw branch baskets
   Int_t tcolor = fTree->GetFillColor();
   TIter next(fTree->GetListOfLeaves());
   TLeaf *leaf;
   Int_t color = 0;
   Double_t xmin,ymin,xmax,ymax;
   TBox box;
   TRegexp re(fBranches.Data(),kTRUE);
   while ((leaf = (TLeaf*)next())) {
      TBranch *branch = leaf->GetBranch();
      TString s = branch->GetName();
      if (strcmp(fBranches.Data(),branch->GetName()) && s.Index(re) == kNPOS) continue;
      Int_t nbaskets = branch->GetMaxBaskets();
      Int_t bcolor = branch->GetFillColor();
      if (bcolor) {
         box.SetFillColor(bcolor);
         box.SetFillStyle(branch->GetFillStyle());
      } else {
         if (!tcolor) color++;
         else color = tcolor;
         box.SetFillColor(color);
         box.SetFillStyle(fTree->GetFillStyle());
      }
      for (Int_t i=0;i<nbaskets;i++) {
         Seek_t bseek = branch->GetBasketSeek(i);
         if (!bseek) break;
         Int_t nbytes = branch->GetBasketBytes()[i];
         Int_t iy = bseek/fXsize;
         Int_t ix = bseek%fXsize;
         Int_t ny = 1+(nbytes+ix)/fXsize;
         for (Int_t j=0;j<ny;j++) {
            if (j == 0) xmin = (Double_t)ix;
            else        xmin = 0;
            xmax = xmin + nbytes;
            if (xmax > fXsize) xmax = fXsize;
            ymin = iy+j;
            ymax = ymin+1;
            nbytes -= (Int_t)(xmax-xmin);
            if (xmax < gPad->GetUxmin()) continue;
            if (xmin > gPad->GetUxmax()) continue;
            if (xmin < gPad->GetUxmin()) xmin = gPad->GetUxmin();
            if (xmax > gPad->GetUxmax()) xmax = gPad->GetUxmax();
            if (ymax < gPad->GetUymin()) continue;
            if (ymin > gPad->GetUymax()) continue;
            if (ymin < gPad->GetUymin()) ymin = gPad->GetUymin();
            if (ymax > gPad->GetUymax()) ymax = gPad->GetUymax();
            box.PaintBox(xmin,ymin,xmax,ymax);
         }
      }
   }
   gPad->RedrawAxis();
}

//______________________________________________________________________________
void TTreeFileMap::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out
}

//______________________________________________________________________________
void TTreeFileMap::ShowEntry()
{
    // Show entry corresponding at the mouse position

   Int_t px = gPad->GetEventX();
   Int_t py = gPad->GetEventY();
   Double_t x = gPad->AbsPixeltoX(px);
   Double_t y = gPad->AbsPixeltoY(py);
   Int_t iy   = (Int_t)y;
   Seek_t pbyte = (Seek_t)(fXsize*iy+x);
   Int_t entry = 1;
   
   TIter next(fTree->GetListOfLeaves());
   TLeaf *leaf;
   while ((leaf = (TLeaf*)next())) {
      TBranch *branch = leaf->GetBranch();
      Int_t offsets = branch->GetEntryOffsetLen();
      Int_t len = leaf->GetLen();
      Int_t nbaskets = branch->GetMaxBaskets();
      for (Int_t i=0;i<nbaskets;i++) {
         Seek_t bseek = branch->GetBasketSeek(i);
         if (!bseek) break;
         Int_t nbytes = branch->GetBasketBytes()[i];
         Int_t entry0 = branch->GetBasketEntry()[i];
         if (pbyte >= bseek && pbyte < bseek+nbytes) {
            entry = entry0;
            if (!offsets) entry += (pbyte-bseek)/len;
            fTree->Show(entry);
            return;
         }
      }
   }
}

//______________________________________________________________________________
void TTreeFileMap::Streamer(TBuffer &R__b)
{
   // Stream an object of class TTreeFileMap.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 0) {
         TTreeFileMap::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }

   } else {
      TTreeFileMap::Class()->WriteBuffer(R__b,this);
   }
}
