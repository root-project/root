// @(#)root/treeplayer:$Name:  $:$Id: TFileMap.cxx,v 1.1 2003/01/15 18:48:16 brun Exp $
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
// TFileMap                                                             //
//                                                                      //
//Begin_Html
/*
<img src="gif/filemap.gif">
*/
//End_Html
//
//  =============================================================================

#include "TFileMap.h"
#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TH1.h"
#include "TBox.h"
#include "TKey.h"
#include "TRegexp.h"

ClassImp(TFileMap)

//______________________________________________________________________________
TFileMap::TFileMap() :TNamed()
{
// Default TreeFileMap constructor

   fFile   = 0;
   fFrame  = 0;
}

//______________________________________________________________________________
TFileMap::TFileMap(const TFile *file, const char *keys, Option_t *option)
         : TNamed("TFileMap","")
{
// TFileMap normal constructor

   fFile     = (TFile*)file;
   fKeys     = keys;
   fOption   = option;
   fOption.ToLower();
   SetBit(kCanDelete);
   
   //create histogram used to draw the map frame

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
   
   if (gPad) gPad->Clear();
   Draw();
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
TFileMap::~TFileMap()
{
//*-*-*-*-*-*-*-*-*-*-*Tree destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   //delete fFrame; //should not be deleted (kCanDelete set)
}

//______________________________________________________________________________
Int_t TFileMap::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute distance from point px,py to this TreeFileMap
// Find the closest object to the mouse, save its path in the TFileMap name.
   
   Int_t pxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t pxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t pymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t pymax = gPad->YtoAbsPixel(gPad->GetUymax());
   if (px > pxmin && px < pxmax && py > pymax && py < pymin) {
      SetName(GetObjectInfo(px,py));
      return 0;
   }
   return fFrame->DistancetoPrimitive(px,py);
}

//______________________________________________________________________________
void TFileMap::DrawObject()
{
// Draw object at the mouse position

   TVirtualPad *padsave = gROOT->GetSelectedPad();
   TObject *obj = GetObject();
   if (obj && obj->InheritsFrom(TTree::Class())) {
      TTree *tree = (TTree*)obj;
      tree->DrawMap("*","same");
      return;
   }
   if (padsave == gPad) {
      //must create a new canvas
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   } else {
      padsave->cd();
   }
   if (obj) obj->Draw();
}


//______________________________________________________________________________
void TFileMap::DumpObject()
{
// Dump object at the mouse position

   TObject *obj = GetObject();
   if (obj) obj->Dump();
}

//______________________________________________________________________________
void TFileMap::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute action corresponding to one event
   
   fFrame->ExecuteEvent(event,px,py);
}

//______________________________________________________________________________
TObject *TFileMap::GetObject()
{
// Retrieve object at the mouse position in memory

   char info[512];
   strcpy(info,GetName());
   char *colon = strstr(info,"::");
   if (!colon) return 0;
   colon--;
   *colon = 0;
   return fFile->Get(info);
}

//______________________________________________________________________________
char *TFileMap::GetObjectInfo(Int_t px, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the keys info in the file corresponding to cursor position px,py
//   in the canvas status bar info panel
      
   static char info[512];
   GetObjectInfoDir(fFile, px, py, info);
   return info;
}

//______________________________________________________________________________
Bool_t TFileMap::GetObjectInfoDir(TDirectory *dir, Int_t px, Int_t py, char *info) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the keys info in the directory
//   corresponding to cursor position px,py
//
   Double_t x = gPad->AbsPixeltoX(px);
   Double_t y = gPad->AbsPixeltoY(py);
   Int_t iy   = (Int_t)y;
   Seek_t pbyte = (Seek_t)(fXsize*iy+x);
   TDirectory *dirsav = gDirectory;
   dir->cd();
   
   TIter next(dir->GetListOfKeys());
   TKey *key;
   while ((key = (TKey*)next())) {
      TDirectory *curdir = gDirectory;
      if (!strcmp(key->GetClassName(),"TDirectory")) {
         curdir->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         Bool_t gotInfo = GetObjectInfoDir(subdir, px, py, info);
         if (gotInfo) {
            dirsav->cd();
            return kTRUE;
         }
         curdir->cd();
         continue;
      }
      Int_t nbytes = key->GetNbytes();
      Seek_t bseek = key->GetSeekKey();
      if (pbyte >= bseek && pbyte < bseek+nbytes) {
         if (curdir == (TDirectory*)fFile) {
            sprintf(info,"%s%s ::%s, nbytes=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),nbytes);
         } else {
            sprintf(info,"%s/%s ::%s, nbytes=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),nbytes);
         }
         dirsav->cd();
         return kTRUE;            
      }
   }
   sprintf(info,"(byte=%d)",pbyte);
   dirsav->cd();
   return kFALSE;
}

//______________________________________________________________________________
void TFileMap::InspectObject()
{
// Inspect object at the mouse position

   TObject *obj = GetObject();
   if (obj) obj->Inspect();
}

//______________________________________________________________________________
void TFileMap::Paint(Option_t *)
{
//  Paint this TFileMap

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
   
   //draw keys
   PaintDir(fFile, fKeys.Data());
   
   fFrame->Draw("sameaxis");
}

//______________________________________________________________________________
void TFileMap::PaintDir(TDirectory *dir, const char *keys)
{
// Paint keys in a directory
   
   TDirectory *dirsav = gDirectory;
   TIter next(dir->GetListOfKeys());
   TKey *key;
   Int_t color = 0;
   Double_t xmin,ymin,xmax,ymax;
   TBox box;
   TRegexp re(keys,kTRUE);
   while ((key = (TKey*)next())) {
      Int_t nbytes = key->GetNbytes();
      Seek_t bseek = key->GetSeekKey();
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (cl) {
         color = (Int_t)(cl->GetUniqueID()%20);
      } else {
         color = 1;
      }
      TString s = key->GetName();
      if (strcmp(fKeys.Data(),key->GetName()) && s.Index(re) == kNPOS) continue;
      if (!strcmp(key->GetClassName(),"TDirectory")) {
         TDirectory *curdir = gDirectory;
         gDirectory->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         PaintDir(subdir,"*");
         curdir->cd();
      }
      box.SetFillColor(color);
      box.SetFillStyle(1001);
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
   dirsav->cd();
}
