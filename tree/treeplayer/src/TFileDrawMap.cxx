// @(#)root/treeplayer:$Id$
// Author: Rene Brun   15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TFileDrawMap
This class is automatically called by TFile::DrawMap.
It draws a canvas showing the internal structure of a ROOT file.
Each key or basket in a file is shown with a fill area drawn
at the byte position of the key/basket in the file.
The Y axis of the canvas shows the number of Kbytes/Mbytes.
The X axis shows the bytes between y(i) and y(i+1).
A color corresponding to the class in the key/basket is automatically
selected using the class unique identifier.

When moving the mouse in the canvas, the "Event Status" panels
shows the object corresponding to the mouse position.
if the object is a key, it shows the class and object name as well as
the file directory name if the file has sub-directories.

if the object is a basket, it shows:
 - the name of the Tree
 - the name of the branch
 - the basket number
 - the entry number in the basket

Special keys like the StreamerInfo record, the Keys List Record
and the Free Blocks Record are also shown.

When clicking the right mouse button, a pop-up menu is shown
with its title identifying the picked object and with the items:
 - DrawObject: in case of a key, the Draw function of the object is called
               in case of a basket, the branch is drawn for all entries
 - DumpObject: in case of a key, the Dump function of the object is called
               in case of a basket, tree->Show(entry) is called
 - InspectObject: the Inspect function is called for the object.

The normal axis zoom functionality can be used to zoom or unzoom
One can also use the TCanvas context menu SetCanvasSize to make
a larger canvas and use the canvas scroll bars.

When the class is built, it is possible to identify a subset of the
objects to be shown. For example, to view only the keys with
names starting with "abc", set the argument keys to "abc*".
The default is to view all the objects.
The argument options can also be used (only one option currently)
When the option "same" is given, the new picture is superimposed.
The option "same" is useful, eg:
to draw all keys with names = "abc" in a first pass
then all keys with names = "uv*" in a second pass, etc.
*/

#include "TFileDrawMap.h"
#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TH2.h"
#include "TBox.h"
#include "TKey.h"
#include "TRegexp.h"
#include "TSystem.h"
#include "strlcpy.h"

ClassImp(TFileDrawMap);

////////////////////////////////////////////////////////////////////////////////
/// Default TreeFileMap constructor.

TFileDrawMap::TFileDrawMap() :TNamed()
{
   fFile   = nullptr;
   fFrame  = nullptr;
   fXsize  = 1000;
   fYsize  = 1000;
}

////////////////////////////////////////////////////////////////////////////////
/// TFileDrawMap normal constructor.
/// see descriptions of arguments above

TFileDrawMap::TFileDrawMap(const TFile *file, const char *keys, Option_t *)
         : TNamed("TFileDrawMap","")
{
   fFile     = (TFile*) file;
   fKeys     = keys;
   SetBit(kCanDelete);

   //create histogram used to draw the map frame

   if (file->GetEND() > 1000000) {
      fXsize = 1000000;
   } else {
      fXsize = 1000;
   }
   fYsize = 1 + Int_t(file->GetEND()/fXsize);

   fFrame = new TH2D("hmapframe","",100,0,fXsize,100,0,fYsize);
   fFrame->SetDirectory(nullptr);
   fFrame->SetBit(TH1::kNoStats);
   fFrame->SetBit(kCanDelete);
   if (fXsize > 1000) {
      fFrame->GetYaxis()->SetTitle("MBytes");
   } else {
      fFrame->GetYaxis()->SetTitle("KBytes");
   }
   fFrame->GetXaxis()->SetTitle("Bytes");

   if (gPad)
      gPad->Clear();

   fFrame->Draw("axis");
   Draw();

   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Tree destructor.

TFileDrawMap::~TFileDrawMap()
{
   //delete fFrame; //should not be deleted (kCanDelete set)
}

////////////////////////////////////////////////////////////////////////////////
/// Returns info which corresponds to recent mouse position
/// In case of normal graphics it is object name
/// In case of web canvas use stored click event position

TString TFileDrawMap::GetRecentInfo()
{
   TString info;
   if (gPad && gPad->IsWeb()) {
      // in case of web canvas one can try to use last click event
      GetObjectInfoDir(fFile, gPad->GetEventX(), gPad->GetEventY(), info);
   } else {
      // last selected place stored as name
      info = GetName();
   }
   return info;
}


////////////////////////////////////////////////////////////////////////////////
/// Show sequence of baskets reads for the list of baskets involved
/// in the list of branches (separated by ",")
/// - if branches="", the branch pointed by the mouse is taken.
/// - if branches="*", all branches are taken
/// Example:
///
///     AnimateTree("x,y,u");

void  TFileDrawMap::AnimateTree(const char *branches)
{
   TString info = GetRecentInfo();

   auto pos = info.Index(", basket=");
   if (pos == kNPOS)
      return;
   info.Remove(pos);

   pos = info.Index(", branch=");
   if (pos == kNPOS)
      return;
   TString select_branches = info(pos + 9, info.Length() - pos - 9);

   auto colon = info.Index("::");
   if (colon == kNPOS)
      return;

   info.Resize(colon - 1);

   auto tree = fFile->Get<TTree>(info);
   if (!tree)
      return;
   if (branches && *branches)
      select_branches = branches;

   // create list of branches
   Int_t nzip = 0;
   TBranch *branch;
   TObjArray list;
   char *comma;
   while((comma = strrchr((char*)select_branches.Data(),','))) {
      *comma = 0;
      comma++;
      while (*comma == ' ') comma++;
      branch = tree->GetBranch(comma);
      if (branch) {
         nzip += (Int_t)branch->GetZipBytes();
         branch->SetUniqueID(0);
         list.Add(branch);
      }
   }
   comma = (char*)select_branches.Data();
   while (*comma == ' ') comma++;
   branch = tree->GetBranch(comma);
   if (branch) {
      nzip += (Int_t)branch->GetZipBytes();
      branch->SetUniqueID(0);
      list.Add(branch);
   }
   Double_t fractionRead = Double_t(nzip)/Double_t(fFile->GetEND());
   Int_t nbranches = list.GetEntries();

   // loop on all tree entries
   Int_t nentries = (Int_t)tree->GetEntries();
   Int_t sleep = 1;
   Int_t stime = (Int_t)(100./(nentries*fractionRead));
   if (stime < 10) {stime=1; sleep = nentries/400;}
   gPad->SetDoubleBuffer(0);             // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode
   for (Int_t entry=0;entry<nentries;entry++) {
      for (Int_t ib=0;ib<nbranches;ib++) {
         branch = (TBranch*)list.At(ib);
         Int_t nbaskets = branch->GetListOfBaskets()->GetSize();
         Int_t basket = TMath::BinarySearch(nbaskets,branch->GetBasketEntry(), (Long64_t) entry);
         Int_t nbytes = branch->GetBasketBytes()[basket];
         Int_t bseek  = branch->GetBasketSeek(basket);
         Int_t entry0 = branch->GetBasketEntry()[basket];
         Int_t entryn = branch->GetBasketEntry()[basket+1];
         Int_t eseek  = (Int_t)(bseek + nbytes*Double_t(entry-entry0)/Double_t(entryn-entry0));
         DrawMarker(ib,branch->GetUniqueID());
         DrawMarker(ib,eseek);
         branch->SetUniqueID(eseek);
         gSystem->ProcessEvents();
         if (entry%sleep == 0) gSystem->Sleep(stime);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to this TreeFileMap.
/// Find the closest object to the mouse, save its path in the TFileDrawMap name.

Int_t TFileDrawMap::DistancetoPrimitive(Int_t px, Int_t py)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Draw marker.

void TFileDrawMap::DrawMarker(Int_t marker, Long64_t eseek)
{
   Int_t iy = gPad->YtoAbsPixel(eseek/fXsize);
   Int_t ix = gPad->XtoAbsPixel(eseek%fXsize);
   Int_t d;
   Int_t mark = marker%4;
   switch (mark) {
      case 0 :
         d = 6; //arrow
         gVirtualX->DrawLine(ix-3*d,iy,ix,iy);
         gVirtualX->DrawLine(ix-d,iy+d,ix,iy);
         gVirtualX->DrawLine(ix-d,iy-d,ix,iy);
         gVirtualX->DrawLine(ix-d,iy-d,ix-d,iy+d);
         break;
      case 1 :
         d = 5; //up triangle
         gVirtualX->DrawLine(ix-d,iy-d,ix+d,iy-d);
         gVirtualX->DrawLine(ix+d,iy-d,ix,iy+d);
         gVirtualX->DrawLine(ix,iy+d,ix-d,iy-d);
         break;
      case 2 :
         d = 5; //open square
         gVirtualX->DrawLine(ix-d,iy-d,ix+d,iy-d);
         gVirtualX->DrawLine(ix+d,iy-d,ix+d,iy+d);
         gVirtualX->DrawLine(ix+d,iy+d,ix-d,iy+d);
         gVirtualX->DrawLine(ix-d,iy+d,ix-d,iy-d);
         break;
      case 3 :
         d = 8; //cross
         gVirtualX->DrawLine(ix-d,iy,ix+d,iy);
         gVirtualX->DrawLine(ix,iy-d,ix,iy+d);
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw object at the mouse position.

void TFileDrawMap::DrawObject()
{
   TVirtualPad *padsave = gROOT->GetSelectedPad();
   if ((padsave == gPad) || (gPad && gPad->IsWeb())) {
      //must create a new canvas
      gROOT->MakeDefCanvas();
   } else if (padsave) {
      padsave->cd();
   }

   TString info = GetRecentInfo();

   // case of a TTree
   auto pbasket = info.Index(", basket=");
   if (pbasket != kNPOS) {
      info.Resize(pbasket);
      auto pbranch = info.Index(", branch=");
      if (pbranch == kNPOS)
         return;

      TString cbranch = info(pbranch + 9, info.Length() - pbranch - 9);

      auto colon = info.Index("::");
      if (colon == kNPOS)
         return;

      info.Resize(colon - 1);

      auto tree = fFile->Get<TTree>(info);
      if (tree)
         tree->Draw(cbranch);
   } else {
      // other objects
      auto obj = GetObject();
      if (obj)
         obj->Draw();
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Dump object at the mouse position.

void TFileDrawMap::DumpObject()
{
   TObject *obj = GetObject();
   if (obj) {
      obj->Dump();
      return;
   }

   TString info = GetRecentInfo();

   auto indx = info.Index("entry=");
   if (indx == kNPOS)
      return;

   Int_t entry = 0;
   sscanf(info.Data() + indx + 6, "%d", &entry);

   auto colon = info.Index("::");
   if (colon == kNPOS)
      return;

   info.Resize(colon - 1);

   auto tree = fFile->Get<TTree>(info);
   if (tree)
      tree->Show(entry);
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve object at the mouse position in memory.

TObject *TFileDrawMap::GetObject()
{
   TString info = GetRecentInfo();

   if (info.Contains("entry="))
      return nullptr;

   auto colon = info.Index("::");
   if (colon == kNPOS)
      return nullptr;

   info.Resize(colon - 1);

   return fFile->Get(info);
}

////////////////////////////////////////////////////////////////////////////////
/// Redefines TObject::GetObjectInfo.
/// Displays the keys info in the file corresponding to cursor position px,py
/// in the canvas status bar info panel

char *TFileDrawMap::GetObjectInfo(Int_t px, Int_t py) const
{
   // Thread safety: this solution is not elegant, but given the action performed
   // by the method, this construct can be considered non-problematic.
   static TString info;
   GetObjectInfoDir(fFile, px, py, info);
   return (char*)info.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Redefines TObject::GetObjectInfo.
/// Displays the keys info in the directory
/// corresponding to cursor position px,py

bool TFileDrawMap::GetObjectInfoDir(TDirectory *dir, Int_t px, Int_t py, TString &info) const
{
   Double_t x = gPad->AbsPixeltoX(px);
   Double_t y = gPad->AbsPixeltoY(py);
   Int_t iy   = (Int_t)y;
   Long64_t pbyte = (Long64_t)(fXsize*iy+x);
   Int_t nbytes;
   Long64_t bseek;
   TDirectory *dirsav = gDirectory;
   dir->cd();

   TIter next(dir->GetListOfKeys());
   TKey *key;
   while ((key = (TKey*)next())) {
      TDirectory *curdir = gDirectory;
      TClass *cl = TClass::GetClass(key->GetClassName());
      // a TDirectory ?
      if (cl && cl == TDirectoryFile::Class()) {
         curdir->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         bool gotInfo = GetObjectInfoDir(subdir, px, py, info);
         if (gotInfo) {
            dirsav->cd();
            return true;
         }
         curdir->cd();
         continue;
      }
      // a TTree ?
      if (cl && cl->InheritsFrom(TTree::Class())) {
         TTree *tree = (TTree*)gDirectory->Get(key->GetName());
         TIter nextb(tree->GetListOfLeaves());
         while (auto leaf = (TLeaf *)nextb()) {
            TBranch *branch = leaf->GetBranch();
            Int_t nbaskets = branch->GetMaxBaskets();
            Int_t offsets = branch->GetEntryOffsetLen();
            Int_t len = leaf->GetLen();
            for (Int_t i = 0; i < nbaskets; i++) {
               bseek = branch->GetBasketSeek(i);
               if (!bseek)
                  break;
               nbytes = branch->GetBasketBytes()[i];
               if (pbyte >= bseek && pbyte < bseek + nbytes) {
                  Int_t entry = branch->GetBasketEntry()[i];
                  if (!offsets) entry += (pbyte-bseek)/len;
                  if (curdir == (TDirectory*)fFile) {
                     info.Form("%s%s ::%s, branch=%s, basket=%d, entry=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),branch->GetName(),i,entry);
                  } else {
                     info.Form("%s/%s ::%s, branch=%s, basket=%d, entry=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),branch->GetName(),i,entry);
                  }
                  return true;
               }
            }
         }
      }
      nbytes = key->GetNbytes();
      bseek = key->GetSeekKey();
      if (pbyte >= bseek && pbyte < bseek+nbytes) {
         if (curdir == (TDirectory*)fFile) {
            info.Form("%s%s ::%s, nbytes=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),nbytes);
         } else {
            info.Form("%s/%s ::%s, nbytes=%d",curdir->GetPath(),key->GetName(),key->GetClassName(),nbytes);
         }
         dirsav->cd();
         return true;
      }
   }
   // Are we in the Keys list
   if (pbyte >= dir->GetSeekKeys() && pbyte < dir->GetSeekKeys()+dir->GetNbytesKeys()) {
      info.Form("%sKeys List, nbytes=%d",dir->GetPath(),dir->GetNbytesKeys());
      dirsav->cd();
      return true;
   }
   if (dir == (TDirectory*)fFile) {
      // Are we in the TStreamerInfo
      if (pbyte >= fFile->GetSeekInfo() && pbyte < fFile->GetSeekInfo()+fFile->GetNbytesInfo()) {
         info.Form("%sStreamerInfo List, nbytes=%d",dir->GetPath(),fFile->GetNbytesInfo());
         dirsav->cd();
         return true;
      }
      // Are we in the Free Segments
      if (pbyte >= fFile->GetSeekFree() && pbyte < fFile->GetSeekFree()+fFile->GetNbytesFree()) {
         info.Form("%sFree List, nbytes=%d",dir->GetPath(),fFile->GetNbytesFree());
         dirsav->cd();
         return true;
      }
   }
   info.Form("(byte=%lld)",pbyte);
   dirsav->cd();
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Inspect object at the mouse position.

void TFileDrawMap::InspectObject()
{
   TObject *obj = GetObject();
   if (obj) obj->Inspect();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this TFileDrawMap.

void TFileDrawMap::Paint(Option_t *)
{
   //draw keys
   PaintDir(fFile, fKeys.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Paint the object at bseek with nbytes using the box object.

void TFileDrawMap::PaintBox(TBox &box, Long64_t bseek, Int_t nbytes)
{
   Int_t iy = bseek/fXsize;
   Int_t ix = bseek%fXsize;
   Int_t ny = 1+(nbytes+ix)/fXsize;
   Double_t xmin,ymin,xmax,ymax;
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
      //box.TAttFill::Modify();
      box.PaintBox(xmin,ymin,xmax,ymax);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint keys in a directory.

void TFileDrawMap::PaintDir(TDirectory *dir, const char *keys)
{
   TDirectory *dirsav = gDirectory;
   TIter next(dir->GetListOfKeys());
   TKey *key;
   Int_t color = 0;
   TBox box;
   TRegexp re(keys,true);
   while ((key = (TKey*)next())) {
      Int_t nbytes = key->GetNbytes();
      Long64_t bseek = key->GetSeekKey();
      TClass *cl = TClass::GetClass(key->GetClassName());
      if (cl) {
         color = (Int_t)(cl->GetUniqueID()%20);
      } else {
         color = 1;
      }
      box.SetFillColor(color);
      box.SetFillStyle(1001);
      TString s = key->GetName();
      if (strcmp(fKeys.Data(),key->GetName()) && s.Index(re) == kNPOS) continue;
      // a TDirectory ?
      if (cl && cl == TDirectoryFile::Class()) {
         TDirectory *curdir = gDirectory;
         gDirectory->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         PaintDir(subdir,"*");
         curdir->cd();
      }
      PaintBox(box,bseek,nbytes);
      // a TTree ?
      if (cl && cl->InheritsFrom(TTree::Class())) {
         TTree *tree = (TTree*)gDirectory->Get(key->GetName());
         TIter nextb(tree->GetListOfLeaves());
         while (auto leaf = (TLeaf*)nextb()) {
            TBranch *branch = leaf->GetBranch();
            color = branch->GetFillColor();
            if (color == 0) {
               if (fBranchColors.find(branch) == fBranchColors.end()) {
                  gPad->IncrementPaletteColor(1, "pfc");
                  fBranchColors[branch] = gPad->NextPaletteColor();
               }
               color = fBranchColors[branch];
            }
            box.SetFillColor(color);
            Int_t nbaskets = branch->GetMaxBaskets();
            for (Int_t i=0;i<nbaskets;i++) {
               bseek = branch->GetBasketSeek(i);
               if (!bseek) break;
               nbytes = branch->GetBasketBytes()[i];
               PaintBox(box,bseek,nbytes);
            }
         }
      }
   }

   // draw the box for Keys list
   box.SetFillColor(50);
   box.SetFillStyle(1001);
   PaintBox(box,dir->GetSeekKeys(),dir->GetNbytesKeys());
   if (dir == (TDirectory*)fFile) {
      // draw the box for TStreamerInfo
      box.SetFillColor(6);
      box.SetFillStyle(3008);
      PaintBox(box,fFile->GetSeekInfo(),fFile->GetNbytesInfo());
      // draw the box for Free Segments
      box.SetFillColor(1);
      box.SetFillStyle(1001);
      PaintBox(box,fFile->GetSeekFree(),fFile->GetNbytesFree());
   }
   dirsav->cd();
}
