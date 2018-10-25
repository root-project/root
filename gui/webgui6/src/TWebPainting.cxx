// Author:  Sergey Linev, GSI  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebPainting.h"

Float_t *TWebPainting::Reserve(Int_t sz)
{
   if (sz <= 0)
      return nullptr;

   if (fSize + sz > fBuf.GetSize()) {
      Int_t nextsz = fBuf.GetSize() + TMath::Max(1024, (sz/128 + 1) * 128);
      fBuf.Set(nextsz);
   }

   Float_t *res = fBuf.GetArray() + fSize;
   fSize += sz;
   return res; // return size where drawing can start
}


void TWebPainting::AddLineAttr(const TAttLine &attr)
{
   AddOper(std::string("lattr:") +
           std::to_string((int) attr.GetLineColor()) + ":" +
           std::to_string((int) attr.GetLineStyle()) + ":" +
           std::to_string((int) attr.GetLineWidth()));
}

void TWebPainting::AddFillAttr(const TAttFill &attr)
{
   AddOper(std::string("fattr:") +
           std::to_string((int) attr.GetFillColor()) + ":" +
           std::to_string((int) attr.GetFillStyle()));
}

void TWebPainting::AddTextAttr(const TAttText &attr)
{
   AddOper(std::string("tattr:") +
           std::to_string((int) attr.GetTextColor()) + ":" +
           std::to_string((int) attr.GetTextFont()) + ":" +
           std::to_string((int) (attr.GetTextSize()>=1 ? attr.GetTextSize() : -1000*attr.GetTextSize())) + ":" +
           std::to_string((int) attr.GetTextAlign()) + ":" +
           std::to_string((int) attr.GetTextAngle()));
}

void TWebPainting::AddMarkerAttr(const TAttMarker &attr)
{
   AddOper(std::string("mattr:") +
           std::to_string((int) attr.GetMarkerColor()) + ":" +
           std::to_string((int) attr.GetMarkerStyle()) + ":" +
           std::to_string((int) attr.GetMarkerSize()));
}

void TWebPainting::AddColor(TColor *col, Bool_t onlyindx)
{
   if (!col) return;
   if (onlyindx) {
      AddOper("col:" + std::to_string(col->GetNumber()));
   } else if (col->GetAlpha() == 1) {
      AddOper("rgb:" + std::to_string(col->GetNumber()));
      auto buf = Reserve(3);
      buf[0] = (int) (255*col->GetRed());
      buf[1] = (int) (255*col->GetGreen());
      buf[2] = (int) (255*col->GetBlue());
   } else {
      AddOper("rga:" + std::to_string(col->GetNumber()));
      auto buf = Reserve(4);
      buf[0] = (int) (255*col->GetRed());
      buf[1] = (int) (255*col->GetGreen());
      buf[2] = (int) (255*col->GetBlue());
      buf[3] = (int) (1000*col->GetAlpha());
   }
}

