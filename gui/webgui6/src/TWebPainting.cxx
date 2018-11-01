// Author:  Sergey Linev, GSI  10/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebPainting.h"

///////////////////////////////////////////////////////////////////////////////////////
/// Constructor

TWebPainting::TWebPainting()
{
   fLastFill.SetFillStyle(9999);
   fLastLine.SetLineWidth(-123);
   fLastMarker.SetMarkerStyle(9999);
}

///////////////////////////////////////////////////////////////////////////////////////
/// Add next custom operator to painting
/// Operations are separated by semicolons

void TWebPainting::AddOper(const std::string &oper)
{
   if (!fOper.empty())
      fOper.append(";");
   fOper.append(oper);
}

///////////////////////////////////////////////////////////////////////////////////////
/// Reserve place in the float buffer
/// Returns pointer on first element in reserved area

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

///////////////////////////////////////////////////////////////////////////////////////
/// Store line attributes
/// If attributes were not changed - ignore operation

void TWebPainting::AddLineAttr(const TAttLine &attr)
{
   if ((attr.GetLineColor() == fLastLine.GetLineColor()) &&
       (attr.GetLineStyle() == fLastLine.GetLineStyle()) &&
       (attr.GetLineWidth() == fLastLine.GetLineWidth())) return;

   fLastLine = attr;

   AddOper(std::string("lattr:") +
           std::to_string((int) attr.GetLineColor()) + ":" +
           std::to_string((int) attr.GetLineStyle()) + ":" +
           std::to_string((int) attr.GetLineWidth()));
}

///////////////////////////////////////////////////////////////////////////////////////
/// Store fill attributes
/// If attributes were not changed - ignore operation

void TWebPainting::AddFillAttr(const TAttFill &attr)
{
   if ((fLastFill.GetFillColor() == attr.GetFillColor()) &&
       (fLastFill.GetFillStyle() == attr.GetFillStyle())) return;

   fLastFill = attr;

   AddOper(std::string("fattr:") +
           std::to_string((int) attr.GetFillColor()) + ":" +
           std::to_string((int) attr.GetFillStyle()));
}

///////////////////////////////////////////////////////////////////////////////////////
/// Store text attributes
/// If attributes were not changed - ignore operation

void TWebPainting::AddTextAttr(const TAttText &attr)
{
   AddOper(std::string("tattr:") +
           std::to_string((int) attr.GetTextColor()) + ":" +
           std::to_string((int) attr.GetTextFont()) + ":" +
           std::to_string((int) (attr.GetTextSize()>=1 ? attr.GetTextSize() : -1000*attr.GetTextSize())) + ":" +
           std::to_string((int) attr.GetTextAlign()) + ":" +
           std::to_string((int) attr.GetTextAngle()));
}

///////////////////////////////////////////////////////////////////////////////////////
/// Store marker attributes
/// If attributes were not changed - ignore operation

void TWebPainting::AddMarkerAttr(const TAttMarker &attr)
{
   if ((attr.GetMarkerColor() == fLastMarker.GetMarkerColor()) &&
       (attr.GetMarkerStyle() == fLastMarker.GetMarkerStyle()) &&
       (attr.GetMarkerSize() == fLastMarker.GetMarkerSize())) return;

   fLastMarker = attr;

   AddOper(std::string("mattr:") +
           std::to_string((int) attr.GetMarkerColor()) + ":" +
           std::to_string((int) attr.GetMarkerStyle()) + ":" +
           std::to_string((int) attr.GetMarkerSize()));
}

///////////////////////////////////////////////////////////////////////////////////////
/// Add custom color to operations

void TWebPainting::AddColor(Int_t indx, TColor *col)
{
   if (!col) return;

   TString code;

   if (col->GetAlpha() == 1)
      code.Form("rgb(%d,%d,%d)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()));
   else
      code.Form("rgba(%d,%d,%d,%5.3f)", (int) (255*col->GetRed()), (int) (255*col->GetGreen()), (int) (255*col->GetBlue()), col->GetAlpha());
   if (indx>=0)
      code.Prepend(TString::Format("%d:", indx));

   AddOper(code.Data());
}

