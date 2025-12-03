/// \file RTreeMapPainter.cxx
/// \ingroup TreeMap ROOT7
/// \author Patryk Tymoteusz Pilichowski <patryk.tymoteusz.pilichowski@cern.ch>
/// \date 2025-08-21
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RTreeMapPainter.hxx>

#include <TCanvas.h>
#include <TPad.h>
#include <TBox.h>
#include <TLatex.h>
#include <TColor.h>

#include <cmath>

void ROOT::Experimental::RTreeMapPainter::Paint(Option_t *)
{
   if (!gPad)
      return;
   gPad->Clear();
   gPad->Range(0, 0, 1, 1);
   gPad->cd();
   gPad->SetEditable(kFALSE);
   DrawTreeMap(fNodes[0], Rect(Vec2(0.025, 0.05), Vec2(0.825, 0.9)), 0);
   DrawLegend();
}

void ROOT::Experimental::RTreeMapPainter::AddBox(const Rect &rect, const RGBColor &color, float borderWidth) const
{
   auto box = new TBox(rect.fBottomLeft.x, rect.fBottomLeft.y, rect.fTopRight.x, rect.fTopRight.y);
   box->SetFillColor(TColor::GetColor(color.r, color.g, color.b, color.a));
   box->SetLineColor(kGray);
   box->SetLineWidth(std::ceil(borderWidth));
   gPad->Add(box, "l");
}

void ROOT::Experimental::RTreeMapPainter::AddText(const Vec2 &pos, const std::string &content, float size,
                                                  const RGBColor &color, bool alignCenter) const
{
   auto t = new TLatex(pos.x, pos.y, content.c_str());
   t->SetTextFont(42);
   t->SetTextSize(size);
   t->SetTextAlign((alignCenter) ? 22 : 13);
   t->SetTextColor(TColor::GetColor(color.r, color.g, color.b, color.a));
   gPad->Add(t);
}
