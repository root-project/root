/// \file TCanvas.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TCanvas.hxx"

#include "ROOT/TLogger.hxx"

#include <algorithm>
#include <memory>
#include <stdio.h>
#include <string.h>

#include "TROOT.h"

namespace {
static std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> &GetHeldCanvases()
{
   static std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> sCanvases;
   return sCanvases;
}
} // namespace

const std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> &ROOT::Experimental::TCanvas::GetCanvases()
{
   return GetHeldCanvases();
}

// void ROOT::Experimental::TCanvas::Paint() {
//  for (auto&& drw: fPrimitives) {
//    drw->Paint(*this);
//  }
// }

bool ROOT::Experimental::TCanvas::IsModified() const
{
   return fPainter ? fPainter->IsCanvasModified(fModified) : fModified;
}

void ROOT::Experimental::TCanvas::Update(bool async, CanvasCallback_t callback)
{
   if (fPainter)
      fPainter->CanvasUpdated(fModified, async, callback);

   // SnapshotList_t lst;
   // for (auto&& drw: fPrimitives) {
   //   TSnapshot *snap = drw->CreateSnapshot(*this);
   //   lst.push_back(std::unique_ptr<TSnapshot>(snap));
   // }
}

std::shared_ptr<ROOT::Experimental::TCanvas> ROOT::Experimental::TCanvas::Create(const std::string &title)
{
   auto pCanvas = std::make_shared<TCanvas>();
   pCanvas->SetTitle(title);
   GetHeldCanvases().emplace_back(pCanvas);
   return pCanvas;
}

//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// Parameter \par where specifies which program could be used for display creation
/// Possible values:
///
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine (when running via rootqt5), local display, local communication
///  browser - default system web-browser, communication via random http port from range 8800 - 9800
///  <prog> - any program name which will be started instead of default browser, like firefox or /usr/bin/opera
///           one could also specify $url in program name, which will be replaced with canvas URL
///  native - either any available local display or default browser
///
///  Canvas can be displayed in several different places

void ROOT::Experimental::TCanvas::Show(const std::string &where)
{
   if (fPainter) {
      if (!where.empty())
         fPainter->NewDisplay(where);
      return;
   }

   bool batch_mode = gROOT->IsBatch();
   if (!fModified)
      fModified = 1; // 0 is special value, means no changes and no drawings

   fPainter = Internal::TVirtualCanvasPainter::Create(*this, batch_mode);
   if (fPainter) {
      fPainter->NewDisplay(where);
      fPainter->CanvasUpdated(fModified, true, nullptr); // trigger async display
   }
}

//////////////////////////////////////////////////////////////////////////
/// Close all canvas displays

void ROOT::Experimental::TCanvas::Hide()
{
   if (fPainter)
      delete fPainter.release();
}

void ROOT::Experimental::TCanvas::SaveAs(const std::string &filename, bool async, CanvasCallback_t callback)
{
   if (!fPainter)
      fPainter = Internal::TVirtualCanvasPainter::Create(*this, true);
   if (filename.find(".svg") != std::string::npos)
      fPainter->DoWhenReady("SVG", filename, async, callback);
   else if (filename.find(".png") != std::string::npos)
      fPainter->DoWhenReady("PNG", filename, async, callback);
   else if ((filename.find(".jpg") != std::string::npos) || (filename.find(".jpeg") != std::string::npos))
      fPainter->DoWhenReady("JPEG", filename, async, callback);
}

void ROOT::Experimental::TCanvas::RegisterColor(size_t &idx, const TColor &col)
{
   if (idx == (size_t)-1) {
      idx = fColorTable.size();
      fColorTable.push_back(col);
   } else {
      fColorTable[idx] = col;
   }
}

// TODO: removal from GetHeldCanvases().

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::Clear()
{
   if (fUseCount) {
      R__ERROR_HERE("Gpad") << "Refusing to clear a referenced primitive (use count " << fUseCount << ")!";
      return;
   }
   // destroy fVal:
   fVal.~PRIMITIVE();
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::Create(const PRIMITIVE &val)
{
   if (fUseCount) {
      R__ERROR_HERE("Gpad") << "Refusing to create a primitive over an existing one (use count " << fUseCount << ")!";
      return;
   }
   // copy-construct fVal:
   new (fVal) PRIMITIVE(val);
   fUseCount = 1;
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::IncrUse(const PRIMITIVE &val)
{
   if (fUseCount == 0) {
      R__ERROR_HERE("Gpad") << "Refusing to increase use count on a non-existing primitive!";
      return;
   }
   ++fUseCount;
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::DecrUse(const PRIMITIVE &val)
{
   if (fUseCount == 0) {
      R__ERROR_HERE("Gpad") << "Refusing to decrease use count on a non-existing primitive!";
      return;
   }
   --fUseCount;
   if (fUseCount == 0)
      Clear();
}

// Available specialization:
template class ROOT::Experimental::Internal::TOptsAttrAndUseCount<TColor>;
template class ROOT::Experimental::Internal::TOptsAttrAndUseCount<long long>;
template class ROOT::Experimental::Internal::TOptsAttrAndUseCount<double>;

template <class PRIMITIVE>
size_t TOptsAttrTable::Register(const PRIMITIVE &val)
{
   auto isFree = [](const value_type &el) -> bool { return el.IsFree() };
   auto iSlot = std::find_if(fTable.begin(), fTable.last(), isFree);
   if (iSlot != fTable.end()) {
      iSlot.Create(val);
      return iSlot - fTable.begin();
   }
   fTable.emplace_back(val);
   return fTable.size() - 1;
}

// Available specialization:
template class ROOT::Experimental::Internal::TOptsAttrTable<TColor>;
template class ROOT::Experimental::Internal::TOptsAttrTable<long long>;
template class ROOT::Experimental::Internal::TOptsAttrTable<double>;

template ROOT::Experimental::Internal::TOptsAttrTable<TColor> &ROOT::Experimental::TCanvas::GetAttrTable<TColor>()
{
   return fColorTable;
}
template ROOT::Experimental::Internal::TOptsAttrTable<long long> &ROOT::Experimental::TCanvas::GetAttrTable<long long>()
{
   return fIntAttrTable;
}
template ROOT::Experimental::Internal::TOptsAttrTable<double> &ROOT::Experimental::TCanvas::GetAttrTable<double>()
{
   return fFPAttrTable;
}

template const ROOT::Experimental::Internal::TOptsAttrTable<TColor> &
ROOT::Experimental::TCanvas::GetAttrTable<TColor>() const
{
   return fColorTable;
}
template const ROOT::Experimental::Internal::TOptsAttrTable<long long> &
ROOT::Experimental::TCanvas::GetAttrTable<long long>() const
{
   return fIntAttrTable;
}
template const ROOT::Experimental::Internal::TOptsAttrTable<double> &
ROOT::Experimental::TCanvas::GetAttrTable<double>() const
{
   return fFPAttrTable;
}
