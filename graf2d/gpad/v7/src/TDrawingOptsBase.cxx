/// \file TDrawingOptsBase.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDrawingOptsBase.hxx"

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TLogger.hxx"
#include "ROOT/TPad.hxx"

#include "TDrawingOptsReader.hxx" // in src/

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <unordered_map>

using namespace ROOT::Experimental;

namespace {

/// The default attribute values for drawing options.
static Internal::TDrawingOptsReader::Attrs_t &GetDefaultAttrConfig()
{
   static Internal::TDrawingOptsReader::Attrs_t sDefaults = Internal::TDrawingOptsReader::ReadDefaults();
   return sDefaults;
}

template <class PRIMITIVE>
using ParsedAttrs_t = std::unordered_map<std::string, TOptsAttrRef<PRIMITIVE>>;
using AllParsedAttrs_t = std::tuple<ParsedAttrs_t<TColor>, ParsedAttrs_t<long long>, ParsedAttrs_t<double>>;

static AllParsedAttrs_t &GetParsedDefaultAttrs()
{
   static AllParsedAttrs_t sAllParsedAttrs;
   return sAllParsedAttrs;
}

static ParsedAttrs_t<TColor> &GetParsedDefaultAttrsOfAKind(TColor*) {
   return std::get<0>(GetParsedDefaultAttrs());
}

static ParsedAttrs_t<long long> &GetParsedDefaultAttrsOfAKind(long long*) {
   return std::get<1>(GetParsedDefaultAttrs());
}
static ParsedAttrs_t<double> &GetParsedDefaultAttrsOfAKind(double*) {
   return std::get<2>(GetParsedDefaultAttrs());
}

} // unnamed namespace

template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE>::TOptsAttrRef(TDrawingOptsBaseNoDefault &opts, std::string_view name, const std::vector<std::string_view>& optStrings)
{
   static constexpr PRIMITIVE* kNullPtr = (PRIMITIVE*)nullptr;
   auto &parsedAttrs = GetParsedDefaultAttrsOfAKind(kNullPtr);
   TCanvas &canv = opts.GetCanvas();
   std::string strName(name);
   if (&canv == &opts.GetDefaultCanvas()) {
      // We are a member of the default option object.
      auto iIdx = parsedAttrs.find(strName);
      if (iIdx == parsedAttrs.end()) {
         // We haven't read the default yet, do that now:
         PRIMITIVE val = Internal::TDrawingOptsReader(GetDefaultAttrConfig()).Parse(name, kNullPtr, optStrings);
         fIdx = opts.Register(val);
         parsedAttrs[strName] = *this;
      } else {
         fIdx = opts.SameAs(iIdx->second);
      }
   } else {
      auto &defCanv = static_cast<TCanvas&>(opts.GetDefaultCanvas());
      const auto &defaultTable = defCanv.GetAttrTable((PRIMITIVE*)nullptr);
      PRIMITIVE val = defaultTable.Get(parsedAttrs[strName]);
      fIdx = opts.Register(val);
   }
}

namespace ROOT {
namespace Experimental {
template class TOptsAttrRef<TColor>;
template class TOptsAttrRef<long long>;
template class TOptsAttrRef<double>;
} // namespace Experimental
} // namespace ROOT

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
   new (&fVal) PRIMITIVE(val);
   fUseCount = 1;
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::IncrUse()
{
   if (fUseCount == 0) {
      R__ERROR_HERE("Gpad") << "Refusing to increase use count on a non-existing primitive!";
      return;
   }
   ++fUseCount;
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TOptsAttrAndUseCount<PRIMITIVE>::DecrUse()
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
TOptsAttrRef<PRIMITIVE> Internal::TOptsAttrTable<PRIMITIVE>::Register(const PRIMITIVE &val)
{
   auto isFree = [](const value_type &el) -> bool { return el.IsFree(); };
   auto iSlot = std::find_if(fTable.begin(), fTable.end(), isFree);
   if (iSlot != fTable.end()) {
      iSlot->Create(val);
      std::ptrdiff_t offset = iSlot - fTable.begin();
      assert(offset >= 0 && "This offset cannot possibly be negative!");
      return TOptsAttrRef<PRIMITIVE>{static_cast<size_t>(offset)};
   }
   fTable.emplace_back(val);
   return TOptsAttrRef<PRIMITIVE>{fTable.size() - 1};
}

template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE> Internal::TOptsAttrTable<PRIMITIVE>::SameAs(const PRIMITIVE &val)
{
   if (&fTable.front().Get() > &val || &fTable.back().Get() < &val)
      return TOptsAttrRef<PRIMITIVE>{}; // not found.
   std::ptrdiff_t offset = &val - &fTable.front().Get();
   assert(offset >= 0 && "Logic error, how can offset be < 0?");
   TOptsAttrRef<PRIMITIVE> idx{static_cast<size_t>(offset)};
   IncrUse(idx);
   return TOptsAttrRef<PRIMITIVE>{idx};
}

// Provide specializations promised by the header:
template class Internal::TOptsAttrTable<TColor>;
template class Internal::TOptsAttrTable<long long>;
template class Internal::TOptsAttrTable<double>;


template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE> TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Register(TCanvas& canv, const PRIMITIVE &val)
{
   fRefArray.push_back(canv.GetAttrTable((PRIMITIVE*)nullptr).Register(val));
   return fRefArray.back();
}

template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE> TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas& canv, TOptsAttrRef<PRIMITIVE> idx)
{
   canv.GetAttrTable((PRIMITIVE*)nullptr).IncrUse(idx);
   return idx;
}

template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE> TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas& canv, const PRIMITIVE &val) {
   return canv.GetAttrTable((PRIMITIVE*)nullptr).SameAs(val);
}


template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Update(TCanvas &canv, TOptsAttrRef<PRIMITIVE> idx, const PRIMITIVE &val)
{
   canv.GetAttrTable((PRIMITIVE*)nullptr).Update(idx, val);
}

template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Release(TCanvas &canv) {
   for (auto idx: fRefArray)
      canv.GetAttrTable((PRIMITIVE*)nullptr).DecrUse(idx);
   fRefArray.clear();
}

template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::RegisterCopy(TCanvas &canv) {
   for (auto idx: fRefArray)
      canv.GetAttrTable((PRIMITIVE*)nullptr).IncrUse(idx);
}

template <class PRIMITIVE>
TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::~OptsAttrRefArr() {
   if (!fRefArray.empty())
      R__ERROR_HERE("Gpad") << "Drawing attributes table not empty - must call Release() before!";
}

// Provide specializations promised by the header:
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<TColor>;
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<long long>;
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<double>;


TDrawingOptsBaseNoDefault::TDrawingOptsBaseNoDefault(TPadBase &pad): fCanvas(&pad.GetCanvas())
{
}

ROOT::Experimental::TPadBase &TDrawingOptsBaseNoDefault::GetDefaultCanvas()
{
   static TCanvas sCanv;
   return sCanv;
}

TDrawingOptsBaseNoDefault::~TDrawingOptsBaseNoDefault()
{
   fColorIdx.Release(GetCanvas());
   fIntIdx.Release(GetCanvas());
   fFPIdx.Release(GetCanvas());
}

TDrawingOptsBaseNoDefault::TDrawingOptsBaseNoDefault(const TDrawingOptsBaseNoDefault &other):
fCanvas(other.fCanvas),
fColorIdx(other.fColorIdx),
fIntIdx(other.fIntIdx),
fFPIdx(other.fFPIdx)
{
   fColorIdx.RegisterCopy(GetCanvas());
   fIntIdx.RegisterCopy(GetCanvas());
   fFPIdx.RegisterCopy(GetCanvas());
}
