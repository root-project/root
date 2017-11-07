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

#include "ROOT/TDrawingAttrs.hxx"

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TLogger.hxx"

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
using ParsedAttrs_t = std::unordered_map<std::string, TDrawingAttrRef<PRIMITIVE>>;
using AllParsedAttrs_t = std::tuple<ParsedAttrs_t<TColor>, ParsedAttrs_t<long long>, ParsedAttrs_t<double>>;

static AllParsedAttrs_t &GetParsedDefaultAttrs()
{
   static AllParsedAttrs_t sAllParsedAttrs;
   return sAllParsedAttrs;
}

static ParsedAttrs_t<TColor> &GetParsedDefaultAttrsOfAKind(TColor *)
{
   return std::get<0>(GetParsedDefaultAttrs());
}

static ParsedAttrs_t<long long> &GetParsedDefaultAttrsOfAKind(long long *)
{
   return std::get<1>(GetParsedDefaultAttrs());
}
static ParsedAttrs_t<double> &GetParsedDefaultAttrsOfAKind(double *)
{
   return std::get<2>(GetParsedDefaultAttrs());
}

} // unnamed namespace

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE>::TDrawingAttrRef(TDrawingOptsBaseNoDefault &opts, const std::string &attrName,
   const PRIMITIVE &deflt, const std::vector<std::string_view> &optStrings)
{
   std::string fullName = opts.GetName() + "." + attrName;
   static constexpr PRIMITIVE *kNullPtr = (PRIMITIVE *)nullptr;
   auto &parsedAttrs = GetParsedDefaultAttrsOfAKind(kNullPtr);
   TCanvas &canv = opts.GetCanvas();
   if (TPadDrawingOpts::IsDefaultCanvas(canv)) {
      // We are a member of the default option object.
      auto iIdx = parsedAttrs.find(fullName);
      if (iIdx == parsedAttrs.end()) {
         // We haven't read the default yet, do that now:
         PRIMITIVE val = Internal::TDrawingOptsReader(GetDefaultAttrConfig()).Parse(fullName, deflt, optStrings);
         fIdx = opts.Register(val);
         parsedAttrs[fullName] = *this;
      } else {
         fIdx = opts.SameAs(iIdx->second);
      }
   } else {
      auto &defCanv = static_cast<TCanvas &>(opts.GetDefaultCanvas(TStyle::GetCurrent()));
      const auto &defaultTable = defCanv.GetAttrTable((PRIMITIVE *)nullptr);
      PRIMITIVE val = defaultTable.Get(parsedAttrs[fullName]);
      fIdx = opts.Register(val);
   }
}

namespace ROOT {
namespace Experimental {
template class TDrawingAttrRef<TColor>;
template class TDrawingAttrRef<long long>;
template class TDrawingAttrRef<double>;
} // namespace Experimental
} // namespace ROOT

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TDrawingAttrAndUseCount<PRIMITIVE>::Clear()
{
   if (fUseCount) {
      R__ERROR_HERE("Gpad") << "Refusing to clear a referenced primitive (use count " << fUseCount << ")!";
      return;
   }
   // destroy fVal:
   fVal.~PRIMITIVE();
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TDrawingAttrAndUseCount<PRIMITIVE>::Create(const PRIMITIVE &val)
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
void ROOT::Experimental::Internal::TDrawingAttrAndUseCount<PRIMITIVE>::IncrUse()
{
   if (fUseCount == 0) {
      R__ERROR_HERE("Gpad") << "Refusing to increase use count on a non-existing primitive!";
      return;
   }
   ++fUseCount;
}

template <class PRIMITIVE>
void ROOT::Experimental::Internal::TDrawingAttrAndUseCount<PRIMITIVE>::DecrUse()
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
template class ROOT::Experimental::Internal::TDrawingAttrAndUseCount<TColor>;
template class ROOT::Experimental::Internal::TDrawingAttrAndUseCount<long long>;
template class ROOT::Experimental::Internal::TDrawingAttrAndUseCount<double>;

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE> Internal::TDrawingAttrTable<PRIMITIVE>::Register(const PRIMITIVE &val)
{
   auto isFree = [](const value_type &el) -> bool { return el.IsFree(); };
   auto iSlot = std::find_if(fTable.begin(), fTable.end(), isFree);
   if (iSlot != fTable.end()) {
      iSlot->Create(val);
      std::ptrdiff_t offset = iSlot - fTable.begin();
      assert(offset >= 0 && "This offset cannot possibly be negative!");
      return TDrawingAttrRef<PRIMITIVE>{static_cast<size_t>(offset)};
   }
   fTable.emplace_back(val);
   return TDrawingAttrRef<PRIMITIVE>{fTable.size() - 1};
}

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE> Internal::TDrawingAttrTable<PRIMITIVE>::SameAs(const PRIMITIVE &val)
{
   if (&fTable.front().Get() > &val || &fTable.back().Get() < &val)
      return TDrawingAttrRef<PRIMITIVE>{}; // not found.
   std::ptrdiff_t offset = &val - &fTable.front().Get();
   assert(offset >= 0 && "Logic error, how can offset be < 0?");
   TDrawingAttrRef<PRIMITIVE> idx{static_cast<size_t>(offset)};
   IncrUse(idx);
   return TDrawingAttrRef<PRIMITIVE>{idx};
}

// Provide specializations promised by the header:
template class Internal::TDrawingAttrTable<TColor>;
template class Internal::TDrawingAttrTable<long long>;
template class Internal::TDrawingAttrTable<double>;

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE>
TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Register(TCanvas &canv, const PRIMITIVE &val)
{
   fRefArray.push_back(canv.GetAttrTable((PRIMITIVE *)nullptr).Register(val));
   return fRefArray.back();
}

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE>
TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx)
{
   canv.GetAttrTable((PRIMITIVE *)nullptr).IncrUse(idx);
   return idx;
}

template <class PRIMITIVE>
TDrawingAttrRef<PRIMITIVE>
TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas &canv, const PRIMITIVE &val)
{
   return canv.GetAttrTable((PRIMITIVE *)nullptr).SameAs(val);
}

template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Update(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx,
                                                                  const PRIMITIVE &val)
{
   canv.GetAttrTable((PRIMITIVE *)nullptr).Update(idx, val);
}

template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Release(TCanvas &canv)
{
   for (auto idx: fRefArray)
      canv.GetAttrTable((PRIMITIVE *)nullptr).DecrUse(idx);
   fRefArray.clear();
}

template <class PRIMITIVE>
void TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::RegisterCopy(TCanvas &canv)
{
   for (auto idx: fRefArray)
      canv.GetAttrTable((PRIMITIVE *)nullptr).IncrUse(idx);
}

template <class PRIMITIVE>
const PRIMITIVE &TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Get(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx) const
{
   return canv.GetAttrTable((PRIMITIVE *)nullptr).Get(idx);
}

template <class PRIMITIVE>
PRIMITIVE &TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::Get(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx)
{
   return canv.GetAttrTable((PRIMITIVE *)nullptr).Get(idx);
}

template <class PRIMITIVE>
TDrawingOptsBaseNoDefault::OptsAttrRefArr<PRIMITIVE>::~OptsAttrRefArr()
{
   if (!fRefArray.empty())
      R__ERROR_HERE("Gpad") << "Drawing attributes table not empty - must call Release() before!";
}

// Provide specializations promised by the header:
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<TColor>;
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<long long>;
template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<double>;
