/// \file TFrame.cxx
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

#include <cassert>

using namespace ROOT::Experimental;

namespace {

/// The default attribute values for drawing options.
static TDrawingOptsReader::DefaultAttrs_t &GetDefaultAttrConfig()
{
   static TDrawingOptsReader::DefaultAttrs_t sDefaults = TDrawingOptsReader::ReadDefaults();
}

template <class PRIMITIVE>
using ParsedAttrs_t = std::unordered_map<std::string, TOptsAttrRef<T>>;
using AllParsedAttrs_t = std::tuple<ParsedAttrs_t<TColor>, ParsedAttrs_t<long long>, ParsedAttrs_t<double>>;

static AllParsedAttrs_t &GetParsedDefaultAttrs()
{
   static AllParsedAttrs_t &sAllParsedAttrs;
   return sAllParsedAttrs;
}

} // unnamed namespace

template <class PRIMITIVE>
TOptsAttrRef::TOptsAttrRef(TDrawingOptsBase &opts, std::string_ref name, std::vector<std::string_view> opts)
{
   TCanvas &canv = opts.GetCanvas();
   if (&canv == &opts.GetDefaultCanvas()) {
      // We are a member of the default option object.
      auto iIdx = GetParsedDefaultAttrs().find(name);
      if (iIdx == GetParsedDefaultAttrs().end()) {
         // We haven't read the default yet, do that now:
         PRIMITIVE val = TDrawingOptsReader(GetDefaultAttrConfig()).Parse(name, (const PRIMITIVE *)nullptr, opts);
         fIdx = opts.Register(val);
         GetParsedDefaultAttrs()[name] = fIdx;
      } else {
         fIdx = opts.SameAs(iIdx);
      }
   } else {
      const auto &defaultTable = opts.GetDefaultCanvas().GetAttrTable<PRIMITIVE>();
      PRIMITIVE val = defaultTable.Get(GetParsedDefaultAttrs()[name]);
      fIdx = opts.Register(val);
   }
}

template <class PRIMITIVE>
TOptsAttrIdx<PRIMITIVE> Internal::TDrawingOptsBase::OptsAttrRefArr<PRIMITIVE>::Register(TCanvas& canv, const PRIMITIVE &val)
{
   fRefArray.push_back(canv.GetAttrTable<PRIMITIVE>().Register(col));
   return GetIndexVec<PRIMITIVE>().back();
}

template <class PRIMITIVE>
TOptsAttrIdx<PRIMITIVE> Internal::TDrawingOptsBase::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas& canv, TOptsAttrIdx<PRIMITIVE> idx)
{
   canv.GetAttrTable<PRIMITIVE>().IncrUse(idx));
   return idx;
}

template <class PRIMITIVE>
TOptsAttrRef<PRIMITIVE> Internal::TDrawingOptsBase::OptsAttrRefArr<PRIMITIVE>::SameAs(TCanvas& canv, const PRIMITIVE &val) {
   return canv.GetAttrTable<PRIMITIVE>().SameAs(val);
}


template <class PRIMITIVE>
void Internal::TDrawingOptsBase::OptsAttrRefArr<PRIMITIVE>::Update(TOptsAttrIdx<PRIMITIVE> idx, const PRIMITIVE &val)
{
   fCanvas.GetAttrTable<PRIMITIVE>().Update(idx, val);
}

// Provide specializations promised by the header:
template class TDrawingOptsBase::OptsAttrRefArr<TColor>;
template class TDrawingOptsBase::OptsAttrRefArr<long long>;
template class TDrawingOptsBase::OptsAttrRefArr<double>;


Internal::TDrawingOptsBase::TDrawingOptsBase(TPadBase &pad): fCanvas(pad.GetCanvas())
{
}

ROOT::Experimental::TCanvas &Internal::TDrawingOptsBase::GetDefaultCanvas()
{
   static TCanvas sCanv;
   return sCanv;
}

namespace Internal {
   struct TDrawingOptsBase;
}

DrawingOptsBase::~TDrawingOptsBase()
{
   for (auto idx: fColorIdx)
      fCanvas.GetColorAttrTable().DecrUse(idx);
   for (auto idx: fIntIdx)
      fCanvas.GetIntAttrTable().DecrUse(idx);
   for (auto idx: fFPIdx)
      fCanvas.GetFPAttrTable().DecrUse(idx);
}

// Available specializations
template std::vector<TOptsAttrIdx<TColor>> &Internal::TDrawingOptsBase::GetIndexVec<TColor>()
{
   return fColorIdx;
}
template std::vector<TOptsAttrIdx<long long>> &Internal::TDrawingOptsBase::GetIndexVec<long long>()
{
   return fIntIdx;
}
template std::vector<TOptsAttrIdx<double>> &Internal::TDrawingOptsBase::GetIndexVec<double>()
{
   return fFPIdx;
}

template const std::vector<TOptsAttrIdx<TColor>> &Internal::TDrawingOptsBase::GetIndexVec<TColor>() const
{
   return fColorIdx;
}
template const std::vector<TOptsAttrIdx<long long>> &Internal::TDrawingOptsBase::GetIndexVec<long long>() const
{
   return fIntIdx;
}
template const std::vector<TOptsAttrIdx<double>> &Internal::TDrawingOptsBase::GetIndexVec<double>() const
{
   return fFPIdx;
}
