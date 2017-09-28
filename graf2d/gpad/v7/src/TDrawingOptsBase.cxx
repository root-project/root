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

std::map<std::string, std::string> Internal::ReadDrawingOptsDefaultConfig(std::string_view /*section*/)
{
   assert(0 && "Not yet implemented!");
}

Internal::TDrawingOptsBase::TDrawingOptsBase(TPadBase &pad, const Attrs& attrs)
   : fCanvas(pad.GetCanvas())
{
   for (auto&& a: attrs.fCols)
      a.Init(*this);
   for (auto&& a: attrs.fInts)
      a.Init(*this);
   for (auto&& a: attrs.fFPs)
      a.Init(*this);
}

ROOT::Experimental::TCanvas &Internal::TDrawingOptsBase::GetDefaultCanvas()
{
   static TCanvas sCanv;
   return sCanv;
}

Internal::TDrawingOptsBase::~TDrawingOptsBase()
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

template <class PRIMITIVE>
TOptsAttrIdx<PRIMITIVE> Internal::TDrawingOptsBase<PRIMITIVE>::Register(const PRIMITIVE &col)
{
   GetIndexVec<PRIMITIVE>().push_back(fCanvas.GetAttrTable<PRIMITIVE>().Register(col));
   return GetIndexVec<PRIMITIVE>().back();
}

template TOptsAttrIdx<TColor> Internal::TDrawingOptsBase<TColor>::Register(const TColor &col);
template TOptsAttrIdx<long long> Internal::TDrawingOptsBase<long long>::Register(const long long &col);
template TOptsAttrIdx<double> Internal::TDrawingOptsBase<double>::Register(const double &col);

template <class PRIMITIVE>
void Internal::TDrawingOptsBase<PRIMITIVE>::Update(TOptsAttrIdx<PRIMITIVE> idx, const PRIMITIVE &val)
{
   fCanvas.GetAttrTable<PRIMITIVE>().Update(idx, val);
}

template TOptsAttrIdx<TColor> Internal::TDrawingOptsBase::Register(const TColor &);
template TOptsAttrIdx<long long> Internal::TDrawingOptsBase::Register(const long long &);
template TOptsAttrIdx<double> Internal::TDrawingOptsBase::Register(const double &);

void Internal::TDrawingOptsBase::Init(TDrawingOptsBase &opts)
{
   fIdxMemRef = ops.Register(GetDefaultCanvas().GetAttrTable<PRIMITIVE>().Get(fDefaultIdx));
}

template void Internal::TDrawingOptsBase<TColor>::Init(TDrawingOptsBase &opts);
template void Internal::TDrawingOptsBase<long long>::Init(TDrawingOptsBase &opts);
template void Internal::TDrawingOptsBase<double>::Init(TDrawingOptsBase &opts);

template class Internal::TDrawingOptsBase::AttrIdxAndDefault<TColor>;
template class Internal::TDrawingOptsBase::AttrIdxAndDefault<long long>;
template class Internal::TDrawingOptsBase::AttrIdxAndDefault<double>;
