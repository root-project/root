/// \file RHistDrawable.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"

#include "ROOT/RHistImpl.hxx"

#include "TSystem.h" // RSystem::Load

#include <cassert>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

void ROOT::Experimental::Internal::LoadHistPainterLibrary()
{
   gSystem->Load("libHistPainter");
}

template <int DIMENSION>
RHistPainterBase<DIMENSION>::RHistPainterBase()
{
   GetPainterPtr() = this;
}

template <int DIMENSION>
RHistPainterBase<DIMENSION>::~RHistPainterBase()
{
   GetPainterPtr() = nullptr;
}

template <int DIMENSION>
RHistPainterBase<DIMENSION> *&RHistPainterBase<DIMENSION>::GetPainterPtr()
{
   static RHistPainterBase<DIMENSION> *painter = nullptr;

   return painter;
}

template <int DIMENSION>
RHistPainterBase<DIMENSION> *RHistPainterBase<DIMENSION>::GetPainter()
{
   // Trigger loading of the painter library within the init guard of the static:
   static int triggerLibLoad = (LoadHistPainterLibrary(), 0);

   (void)triggerLibLoad; // unused.

   return GetPainterPtr();
}

// GCC 5 needs to have that outlined - is that a compiler bug?
template <int DIMENSIONS>
RHistDrawable<DIMENSIONS>::RHistDrawable() : RDrawable("hist") {}

/// Paint the histogram
template <int DIMENSIONS>
void RHistDrawable<DIMENSIONS>::Paint(Internal::RPadPainter &pad)
{
   Internal::RHistPainterBase<DIMENSIONS>::GetPainter()->Paint(*this, pad);
}


namespace ROOT {
namespace Experimental {

namespace Internal {
template class RHistPainterBase<1>;
template class RHistPainterBase<2>;
template class RHistPainterBase<3>;
} // namespace Internal

template class RHistDrawable<1>;
template class RHistDrawable<2>;
template class RHistDrawable<3>;

std::shared_ptr<RHistDrawable<1>> GetDrawable(const std::shared_ptr<RH1D> &histimpl)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

std::shared_ptr<RHistDrawable<1>> GetDrawable(const std::shared_ptr<RH1I> &histimpl)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

std::shared_ptr<RHistDrawable<1>> GetDrawable(const std::shared_ptr<RH1C> &histimpl)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

std::shared_ptr<RHistDrawable<1>> GetDrawable(const std::shared_ptr<RH1F> &histimpl)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

std::shared_ptr<RHistDrawable<2>> GetDrawable(const std::shared_ptr<RH2D> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

std::shared_ptr<RHistDrawable<2>> GetDrawable(const std::shared_ptr<RH2I> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

std::shared_ptr<RHistDrawable<2>> GetDrawable(const std::shared_ptr<RH2C> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

std::shared_ptr<RHistDrawable<2>> GetDrawable(const std::shared_ptr<RH2F> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

std::shared_ptr<RHistDrawable<3>> GetDrawable(const std::shared_ptr<RH3D> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

std::shared_ptr<RHistDrawable<3>> GetDrawable(const std::shared_ptr<RH3I> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

std::shared_ptr<RHistDrawable<3>> GetDrawable(const std::shared_ptr<RH3C> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

std::shared_ptr<RHistDrawable<3>> GetDrawable(const std::shared_ptr<RH3F> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

} // namespace Experimental
} // namespace ROOT
