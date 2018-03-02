/// \file THistDrawable.cxx
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

#include "ROOT/THistDrawable.hxx"

#include "ROOT/THistImpl.hxx"

#include "TSystem.h" // TSystem::Load

#include <cassert>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

void ROOT::Experimental::Internal::LoadHistPainterLibrary()
{
   gSystem->Load("libHistPainter");
}

template <int DIMENSION>
THistPainterBase<DIMENSION>::THistPainterBase()
{
   GetPainterPtr() = this;
}

template <int DIMENSION>
THistPainterBase<DIMENSION>::~THistPainterBase()
{
   GetPainterPtr() = nullptr;
}

template <int DIMENSION>
THistPainterBase<DIMENSION> *&THistPainterBase<DIMENSION>::GetPainterPtr()
{
   static THistPainterBase<DIMENSION> *painter = nullptr;

   return painter;
}

template <int DIMENSION>
THistPainterBase<DIMENSION> *THistPainterBase<DIMENSION>::GetPainter()
{
   // Trigger loading of the painter library within the init guard of the static:
   static int triggerLibLoad = (LoadHistPainterLibrary(), 0);

   (void)triggerLibLoad; // unused.

   return GetPainterPtr();
}

template <class DERIVED>
void THistDrawableBase<DERIVED>::PopulateMenu(TMenuItems &)
{
   // here should be filling of context menu for the given object
}

// GCC 5 needs to have that outlined - is that a compiler bug?
template <int DIMENSIONS>
THistDrawable<DIMENSIONS>::THistDrawable() = default;

/// Paint the histogram
template <int DIMENSIONS>
void THistDrawable<DIMENSIONS>::Paint(Internal::TVirtualCanvasPainter &canv)
{
   Internal::THistPainterBase<DIMENSIONS>::GetPainter()->Paint(*this, fOpts, canv);
}

namespace ROOT {
namespace Experimental {

namespace Internal {
template class THistPainterBase<1>;
template class THistPainterBase<2>;
template class THistPainterBase<3>;
} // namespace Internal

template class THistDrawableBase<THistDrawable<1>>;
template class THistDrawableBase<THistDrawable<2>>;
template class THistDrawableBase<THistDrawable<3>>;

template class THistDrawable<1>;
template class THistDrawable<2>;
template class THistDrawable<3>;
} // namespace Experimental
} // namespace ROOT
