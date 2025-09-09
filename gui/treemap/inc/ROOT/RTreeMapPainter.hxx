/// \file ROOT/RTreeMapPainter.hxx
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

#ifndef TTREEMAP_HXX
#define TTREEMAP_HXX

#include "RTreeMapBase.hxx"

#include "TROOT.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TPad.h"

#include <vector>

namespace ROOT::Experimental {
class RNTupleInspector;

// clang-format off
/**
\class ROOT::Experimental::RTreeMapPainter
\ingroup TreeMap
\brief Logic for drawing a treemap on a TVirtualPad

One can visualize an RNTuple in a TCanvas as a treemap like this:
~~~ {.cpp}
auto tm = RTreeMapPainter::ImportRNTuple("file.root", "ntuple_name");
auto c = new TCanvas("c_tm","TreeMap");
c->Add(tm.release());
~~~
*/
// clang-format on
class RTreeMapPainter final : public ROOT::Experimental::RTreeMapBase, public TObject {
public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for converting an RNTuple to RTreeMapPainter given RNTupleInspector
   static std::unique_ptr<RTreeMapPainter> ImportRNTuple(const ROOT::Experimental::RNTupleInspector &insp);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for converting an RNTuple to RTreeMapPainter given file and tuple names
   static std::unique_ptr<RTreeMapPainter> ImportRNTuple(std::string_view sourceFileName, std::string_view tupleName);

   struct Node final : public ROOT::Experimental::RTreeMapBase::Node, public TObject {
   public:
      ClassDefOverride(Node, 1);
   };
   RTreeMapPainter() = default;
   void Paint(Option_t *opt) override;

   ClassDefOverride(RTreeMapPainter, 1);

   ~RTreeMapPainter() override = default;

private:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing a box on TVirtualPad
   void AddBox(const Rect &rect, const RGBColor &color, float borderWidth) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing a text on TVirtualPad
   void AddText(const Vec2 &pos, const std::string &content, float size, const RGBColor &color = RGBColor(0, 0, 0),
                bool alignCenter = false) const final;
};
} // namespace ROOT::Experimental
#endif