/// \file ROOT/RTreeMapBase.hxx
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

#ifndef RTREEMAPBASE_HXX
#define RTREEMAPBASE_HXX

#include <cstdint>
#include <string>
#include <vector>

namespace ROOT::Experimental {

// clang-format off
/**
\class ROOT::Experimental::RTreeMapBase
\ingroup TreeMap
\brief Base logic for drawing a treemap visualization

A treemap can be used for analyzing a hierarchical data structure whose elements have a certain size. It visualizes this
hierarchical data as nested rectangles which allows for easy comparison of proportions within categories, but
also the whole structure. The squarification algorithm is used to make these rectangles as close to squares as
possible for visual clarity.

Furthermore, we assume that each node has a type and that the size of a non-leaf node equals to the total size of its children. This
allows for drawing a legend of types of leaf nodes, and see which types occupy how much of the total space.

Note: this visualization class/technique is independent/unrelated to `TTree`.
*/
// clang-format on
class RTreeMapBase {
public:
   struct Node {
      std::string fName, fType;
      uint64_t fSize;
      uint64_t fChildrenIdx;
      uint64_t fNChildren;
      Node() = default;
      Node(const std::string &name, const std::string &type, uint64_t size, uint64_t childrenIdx, uint64_t nChildren)
         : fName(name), fType(type), fSize(size), fChildrenIdx(childrenIdx), fNChildren(nChildren)
      {
      }
   };

   struct Vec2 {
      float x, y;
      Vec2(float xArg, float yArg) : x(xArg), y(yArg) {}
   };
   struct Rect {
      Vec2 fBottomLeft, fTopRight;
      Rect(const Vec2 &bottomLeftArg, const Vec2 &topRightArg) : fBottomLeft(bottomLeftArg), fTopRight(topRightArg) {}
   };
   struct RGBColor {
      uint8_t r, g, b, a;
      RGBColor(uint8_t rArg, uint8_t gArg, uint8_t bArg, uint8_t aArg = 255) : r(rArg), g(gArg), b(bArg), a(aArg) {}
   };
   std::vector<Node> fNodes;
   RTreeMapBase() = default;
   virtual ~RTreeMapBase() = default;

protected:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing the entirety of the treemap.
   void DrawTreeMap(const Node &elem, Rect rect, int depth) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing the legend of leaf types
   void DrawLegend() const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing a box
   virtual void AddBox(const Rect &rect, const RGBColor &color, float borderWidth = 0.15f) const = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Logic for drawing a text
   virtual void AddText(const Vec2 &pos, const std::string &content, float size,
                        const RGBColor &color = RGBColor(0, 0, 0), bool alignCenter = false) const = 0;
};
} // namespace ROOT::Experimental
#endif