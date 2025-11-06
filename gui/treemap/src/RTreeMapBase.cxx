/// \file RTreeMapBase.cxx
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

#include <ROOT/RTreeMapBase.hxx>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <algorithm>

static constexpr float kIndentationOffset = 0.005f;
static constexpr float kPadTextOffset = 0.004f;
static constexpr float kTextSizeFactor = 0.009f;
static constexpr const char *kUnits[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB"};

using namespace ROOT::Experimental;

static uint64_t ComputeFnv(const std::string &str)
{
   uint64_t h = 14695981039346656037ULL;
   for (char c : str)
      h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ULL;
   return h;
}

static RTreeMapBase::RGBColor ComputeColor(const std::string &str)
{
   const uint64_t hash = ComputeFnv(str);
   return RTreeMapBase::RGBColor((hash >> 16) & 0xFF, (hash >> 8) & 0xFF, hash & 0xFF);
}

static std::string GetFloatStr(const float &n, const uint8_t &precision)
{
   std::stringstream stream;
   stream << std::fixed << std::setprecision(precision) << n;
   return stream.str();
}

static std::string GetDataStr(uint64_t bytes)
{
   const uint64_t order = std::log10(bytes) / 3.0f;
   const std::string unit = kUnits[order];
   const float finalSize = static_cast<float>(bytes) / std::pow(1000, order);
   return GetFloatStr(finalSize, 2) + unit;
}

static std::vector<std::pair<std::string, uint64_t>> GetDiskOccupation(const std::vector<RTreeMapBase::Node> &nodes)
{
   std::unordered_map<std::string, uint64_t> acc;
   for (const auto &node : nodes) {
      if (node.fNChildren > 0)
         continue;
      acc[node.fType] += node.fSize;
   }

   std::vector<std::pair<std::string, uint64_t>> vec;
   vec.reserve(acc.size());
   for (auto &p : acc)
      vec.emplace_back(std::move(p.first), p.second);

   std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
   return vec;
}

/* algorithm: https://vanwijk.win.tue.nl/stm.pdf */
static float ComputeWorstRatio(const std::vector<RTreeMapBase::Node> &row, float width, float height,
                               uint64_t totalSize, bool horizontalRows)
{
   if (row.empty())
      return 0.0f;
   uint64_t sumRow = 0;
   for (const auto &child : row)
      sumRow += child.fSize;
   if (sumRow == 0)
      return 0.0f;
   float worstRatio = 0.0f;
   for (const auto &child : row) {
      float ratio = horizontalRows ? static_cast<float>(child.fSize * width * totalSize) / (sumRow * sumRow * height)
                                   : static_cast<float>(child.fSize * height * totalSize) / (sumRow * sumRow * width);
      float aspectRatio = std::max(ratio, 1.0f / ratio);
      if (aspectRatio > worstRatio)
         worstRatio = aspectRatio;
   }
   return worstRatio;
}

static std::vector<std::pair<RTreeMapBase::Node, RTreeMapBase::Rect>>
SquarifyChildren(const std::vector<RTreeMapBase::Node> &children, RTreeMapBase::Rect rect, bool horizontalRows,
                 uint64_t totalSize)
{
   float width = rect.fTopRight.x - rect.fBottomLeft.x;
   float height = rect.fTopRight.y - rect.fBottomLeft.y;
   std::vector<RTreeMapBase::Node> remainingChildren = children;
   std::sort(remainingChildren.begin(), remainingChildren.end(),
             [](const RTreeMapBase::Node &a, const RTreeMapBase::Node &b) { return a.fSize > b.fSize; });
   std::vector<std::pair<RTreeMapBase::Node, RTreeMapBase::Rect>> result;
   RTreeMapBase::Vec2 remainingBegin = rect.fBottomLeft;
   while (!remainingChildren.empty()) {
      std::vector<RTreeMapBase::Node> row;
      float currentWorstRatio = std::numeric_limits<float>::max();
      float remainingWidth = rect.fTopRight.x - remainingBegin.x;
      float remainingHeight = rect.fTopRight.y - remainingBegin.y;
      if (remainingWidth <= 0 || remainingHeight <= 0)
         break;
      while (!remainingChildren.empty()) {
         row.push_back(remainingChildren.front());
         remainingChildren.erase(remainingChildren.begin());
         float newWorstRatio = ComputeWorstRatio(row, remainingWidth, remainingHeight, totalSize, horizontalRows);
         if (newWorstRatio > currentWorstRatio) {
            remainingChildren.insert(remainingChildren.begin(), row.back());
            row.pop_back();
            break;
         }
         currentWorstRatio = newWorstRatio;
      }
      uint64_t sumRow = 0;
      for (const auto &child : row)
         sumRow += child.fSize;
      if (sumRow == 0)
         continue;
      float dimension = horizontalRows ? (static_cast<float>(sumRow) / totalSize * height)
                                       : (static_cast<float>(sumRow) / totalSize * width);
      float position = 0.0f;
      for (const auto &child : row) {
         float childDimension = static_cast<float>(child.fSize) / sumRow * (horizontalRows ? width : height);
         RTreeMapBase::Vec2 childBegin = horizontalRows
                                            ? RTreeMapBase::Vec2{remainingBegin.x + position, remainingBegin.y}
                                            : RTreeMapBase::Vec2{remainingBegin.x, remainingBegin.y + position};
         RTreeMapBase::Vec2 childEnd =
            horizontalRows
               ? RTreeMapBase::Vec2{remainingBegin.x + position + childDimension, remainingBegin.y + dimension}
               : RTreeMapBase::Vec2{remainingBegin.x + dimension, remainingBegin.y + position + childDimension};
         result.push_back({child, {childBegin, childEnd}});
         position += childDimension;
      }
      if (horizontalRows)
         remainingBegin.y += dimension;
      else
         remainingBegin.x += dimension;
   }
   return result;
}
void RTreeMapBase::DrawLegend() const
{
   const auto diskOccupation = GetDiskOccupation(fNodes);

   if (fNodes.empty())
      return;
   const uint64_t totalSize = fNodes[0].fSize;
   if (totalSize == 0)
      return;

   uint8_t counter = 0;
   for (const auto &entry : diskOccupation) {
      const auto &typeName = entry.first;
      const uint64_t entrySize = entry.second;
      if (entrySize == 0)
         continue;

      const auto offset = 0.835f, factor = 0.05f;
      const auto posY = offset - counter * factor;

      AddBox(Rect(Vec2(offset, posY), Vec2(offset + factor, posY - factor)), ComputeColor(typeName));

      const float percent = (entrySize / static_cast<float>(totalSize)) * 100.0f;
      const auto content = "(" + GetDataStr(entrySize) + " / " + GetDataStr(totalSize) + ")";

      float currOffset = 0.0125f;
      for (const auto &currContent : {typeName, content, GetFloatStr(percent, 2) + "%"}) {
         AddText(Vec2(offset + factor, posY - currOffset), currContent, kTextSizeFactor);
         currOffset += 0.01f;
      }

      counter++;
   }
}

void RTreeMapBase::DrawTreeMap(const RTreeMapBase::Node &element, RTreeMapBase::Rect rect, int depth) const
{
   RTreeMapBase::Rect drawRect = RTreeMapBase::Rect(RTreeMapBase::Vec2(rect.fBottomLeft.x, rect.fBottomLeft.y),
                                                    RTreeMapBase::Vec2(rect.fTopRight.x, rect.fTopRight.y));
   bool isLeaf = (element.fNChildren == 0);
   RTreeMapBase::RGBColor boxColor = isLeaf ? ComputeColor(element.fType) : RTreeMapBase::RGBColor(100, 100, 100);
   AddBox(drawRect, boxColor, 0.15f);

   const std::string label = element.fName + " (" + GetDataStr(element.fSize) + ")";
   const Vec2 &labelPos = isLeaf ? Vec2((drawRect.fBottomLeft.x + drawRect.fTopRight.x) / 2.0f,
                                        (drawRect.fBottomLeft.y + drawRect.fTopRight.y) / 2.0f)
                                 : Vec2(drawRect.fBottomLeft.x + kPadTextOffset, drawRect.fTopRight.y - kPadTextOffset);

   float rectWidth = rect.fTopRight.x - rect.fBottomLeft.x;
   float rectHeight = rect.fTopRight.y - rect.fBottomLeft.y;
   float textSize = std::min(std::min(rectWidth, rectHeight) * 0.1f, kTextSizeFactor);
   AddText(labelPos, label, textSize, RTreeMapBase::RGBColor(255, 255, 255), isLeaf);

   if (!isLeaf) {
      float indent = kIndentationOffset;
      RTreeMapBase::Rect innerRect =
         RTreeMapBase::Rect(RTreeMapBase::Vec2(rect.fBottomLeft.x + indent, rect.fBottomLeft.y + indent),
                            RTreeMapBase::Vec2(rect.fTopRight.x - indent, rect.fTopRight.y - indent * 4.0f));
      std::vector<RTreeMapBase::Node> children;
      for (std::uint64_t i = 0; i < element.fNChildren; ++i)
         children.push_back(fNodes[element.fChildrenIdx + i]);
      uint64_t totalSize = 0;
      for (const auto &child : children)
         totalSize += child.fSize;
      if (totalSize == 0)
         return;
      float width = innerRect.fTopRight.x - innerRect.fBottomLeft.x;
      float height = innerRect.fTopRight.y - innerRect.fBottomLeft.y;
      bool horizontalRows = width > height;
      auto childRects = SquarifyChildren(children, innerRect, horizontalRows, totalSize);
      for (const auto &[child, childRect] : childRects)
         DrawTreeMap(child, childRect, depth + 1);
   }
}
