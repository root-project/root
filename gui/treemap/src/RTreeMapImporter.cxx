/// \file RTreeMapImporter.cxx
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

#include <ROOT/RTreeMapPainter.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RColumnElementBase.hxx>

#include <queue>

using namespace ROOT::Experimental;

static RTreeMapBase::Node CreateNode(const RNTupleInspector &insp, const ROOT::RFieldDescriptor &fldDesc,
                                     std::uint64_t childrenIdx, std::uint64_t nChildren, ROOT::DescriptorId_t rootId,
                                     size_t rootSize)
{
   uint64_t size =
      (rootId != fldDesc.GetId()) ? insp.GetFieldTreeInspector(fldDesc.GetId()).GetCompressedSize() : rootSize;
   return {fldDesc.GetFieldName(), "", size, childrenIdx, nChildren};
}

static RTreeMapBase::Node CreateNode(const RNTupleInspector::RColumnInspector &colInsp, std::uint64_t childrenIdx)
{
   return {"", ROOT::Internal::RColumnElementBase::GetColumnTypeName(colInsp.GetType()), colInsp.GetCompressedSize(),
           childrenIdx, 0};
}

std::unique_ptr<RTreeMapPainter> RTreeMapPainter::ImportRNTuple(const ROOT::Experimental::RNTupleInspector &insp)
{
   auto treemap = std::make_unique<RTreeMapPainter>();
   const auto &descriptor = insp.GetDescriptor();
   const auto rootId = descriptor.GetFieldZero().GetId();
   size_t rootSize = 0;
   for (const auto &childId : descriptor.GetFieldDescriptor(rootId).GetLinkIds()) {
      rootSize += insp.GetFieldTreeInspector(childId).GetCompressedSize();
   }

   std::queue<std::pair<uint64_t, bool>> queue; // (columnid/fieldid, isfield)
   queue.emplace(rootId, true);
   while (!queue.empty()) {
      size_t levelSize = queue.size();
      size_t levelChildrenStart = treemap->fNodes.size() + levelSize;
      for (size_t i = 0; i < levelSize; ++i) {
         const auto &current = queue.front();
         queue.pop();

         size_t nChildren = 0;
         if (current.second) {
            std::vector<uint64_t> children;
            const auto &fldDesc = descriptor.GetFieldDescriptor(current.first);
            children = fldDesc.GetLinkIds();
            for (const auto childId : children) {
               queue.emplace(childId, 1);
            }
            for (const auto &columnDesc : descriptor.GetColumnIterable(fldDesc.GetId())) {
               const auto &columnId = columnDesc.GetPhysicalId();
               children.push_back(columnId);
               queue.emplace(columnId, 0);
            }
            nChildren = children.size();
            const auto &node = CreateNode(insp, fldDesc, levelChildrenStart, nChildren, rootId, rootSize);
            treemap->fNodes.push_back(node);
         } else {
            const auto &colInsp = insp.GetColumnInspector(current.first);
            const auto &node = CreateNode(colInsp, levelChildrenStart);
            treemap->fNodes.push_back(node);
         }

         levelChildrenStart += nChildren;
      }
   }
   return treemap;
}

std::unique_ptr<RTreeMapPainter> RTreeMapPainter::ImportRNTuple(std::string_view sourceFileName, std::string_view tupleName)
{
   auto insp = RNTupleInspector::Create(tupleName, sourceFileName);
   return RTreeMapPainter::ImportRNTuple(*insp);
}