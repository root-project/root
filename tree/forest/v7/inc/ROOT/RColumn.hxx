/// \file ROOT/RColumn.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RColumn
#define ROOT7_RColumn

#include <ROOT/RColumnElement.hxx>
#include <ROOT/RForestUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

class RColumnModel;

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RColumn
\ingroup Forest
\brief A column is a storage-backed array of a simple, fixed-size type, from which pages can be mapped into memory.

On the primitives data layer, the RColumn and RColumnElement are the equivalents to RField and RTreeValue on the
logical data layer.
*/
// clang-format on
class RColumn {
private:
   RColumnModel fModel;
   RPageSink* fPageSink;
   RPageSource* fPageSource;
   RPageStorage::ColumnHandle_t fHandleSink;
   RPageStorage::ColumnHandle_t fHandleSource;
   /// Open page into which new elements are being written
   RPage fHeadPage;
   /// The number of elements written resp. available in the input tree
   TreeIndex_t fNElements;
   /// The currently mapped page for reading
   RPage fCurrentPage;
   /// Index of the first element in fCurrentPage
   TreeIndex_t fCurrentPageFirst;
   /// Index of the last element in fCurrentPage
   TreeIndex_t fCurrentPageLast;
   /// The tree id is unused when writing but set when connected to a source
   TreeId_t fTreeId;

public:
   RColumn(const RColumnModel& model);
   RColumn(const RColumn&) = delete;
   RColumn& operator =(const RColumn&) = delete;
   ~RColumn() = default;

   void Connect(RPageStorage* pageStorage);

   void Append(const RColumnElementBase& element) {
      void* dst = fHeadPage.Reserve(1);
      if (dst == nullptr) {
         Flush();
         dst = fHeadPage.Reserve(1);
         R__ASSERT(dst != nullptr);
      }
      element.Serialize(dst);
      fNElements++;
   }

   void Read(const TreeIndex_t index, RColumnElementBase* element) {
      if ((index < fCurrentPageFirst) || (index > fCurrentPageLast)) {
         MapPage(index);
      }
      void* src = static_cast<unsigned char *>(fCurrentPage.GetBuffer()) +
                  (index - fCurrentPageFirst) * element->GetSize();
      element->Deserialize(src);
   }

   // TODO: template by the expected C++ type and decide whether to read or to map
   template <typename CppT, EColumnType ColumnT>
   void* Map(const TreeIndex_t index, RColumnElementBase* element) {
      if (!RColumnElement<CppT, ColumnT>::kIsMappable) {
         Read(index, element);
         return element->GetRawContent();
      }

      if ((index < fCurrentPageFirst) || (index > fCurrentPageLast)) {
         MapPage(index);
      }
      return nullptr;
   }

   // Returns the number of mapped values
   TreeIndex_t MapV(const TreeIndex_t /*index*/, const TreeIndex_t /*count*/, void ** /*dst*/) {return 0;/*...*/}
   void ReadV(const TreeIndex_t /*index*/, const TreeIndex_t /*count*/, void * /*dst*/) {/*...*/}

   void Flush();
   void MapPage(const TreeIndex_t index);
   TreeIndex_t GetNElements() { return fNElements; }
   const RColumnModel& GetModel() const { return fModel; }
   TreeId_t GetTreeId() const { return fTreeId; }
   RPageSource* GetPageSource() const { return fPageSource; }
   RPageStorage::ColumnHandle_t GetHandleSource() const { return fHandleSource; }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
