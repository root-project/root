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
#include <ROOT/RTreeUtil.hxx>

#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

class RTreeSink;
class RTreeSource;

namespace Detail {

class RColumnModel;

// clang-format off
/**
\class ROOT::Experimental::RColumn
\ingroup Forest
\brief A column is a storage-backed array of a simple type, from which pages can be mapped into memory

On the primitives data layer, the RColumn and RColumnElement are the equivalents to RBranch and RCargo on the
logical data layer.
*/
// clang-format on
class RColumn {
public:
   RColumn(const RColumnModel &model, RTreeSource &source);
   RColumn(const RColumnModel &model, RTreeSink &sink);

   void Append(const RColumnElementBase &element) {/*...*/}
   void Flush();


   void Read(const TreeIndex_t index, RColumnElementBase* element) {/*...*/}
   void Map(const std::int64_t num, void **dst) {/*...*/}


   // Returns the number of mapped values
   TreeIndex_t MapV(const TreeIndex_t index, const TreeIndex_t count, void **dst) {return 0;/*...*/}

   void ReadV(const TreeIndex_t index, const TreeIndex_t count, void *dst) {/*...*/}

   TreeIndex_t GetNElements();
};

using RColumnCollection_t = std::vector<std::unique_ptr<RColumn>>;

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
