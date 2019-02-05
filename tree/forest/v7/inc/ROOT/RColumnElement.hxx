/// \file ROOT/RColumnElement.hxx
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

#ifndef ROOT7_RColumnElement
#define ROOT7_RColumnElement

#include <ROOT/RColumnModel.hxx>

#include <cstring> // for memcpy

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RColumnElement
\ingroup Forest
\brief A column element points into a memory mapped page, to a particular data item
*/
// clang-format on
class RColumnElementBase {
protected:
   void *fRawContent;
   unsigned int fSize;

   /// Indicates that fRawContent is bitwise identical to the type of the RColumnElement
   bool fIsMovable;
   EColumnType fColumnType;

   virtual void DoSerialize(void* /* destination */) const { }
   virtual void DoDeserialize(void* /* source */) const { }

public:
   RColumnElementBase()
     : fRawContent(nullptr)
     , fSize(0)
     , fIsMovable(false)
     , fColumnType(EColumnType::kUnknown) { }
   RColumnElementBase(void* rawContent, unsigned int size, bool isMovable, EColumnType columnType)
     : fRawContent(rawContent)
     , fSize(size)
     , fIsMovable(isMovable)
     , fColumnType(columnType) { }
   virtual ~RColumnElementBase() { }

   void Serialize(void *destination) const {
     if (!fIsMovable) {
       DoSerialize(destination);
       return;
     }
     std::memcpy(destination, fRawContent, fSize);
   }

   void Deserialize(void *source) {
     if (!fIsMovable) {
       DoDeserialize(source);
       return;
     }
     std::memcpy(fRawContent, source, fSize);
   }

   /// Used to map directly from pages
   void SetRawContent(void *source) {
     if (!fIsMovable) {
       Deserialize(source);
       return;
     }
     fRawContent = source;
   }

   decltype(fSize) GetSize() const { return fSize; }
};

/**
 * Column types that map bit-wise to C++ types
 */
template <typename T>
class RColumnElementDirect : public RColumnElementBase {
public:
   explicit RColumnElementDirect(T* value, EColumnType columnType)
      : RColumnElementBase(value, sizeof(T), true /* isMovable */, columnType) {}
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
