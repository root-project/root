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
   /// Indicates that fRawContent is bitwise identical to the type of the RColumnElement
   bool fIsMovable;
   void *fRawContent;
   unsigned fSize;

   EColumnType fColumnType;

   virtual void DoSerialize(void* /* destination */) const { }
   virtual void DoDeserialize(void* /* source */) const { }

public:
   RColumnElementBase()
     : fIsMovable(false)
     , fRawContent(nullptr)
     , fSize(0)
     , fColumnType(EColumnType::kUnknown) { }
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

   /// Used to map directly from slices
   void SetRawContent(void *source) {
     if (!fIsMovable) {
       Deserialize(source);
       return;
     }
     fRawContent = source;
   }
};

template <typename T>
class RColumnElement : public RColumnElementBase {
   T* fValue;

public:
   template<typename... ArgsT>
   explicit RColumnElement(T *value) : fValue(value) {
   }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
