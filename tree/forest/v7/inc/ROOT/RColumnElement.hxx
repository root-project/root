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
#include <ROOT/RForestUtil.hxx>

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
   /// Points to valid C++ data
   void* fRawContent;
   unsigned int fSize;

   /// Indicates that fRawContent is bitwise identical to the type of the RColumnElement
   bool fIsMappable;

   virtual void DoSerialize(void* /* destination */) const { }
   virtual void DoDeserialize(void* /* source */) const { }

public:
   RColumnElementBase()
     : fRawContent(nullptr)
     , fSize(0)
     , fIsMappable(false)
   {}
   RColumnElementBase(void* rawContent, unsigned int size, bool isMappable)
     : fRawContent(rawContent)
     , fSize(size)
     , fIsMappable(isMappable)
   {}
   virtual ~RColumnElementBase() { }

   void Serialize(void *destination) const {
     if (!fIsMappable) {
       DoSerialize(destination);
       return;
     }
     std::memcpy(destination, fRawContent, fSize);
   }

   void Deserialize(void *source) {
     if (!fIsMappable) {
       DoDeserialize(source);
       return;
     }
     std::memcpy(fRawContent, source, fSize);
   }

   void SetRawContent(void* content) { fRawContent = content; }
   void* GetRawContent() const { return fRawContent; }
   decltype(fSize) GetSize() const { return fSize; }
};

/**
 * Pairs of C++ type and column type, like float and EColumnType::kReal32
 */
template <typename CppT, EColumnType ColumnT>
class RColumnElement : public RColumnElementBase {
public:
   static constexpr bool kIsMappable = false;
   explicit RColumnElement(CppT* value) : RColumnElementBase(value, sizeof(CppT), kIsMappable)
   {
      static_assert(sizeof(CppT) != sizeof(CppT), "No column mapping for this C++ type");
   }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

template <>
class ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal32>
   : public ROOT::Experimental::Detail::RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   explicit RColumnElement(float* value) : RColumnElementBase(value, sizeof(float), kIsMappable) {}
};

template <>
class ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::TreeIndex_t, ROOT::Experimental::EColumnType::kIndex>
   : public ROOT::Experimental::Detail::RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   explicit RColumnElement(ROOT::Experimental::TreeIndex_t* value)
      : RColumnElementBase(value, sizeof(ROOT::Experimental::TreeIndex_t), kIsMappable) {}
};

template <>
class ROOT::Experimental::Detail::RColumnElement<char, ROOT::Experimental::EColumnType::kByte>
   : public ROOT::Experimental::Detail::RColumnElementBase {
public:
   static constexpr bool kIsMappable = true;
   explicit RColumnElement(char* value) : RColumnElementBase(value, sizeof(char), kIsMappable) {}
};

#endif
