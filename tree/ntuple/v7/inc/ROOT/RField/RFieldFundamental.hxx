/// \file ROOT/RField/Fundamental.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RField_Fundamental
#define ROOT7_RField_Fundamental

#ifndef ROOT7_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RColumn.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace ROOT {
namespace Experimental {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for concrete C++ fundamental types
////////////////////////////////////////////////////////////////////////////////

template <>
class RField<void> : public RFieldBase {
public:
   static std::string TypeName() { return "void"; }
   // RField<void> should never be constructed.
   RField() = delete;
   RField(const RField &) = delete;
   RField &operator=(const RField &) = delete;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for integral types
////////////////////////////////////////////////////////////////////////////////

// bool and char are somewhat special, handle them first

extern template class RSimpleField<bool>;

template <>
class RField<bool> final : public RSimpleField<bool> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "bool"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<char>;

template <>
class RField<char> final : public RSimpleField<char> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "char"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

// For other integral types, we introduce an intermediate RIntegralField. It is specialized for fixed-width integer
// types (from std::[u]int8_t to std::[u]int64_t). RField<T> for integral types T is specialized by mapping to the
// corresponding fixed-width integer type (see RIntegralTypeMap).

template <typename T>
class RIntegralField {
   // Instantiating this base template definition should never happen and is an error!
   RIntegralField() = delete;
};

extern template class RSimpleField<std::int8_t>;

template <>
class RIntegralField<std::int8_t> : public RSimpleField<std::int8_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::int8_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::uint8_t>;

template <>
class RIntegralField<std::uint8_t> : public RSimpleField<std::uint8_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::uint8_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::int16_t>;

template <>
class RIntegralField<std::int16_t> : public RSimpleField<std::int16_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::int16_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::uint16_t>;

template <>
class RIntegralField<std::uint16_t> : public RSimpleField<std::uint16_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::uint16_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::int32_t>;

template <>
class RIntegralField<std::int32_t> : public RSimpleField<std::int32_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::int32_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::uint32_t>;

template <>
class RIntegralField<std::uint32_t> : public RSimpleField<std::uint32_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::uint32_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::int64_t>;

template <>
class RIntegralField<std::int64_t> : public RSimpleField<std::int64_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::int64_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

extern template class RSimpleField<std::uint64_t>;

template <>
class RIntegralField<std::uint64_t> : public RSimpleField<std::uint64_t> {
protected:
   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::uint64_t"; }
   explicit RIntegralField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RIntegralField(RIntegralField &&other) = default;
   RIntegralField &operator=(RIntegralField &&other) = default;
   ~RIntegralField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

namespace Internal {
// Map standard integer types to fixed width equivalents.
template <typename T>
struct RIntegralTypeMap {
   using type = T;
};

// RField<char> has its own specialization, we should not need a specialization of RIntegralTypeMap.
// From https://en.cppreference.com/w/cpp/language/types:
// char has the same representation and alignment as either signed char or
// unsigned char, but is always a distinct type.
template <>
struct RIntegralTypeMap<signed char> {
   static_assert(sizeof(signed char) == sizeof(std::int8_t));
   using type = std::int8_t;
};
template <>
struct RIntegralTypeMap<unsigned char> {
   static_assert(sizeof(unsigned char) == sizeof(std::uint8_t));
   using type = std::uint8_t;
};
template <>
struct RIntegralTypeMap<short> {
   static_assert(sizeof(short) == sizeof(std::int16_t));
   using type = std::int16_t;
};
template <>
struct RIntegralTypeMap<unsigned short> {
   static_assert(sizeof(unsigned short) == sizeof(std::uint16_t));
   using type = std::uint16_t;
};
template <>
struct RIntegralTypeMap<int> {
   static_assert(sizeof(int) == sizeof(std::int32_t));
   using type = std::int32_t;
};
template <>
struct RIntegralTypeMap<unsigned int> {
   static_assert(sizeof(unsigned int) == sizeof(std::uint32_t));
   using type = std::uint32_t;
};
template <>
struct RIntegralTypeMap<long> {
   static_assert(sizeof(long) == sizeof(std::int32_t) || sizeof(long) == sizeof(std::int64_t));
   using type = std::conditional_t<sizeof(long) == sizeof(std::int32_t), std::int32_t, std::int64_t>;
};
template <>
struct RIntegralTypeMap<unsigned long> {
   static_assert(sizeof(unsigned long) == sizeof(std::uint32_t) || sizeof(unsigned long) == sizeof(std::uint64_t));
   using type = std::conditional_t<sizeof(unsigned long) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;
};
template <>
struct RIntegralTypeMap<long long> {
   static_assert(sizeof(long long) == sizeof(std::int64_t));
   using type = std::int64_t;
};
template <>
struct RIntegralTypeMap<unsigned long long> {
   static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
   using type = std::uint64_t;
};
} // namespace Internal

template <typename T>
class RField<T, typename std::enable_if<std::is_integral_v<T>>::type> final
   : public RIntegralField<typename Internal::RIntegralTypeMap<T>::type> {
   using MappedType = typename Internal::RIntegralTypeMap<T>::type;
   static_assert(sizeof(T) == sizeof(MappedType), "invalid size of mapped type");
   static_assert(std::is_signed_v<T> == std::is_signed_v<MappedType>, "invalid signedness of mapped type");
   using BaseType = RIntegralField<MappedType>;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

public:
   RField(std::string_view name) : RIntegralField<MappedType>(name) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   T *Map(NTupleSize_t globalIndex) { return reinterpret_cast<T *>(this->BaseType::Map(globalIndex)); }
   T *Map(RClusterIndex clusterIndex) { return reinterpret_cast<T *>(this->BaseType::Map(clusterIndex)); }
   T *MapV(NTupleSize_t globalIndex, NTupleSize_t &nItems)
   {
      return reinterpret_cast<T *>(this->BaseType::MapV(globalIndex, nItems));
   }
   T *MapV(RClusterIndex clusterIndex, NTupleSize_t &nItems)
   {
      return reinterpret_cast<T *>(this->BaseType::MapV(clusterIndex, nItems));
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for floating-point types
////////////////////////////////////////////////////////////////////////////////

extern template class RSimpleField<float>;

template <>
class RField<float> final : public RSimpleField<float> {
   std::size_t fBitWidth = sizeof(float) * 8;
   double fValueMin = std::numeric_limits<float>::min();
   double fValueMax = std::numeric_limits<float>::max();

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      auto cloned = std::make_unique<RField<float>>(newName);
      cloned->fBitWidth = fBitWidth;
      cloned->fValueMin = fValueMin;
      cloned->fValueMax = fValueMax;
      return cloned;
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

public:
   static std::string TypeName() { return "float"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;

   /// Sets this field to use a half precision representation, occupying half as much storage space (16 bits) on disk.
   /// This is mutually exclusive with `SetTruncated` and `SetQuantized`.
   void SetHalfPrecision();
   /// Set the precision of this field to `nBits`. The remaining (32 - `nBits`) bits will be truncated
   /// from the number's mantissa. `nBits` must be $10 <= nBits <= 31$ (this means that at least 1 bit
   /// of mantissa is always preserved). Note that this effectively rounds the number towards 0.
   /// This is mutually exclusive with `SetHalfPrecision` and `SetQuantized`.
   /// \note Calling `SetTruncated(16)` effectively makes this field a `bfloat16` on disk.
   void SetTruncated(std::size_t nBits);
   /// Sets this field to use a quantized integer representation using `nBits` per value.
   /// This call promises that this field will only contain values contained in `[minValue, maxValue]` inclusive.
   /// If a value outside this range is assigned to this field, the behavior is undefined.
   /// This is mutually exclusive with `SetTruncated` and `SetHalfPrecision`.
   void SetQuantized(float minValue, float maxValue, std::size_t nBits);
};

extern template class RSimpleField<double>;

template <>
class RField<double> final : public RSimpleField<double> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "double"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;

   // Set the column representation to 32 bit floating point and the type alias to Double32_t
   void SetDouble32();
};
} // namespace Experimental
} // namespace ROOT

#endif
