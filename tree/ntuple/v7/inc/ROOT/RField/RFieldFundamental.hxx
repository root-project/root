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
   ~RField() final = default;

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
   ~RField() final = default;

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
   ~RField() final = default;

   T *Map(ROOT::NTupleSize_t globalIndex) { return reinterpret_cast<T *>(this->BaseType::Map(globalIndex)); }
   T *Map(RNTupleLocalIndex localIndex) { return reinterpret_cast<T *>(this->BaseType::Map(localIndex)); }
   T *MapV(ROOT::NTupleSize_t globalIndex, ROOT::NTupleSize_t &nItems)
   {
      return reinterpret_cast<T *>(this->BaseType::MapV(globalIndex, nItems));
   }
   T *MapV(RNTupleLocalIndex localIndex, ROOT::NTupleSize_t &nItems)
   {
      return reinterpret_cast<T *>(this->BaseType::MapV(localIndex, nItems));
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for floating-point types
////////////////////////////////////////////////////////////////////////////////

extern template class RSimpleField<double>;
extern template class RSimpleField<float>;

template <typename T>
class RRealField : public RSimpleField<T> {
   using Base = RSimpleField<T>;

   using Base::fAvailableColumns;
   using Base::fColumnRepresentatives;
   using Base::fPrincipalColumn;

   std::size_t fBitWidth = sizeof(T) * 8;
   double fValueMin = std::numeric_limits<T>::min();
   double fValueMax = std::numeric_limits<T>::max();

protected:
   /// Called by derived fields' CloneImpl()
   RRealField(std::string_view name, const RRealField &source)
      : RSimpleField<T>(name, source.GetTypeName()),
        fBitWidth(source.fBitWidth),
        fValueMin(source.fValueMin),
        fValueMax(source.fValueMax)
   {
   }

   void GenerateColumns() final
   {
      const auto r = Base::GetColumnRepresentatives();
      const auto n = r.size();
      fAvailableColumns.reserve(n);
      for (std::uint16_t i = 0; i < n; ++i) {
         auto &column = fAvailableColumns.emplace_back(Internal::RColumn::Create<T>(r[i][0], 0, i));
         if (r[i][0] == ROOT::ENTupleColumnType::kReal32Trunc) {
            column->SetBitsOnStorage(fBitWidth);
         } else if (r[i][0] == ROOT::ENTupleColumnType::kReal32Quant) {
            column->SetBitsOnStorage(fBitWidth);
            column->SetValueRange(fValueMin, fValueMax);
         }
      }
      fPrincipalColumn = fAvailableColumns[0].get();
   }

   void GenerateColumns(const RNTupleDescriptor &desc) final
   {
      std::uint16_t representationIndex = 0;
      do {
         const auto &onDiskTypes = Base::EnsureCompatibleColumnTypes(desc, representationIndex);
         if (onDiskTypes.empty())
            break;

         auto &column =
            fAvailableColumns.emplace_back(Internal::RColumn::Create<T>(onDiskTypes[0], 0, representationIndex));
         if (onDiskTypes[0] == ROOT::ENTupleColumnType::kReal32Trunc) {
            const auto &fdesc = desc.GetFieldDescriptor(Base::GetOnDiskId());
            const auto &coldesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
            column->SetBitsOnStorage(coldesc.GetBitsOnStorage());
         } else if (onDiskTypes[0] == ROOT::ENTupleColumnType::kReal32Quant) {
            const auto &fdesc = desc.GetFieldDescriptor(Base::GetOnDiskId());
            const auto &coldesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
            assert(coldesc.GetValueRange().has_value());
            const auto [valMin, valMax] = *coldesc.GetValueRange();
            column->SetBitsOnStorage(coldesc.GetBitsOnStorage());
            column->SetValueRange(valMin, valMax);
         }
         fColumnRepresentatives.emplace_back(onDiskTypes);
         if (representationIndex > 0) {
            fAvailableColumns[0]->MergeTeams(*fAvailableColumns[representationIndex]);
         }

         representationIndex++;
      } while (true);
      fPrincipalColumn = fAvailableColumns[0].get();
   }

   ~RRealField() override = default;

public:
   using Base::SetColumnRepresentatives;

   RRealField(std::string_view name, std::string_view typeName) : RSimpleField<T>(name, typeName) {}
   RRealField(RRealField &&other) = default;
   RRealField &operator=(RRealField &&other) = default;

   /// Sets this field to use a half precision representation, occupying half as much storage space (16 bits:
   /// 1 sign bit, 5 exponent bits, 10 mantissa bits) on disk.
   /// This is mutually exclusive with `SetTruncated` and `SetQuantized` and supersedes them if called after them.
   void SetHalfPrecision() { SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal16}}); }

   /// Set the on-disk representation of this field to a single-precision float truncated to `nBits`.
   /// The remaining (32 - `nBits`) bits will be truncated from the number's mantissa.
   /// `nBits` must be $10 <= nBits <= 31$ (this means that at least 1 bit
   /// of mantissa is always preserved). Note that this effectively rounds the number towards 0.
   /// For a double-precision field, this implies first a cast to single-precision, then the truncation.
   /// This is mutually exclusive with `SetHalfPrecision` and `SetQuantized` and supersedes them if called after them.
   void SetTruncated(std::size_t nBits)
   {
      const auto &[minBits, maxBits] =
         Internal::RColumnElementBase::GetValidBitRange(ROOT::ENTupleColumnType::kReal32Trunc);
      if (nBits < minBits || nBits > maxBits) {
         throw RException(R__FAIL("SetTruncated() argument nBits = " + std::to_string(nBits) +
                                  " is out of valid range [" + std::to_string(minBits) + ", " +
                                  std::to_string(maxBits) + "])"));
      }
      SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32Trunc}});
      fBitWidth = nBits;
   }

   /// Sets this field to use a quantized integer representation using `nBits` per value.
   /// It must be $1 <= nBits <= 32$.
   /// `minValue` and `maxValue` must not be infinity, NaN or denormal floats, and they must be representable by the
   /// type T.
   /// Calling this function establishes a promise by the caller to RNTuple that this field will only contain values
   /// contained in `[minValue, maxValue]` inclusive. If a value outside this range is assigned to this field, the
   /// behavior is undefined.
   /// This is mutually exclusive with `SetTruncated` and `SetHalfPrecision` and supersedes them if called after them.
   void SetQuantized(double minValue, double maxValue, std::size_t nBits)
   {
      const auto &[minBits, maxBits] =
         Internal::RColumnElementBase::GetValidBitRange(ROOT::ENTupleColumnType::kReal32Quant);
      if (nBits < minBits || nBits > maxBits) {
         throw RException(R__FAIL("SetQuantized() argument nBits = " + std::to_string(nBits) +
                                  " is out of valid range [" + std::to_string(minBits) + ", " +
                                  std::to_string(maxBits) + "])"));
      }
      SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32Quant}});
      fBitWidth = nBits;
      fValueMin = minValue;
      fValueMax = maxValue;
   }
};

template <>
class RField<float> final : public RRealField<float> {
   const RColumnRepresentations &GetColumnRepresentations() const final;

   RField(std::string_view name, const RField &source) : RRealField<float>(name, source) {}

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::unique_ptr<RField>(new RField(newName, *this));
   }

public:
   static std::string TypeName() { return "float"; }

   explicit RField(std::string_view name) : RRealField<float>(name, TypeName()) {}

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

template <>
class RField<double> final : public RRealField<double> {
   const RColumnRepresentations &GetColumnRepresentations() const final;

   RField(std::string_view name, const RField &source) : RRealField<double>(name, source) {}

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::unique_ptr<RField>(new RField(newName, *this));
   }

public:
   static std::string TypeName() { return "double"; }

   explicit RField(std::string_view name) : RRealField<double>(name, TypeName()) {}

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;

   // Set the column representation to 32 bit floating point and the type alias to Double32_t
   void SetDouble32();
};
} // namespace Experimental
} // namespace ROOT

#endif
