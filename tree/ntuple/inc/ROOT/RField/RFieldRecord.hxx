/// \file ROOT/RField/Fundamental.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RField_Record
#define ROOT_RField_Record

#ifndef ROOT_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace ROOT {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

namespace Internal {
std::unique_ptr<RFieldBase> CreateEmulatedRecordField(std::string_view fieldName,
                                                      std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                                      std::string_view emulatedFromType);
} // namespace Internal

/// The field for an untyped record. The subfields are stored consecutively in a memory block, i.e.
/// the memory layout is identical to one that a C++ struct would have
class RRecordField : public RFieldBase {
   friend std::unique_ptr<RFieldBase>
   Internal::CreateEmulatedRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                       std::string_view emulatedFromType);

   class RRecordDeleter : public RDeleter {
   private:
      std::vector<std::unique_ptr<RDeleter>> fItemDeleters;
      std::vector<std::size_t> fOffsets;

   public:
      RRecordDeleter(std::vector<std::unique_ptr<RDeleter>> itemDeleters, const std::vector<std::size_t> &offsets)
         : fItemDeleters(std::move(itemDeleters)), fOffsets(offsets)
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   RRecordField(std::string_view name, const RRecordField &source); // Used by CloneImpl()

   /// If `emulatedFromType` is non-empty, this field was created as a replacement for a ClassField that we lack a
   /// dictionary for and reconstructed from the on-disk information.
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields,
                std::string_view emulatedFromType);

protected:
   std::size_t fMaxAlignment = 1;
   std::size_t fSize = 0;
   std::vector<std::size_t> fOffsets;

   std::size_t GetItemPadding(std::size_t baseOffset, std::size_t itemAlignment) const;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final;

   RRecordField(std::string_view fieldName, std::string_view typeName);

   void AttachItemFields(std::vector<std::unique_ptr<RFieldBase>> itemFields);

   template <std::size_t N>
   void AttachItemFields(std::array<std::unique_ptr<RFieldBase>, N> itemFields)
   {
      fTraits |= kTraitTrivialType;
      for (unsigned i = 0; i < N; ++i) {
         fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
         fSize += GetItemPadding(fSize, itemFields[i]->GetAlignment()) + itemFields[i]->GetValueSize();
         fTraits &= itemFields[i]->GetTraits();
         Attach(std::move(itemFields[i]));
      }
      // Trailing padding: although this is implementation-dependent, most add enough padding to comply with the
      // requirements of the type with strictest alignment
      fSize += GetItemPadding(fSize, fMaxAlignment);
   }

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) override;

public:
   /// Construct a RRecordField based on a vector of child fields. The ownership of the child fields is transferred
   /// to the RRecordField instance.
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields);
   RRecordField(RRecordField &&other) = default;
   RRecordField &operator=(RRecordField &&other) = default;
   ~RRecordField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final
   {
      // The minimum size is 1 to support having vectors of empty records
      return std::max<size_t>(1ul, fSize);
   }
   size_t GetAlignment() const final { return fMaxAlignment; }
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;

   const std::vector<std::size_t> &GetOffsets() const { return fOffsets; }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::pair
////////////////////////////////////////////////////////////////////////////////

/// The generic field for `std::pair<T1, T2>` types
class RPairField : public RRecordField {
private:
   static std::string GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields);

protected:
   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> itemFields,
              const std::array<std::size_t, 2> &offsets);

public:
   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> itemFields);
   RPairField(RPairField &&other) = default;
   RPairField &operator=(RPairField &&other) = default;
   ~RPairField() override = default;
};

template <typename T1, typename T2>
class RField<std::pair<T1, T2>> final : public RPairField {
   using ContainerT = typename std::pair<T1, T2>;

private:
   static std::array<std::unique_ptr<RFieldBase>, 2> BuildItemFields()
   {
      return {std::make_unique<RField<T1>>("_0"), std::make_unique<RField<T2>>("_1")};
   }

   static std::array<std::size_t, 2> BuildItemOffsets()
   {
      auto pair = ContainerT();
      auto offsetFirst = reinterpret_cast<std::uintptr_t>(&(pair.first)) - reinterpret_cast<std::uintptr_t>(&pair);
      auto offsetSecond = reinterpret_cast<std::uintptr_t>(&(pair.second)) - reinterpret_cast<std::uintptr_t>(&pair);
      return {offsetFirst, offsetSecond};
   }

public:
   static std::string TypeName() { return "std::pair<" + RField<T1>::TypeName() + "," + RField<T2>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RPairField(name, BuildItemFields(), BuildItemOffsets())
   {
      R__ASSERT(fMaxAlignment >= std::max(alignof(T1), alignof(T2)));
      R__ASSERT(fSize >= sizeof(ContainerT));
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::tuple
////////////////////////////////////////////////////////////////////////////////

/// The generic field for `std::tuple<Ts...>` types
class RTupleField : public RRecordField {
private:
   static std::string GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields);

protected:
   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields,
               const std::vector<std::size_t> &offsets);

public:
   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields);
   RTupleField(RTupleField &&other) = default;
   RTupleField &operator=(RTupleField &&other) = default;
   ~RTupleField() override = default;
};

template <typename... ItemTs>
class RField<std::tuple<ItemTs...>> final : public RTupleField {
   using ContainerT = typename std::tuple<ItemTs...>;

private:
   template <typename HeadT, typename... TailTs>
   static std::string BuildItemTypes()
   {
      std::string result = RField<HeadT>::TypeName();
      if constexpr (sizeof...(TailTs) > 0)
         result += "," + BuildItemTypes<TailTs...>();
      return result;
   }

   template <typename HeadT, typename... TailTs>
   static void _BuildItemFields(std::vector<std::unique_ptr<RFieldBase>> &itemFields, unsigned int index = 0)
   {
      itemFields.emplace_back(new RField<HeadT>("_" + std::to_string(index)));
      if constexpr (sizeof...(TailTs) > 0)
         _BuildItemFields<TailTs...>(itemFields, index + 1);
   }
   static std::vector<std::unique_ptr<RFieldBase>> BuildItemFields()
   {
      std::vector<std::unique_ptr<RFieldBase>> result;
      _BuildItemFields<ItemTs...>(result);
      return result;
   }

   template <unsigned Index, typename HeadT, typename... TailTs>
   static void _BuildItemOffsets(std::vector<std::size_t> &offsets, const ContainerT &tuple)
   {
      auto offset =
         reinterpret_cast<std::uintptr_t>(&std::get<Index>(tuple)) - reinterpret_cast<std::uintptr_t>(&tuple);
      offsets.emplace_back(offset);
      if constexpr (sizeof...(TailTs) > 0)
         _BuildItemOffsets<Index + 1, TailTs...>(offsets, tuple);
   }
   static std::vector<std::size_t> BuildItemOffsets()
   {
      std::vector<std::size_t> result;
      _BuildItemOffsets<0, ItemTs...>(result, ContainerT());
      return result;
   }

public:
   static std::string TypeName() { return "std::tuple<" + BuildItemTypes<ItemTs...>() + ">"; }
   explicit RField(std::string_view name) : RTupleField(name, BuildItemFields(), BuildItemOffsets())
   {
      R__ASSERT(fMaxAlignment >= std::max({alignof(ItemTs)...}));
      R__ASSERT(fSize >= sizeof(ContainerT));
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

} // namespace ROOT

#endif
