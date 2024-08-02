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

#ifndef ROOT7_RField_Record
#define ROOT7_RField_Record

#ifndef ROOT7_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

/// The field for an untyped record. The subfields are stored consequitively in a memory block, i.e.
/// the memory layout is identical to one that a C++ struct would have
class RRecordField : public RFieldBase {
private:
   class RRecordDeleter : public RDeleter {
   private:
      std::vector<std::unique_ptr<RDeleter>> fItemDeleters;
      std::vector<std::size_t> fOffsets;

   public:
      RRecordDeleter(std::vector<std::unique_ptr<RDeleter>> &itemDeleters, const std::vector<std::size_t> &offsets)
         : fItemDeleters(std::move(itemDeleters)), fOffsets(offsets)
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

protected:
   std::size_t fMaxAlignment = 1;
   std::size_t fSize = 0;
   std::vector<std::size_t> fOffsets;

   std::size_t GetItemPadding(std::size_t baseOffset, std::size_t itemAlignment) const;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;

   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
                const std::vector<std::size_t> &offsets, std::string_view typeName = "");

   template <std::size_t N>
   RRecordField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, N> &&itemFields,
                const std::array<std::size_t, N> &offsets, std::string_view typeName = "")
      : ROOT::Experimental::RFieldBase(fieldName, typeName, ENTupleStructure::kRecord, false /* isSimple */)
   {
      fTraits |= kTraitTrivialType;
      for (unsigned i = 0; i < N; ++i) {
         fOffsets.push_back(offsets[i]);
         fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
         fSize += GetItemPadding(fSize, itemFields[i]->GetAlignment()) + itemFields[i]->GetValueSize();
         fTraits &= itemFields[i]->GetTraits();
         Attach(std::move(itemFields[i]));
      }
   }

public:
   /// Construct a RRecordField based on a vector of child fields. The ownership of the child fields is transferred
   /// to the RRecordField instance.
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields);
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &itemFields);
   RRecordField(RRecordField &&other) = default;
   RRecordField &operator=(RRecordField &&other) = default;
   ~RRecordField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return fSize; }
   size_t GetAlignment() const final { return fMaxAlignment; }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::pair
////////////////////////////////////////////////////////////////////////////////

/// The generic field for `std::pair<T1, T2>` types
class RPairField : public RRecordField {
private:
   class RPairDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RPairDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   static std::string GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override { return std::make_unique<RPairDeleter>(fClass); }

   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> &&itemFields,
              const std::array<std::size_t, 2> &offsets);

public:
   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> &itemFields);
   RPairField(RPairField &&other) = default;
   RPairField &operator=(RPairField &&other) = default;
   ~RPairField() override = default;
};

template <typename T1, typename T2>
class RField<std::pair<T1, T2>> final : public RPairField {
   using ContainerT = typename std::pair<T1, T2>;

private:
   template <typename Ty1, typename Ty2>
   static std::array<std::unique_ptr<RFieldBase>, 2> BuildItemFields()
   {
      return {std::make_unique<RField<Ty1>>("_0"), std::make_unique<RField<Ty2>>("_1")};
   }

   static std::array<std::size_t, 2> BuildItemOffsets()
   {
      auto pair = ContainerT();
      auto offsetFirst = reinterpret_cast<std::uintptr_t>(&(pair.first)) - reinterpret_cast<std::uintptr_t>(&pair);
      auto offsetSecond = reinterpret_cast<std::uintptr_t>(&(pair.second)) - reinterpret_cast<std::uintptr_t>(&pair);
      return {offsetFirst, offsetSecond};
   }

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      std::array<std::unique_ptr<RFieldBase>, 2> items{fSubFields[0]->Clone(fSubFields[0]->GetFieldName()),
                                                       fSubFields[1]->Clone(fSubFields[1]->GetFieldName())};
      return std::make_unique<RField<std::pair<T1, T2>>>(newName, std::move(items));
   }

   void ConstructValue(void *where) const final { new (where) ContainerT(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<ContainerT>>(); }

public:
   static std::string TypeName() { return "std::pair<" + RField<T1>::TypeName() + "," + RField<T2>::TypeName() + ">"; }
   explicit RField(std::string_view name, std::array<std::unique_ptr<RFieldBase>, 2> &&itemFields)
      : RPairField(name, std::move(itemFields), BuildItemOffsets())
   {
      fMaxAlignment = std::max(alignof(T1), alignof(T2));
      fSize = sizeof(ContainerT);
   }
   explicit RField(std::string_view name) : RField(name, BuildItemFields<T1, T2>()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::tuple
////////////////////////////////////////////////////////////////////////////////

/// The generic field for `std::tuple<Ts...>` types
class RTupleField : public RRecordField {
private:
   class RTupleDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RTupleDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   static std::string GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override { return std::make_unique<RTupleDeleter>(fClass); }

   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
               const std::vector<std::size_t> &offsets);

public:
   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &itemFields);
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
   template <typename... Ts>
   static std::vector<std::unique_ptr<RFieldBase>> BuildItemFields()
   {
      std::vector<std::unique_ptr<RFieldBase>> result;
      _BuildItemFields<Ts...>(result);
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
   template <typename... Ts>
   static std::vector<std::size_t> BuildItemOffsets()
   {
      std::vector<std::size_t> result;
      _BuildItemOffsets<0, Ts...>(result, ContainerT());
      return result;
   }

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      std::vector<std::unique_ptr<RFieldBase>> items;
      for (auto &item : fSubFields)
         items.push_back(item->Clone(item->GetFieldName()));
      return std::make_unique<RField<std::tuple<ItemTs...>>>(newName, std::move(items));
   }

   void ConstructValue(void *where) const final { new (where) ContainerT(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<ContainerT>>(); }

public:
   static std::string TypeName() { return "std::tuple<" + BuildItemTypes<ItemTs...>() + ">"; }
   explicit RField(std::string_view name, std::vector<std::unique_ptr<RFieldBase>> &&itemFields)
      : RTupleField(name, std::move(itemFields), BuildItemOffsets<ItemTs...>())
   {
      fMaxAlignment = std::max({alignof(ItemTs)...});
      fSize = sizeof(ContainerT);
   }
   explicit RField(std::string_view name) : RField(name, BuildItemFields<ItemTs...>()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

} // namespace Experimental
} // namespace ROOT

#endif
