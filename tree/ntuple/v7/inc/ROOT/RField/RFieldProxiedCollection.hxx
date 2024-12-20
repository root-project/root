/// \file ROOT/RField/ProxiedCollection.hxx
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

#ifndef ROOT7_RField_ProxiedCollection
#define ROOT7_RField_ProxiedCollection

#ifndef ROOT7_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TVirtualCollectionProxy.h>

#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

/// The field for a class representing a collection of elements via `TVirtualCollectionProxy`.
/// Objects of such type behave as collections that can be accessed through the corresponding member functions in
/// `TVirtualCollectionProxy`. For STL collections, these proxies are provided. Custom classes need to implement the
/// corresponding member functions in `TVirtualCollectionProxy`. At a bare minimum, the user is required to provide an
/// implementation for the following functions in `TVirtualCollectionProxy`: `HasPointers()`, `GetProperties()`,
/// `GetValueClass()`, `GetType()`, `PushProxy()`, `PopProxy()`, `GetFunctionCreateIterators()`, `GetFunctionNext()`,
/// and `GetFunctionDeleteTwoIterators()`.
///
/// The collection proxy for a given class can be set via `TClass::CopyCollectionProxy()`.
class RProxiedCollectionField : public RFieldBase {
protected:
   /// Allows for iterating over the elements of a proxied collection. RCollectionIterableOnce avoids an additional
   /// iterator copy (see `TVirtualCollectionProxy::GetFunctionCopyIterator`) and thus can only be iterated once.
   class RCollectionIterableOnce {
   public:
      struct RIteratorFuncs {
         TVirtualCollectionProxy::CreateIterators_t fCreateIterators;
         TVirtualCollectionProxy::DeleteTwoIterators_t fDeleteTwoIterators;
         TVirtualCollectionProxy::Next_t fNext;
      };
      static RIteratorFuncs GetIteratorFuncs(TVirtualCollectionProxy *proxy, bool readFromDisk);

   private:
      class RIterator {
         const RCollectionIterableOnce &fOwner;
         void *fIterator = nullptr;
         void *fElementPtr = nullptr;

         void Advance()
         {
            auto fnNext_Contig = [&]() {
               // Array-backed collections (e.g. kSTLvector) directly use the pointer-to-iterator-data as a
               // pointer-to-element, thus saving an indirection level (see documentation for TVirtualCollectionProxy)
               auto &iter = reinterpret_cast<unsigned char *&>(fIterator), p = iter;
               iter += fOwner.fStride;
               return p;
            };
            fElementPtr = fOwner.fStride ? fnNext_Contig() : fOwner.fIFuncs.fNext(fIterator, fOwner.fEnd);
         }

      public:
         using iterator_category = std::forward_iterator_tag;
         using iterator = RIterator;
         using difference_type = std::ptrdiff_t;
         using pointer = void *;

         RIterator(const RCollectionIterableOnce &owner) : fOwner(owner) {}
         RIterator(const RCollectionIterableOnce &owner, void *iter) : fOwner(owner), fIterator(iter) { Advance(); }
         iterator operator++()
         {
            Advance();
            return *this;
         }
         pointer operator*() const { return fElementPtr; }
         bool operator!=(const iterator &rh) const { return fElementPtr != rh.fElementPtr; }
         bool operator==(const iterator &rh) const { return fElementPtr == rh.fElementPtr; }
      };

      const RIteratorFuncs &fIFuncs;
      const std::size_t fStride;
      unsigned char fBeginSmallBuf[TVirtualCollectionProxy::fgIteratorArenaSize];
      unsigned char fEndSmallBuf[TVirtualCollectionProxy::fgIteratorArenaSize];
      void *fBegin = &fBeginSmallBuf;
      void *fEnd = &fEndSmallBuf;

   public:
      /// Construct a `RCollectionIterableOnce` that iterates over `collection`.  If elements are guaranteed to be
      /// contiguous in memory (e.g. a vector), `stride` can be provided for faster iteration, i.e. the address of each
      /// element is known given the base pointer.
      RCollectionIterableOnce(void *collection, const RIteratorFuncs &ifuncs, TVirtualCollectionProxy *proxy,
                              std::size_t stride = 0U)
         : fIFuncs(ifuncs), fStride(stride)
      {
         fIFuncs.fCreateIterators(collection, &fBegin, &fEnd, proxy);
      }
      ~RCollectionIterableOnce() { fIFuncs.fDeleteTwoIterators(fBegin, fEnd); }

      RIterator begin() { return RIterator(*this, fBegin); }
      RIterator end() { return fStride ? RIterator(*this, fEnd) : RIterator(*this); }
   }; // class RCollectionIterableOnce

   class RProxiedCollectionDeleter : public RDeleter {
   private:
      std::shared_ptr<TVirtualCollectionProxy> fProxy;
      std::unique_ptr<RDeleter> fItemDeleter;
      std::size_t fItemSize = 0;
      RCollectionIterableOnce::RIteratorFuncs fIFuncsWrite;

   public:
      explicit RProxiedCollectionDeleter(std::shared_ptr<TVirtualCollectionProxy> proxy) : fProxy(proxy) {}
      RProxiedCollectionDeleter(std::shared_ptr<TVirtualCollectionProxy> proxy, std::unique_ptr<RDeleter> itemDeleter,
                                size_t itemSize)
         : fProxy(proxy), fItemDeleter(std::move(itemDeleter)), fItemSize(itemSize)
      {
         fIFuncsWrite = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), false /* readFromDisk */);
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   /// The collection proxy is needed by the deleters and thus defined as a shared pointer
   std::shared_ptr<TVirtualCollectionProxy> fProxy;
   Int_t fProperties;
   Int_t fCollectionType;
   /// Two sets of functions to operate on iterators, to be used depending on the access type.  The direction preserves
   /// the meaning from TVirtualCollectionProxy, i.e. read from disk / write to disk, respectively
   RCollectionIterableOnce::RIteratorFuncs fIFuncsRead;
   RCollectionIterableOnce::RIteratorFuncs fIFuncsWrite;
   std::size_t fItemSize;
   ClusterSize_t fNWritten;

   /// Constructor used when the value type of the collection is not known in advance, i.e. in the case of custom
   /// collections.
   RProxiedCollectionField(std::string_view fieldName, std::string_view typeName, TClass *classp);
   /// Constructor used when the value type of the collection is known in advance, e.g. in `RSetField`.
   RProxiedCollectionField(std::string_view fieldName, std::string_view typeName,
                           std::unique_ptr<RFieldBase> itemField);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;
   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RProxiedCollectionField(std::string_view fieldName, std::string_view typeName);
   RProxiedCollectionField(RProxiedCollectionField &&other) = default;
   RProxiedCollectionField &operator=(RProxiedCollectionField &&other) = default;
   ~RProxiedCollectionField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return fProxy->Sizeof(); }
   size_t GetAlignment() const final { return alignof(std::max_align_t); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
   void GetCollectionInfo(NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RClusterIndex clusterIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(clusterIndex, collectionStart, size);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for classes with collection proxies
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename = void>
struct HasCollectionProxyMemberType : std::false_type {
};
template <typename T>
struct HasCollectionProxyMemberType<
   T, typename std::enable_if<std::is_same<typename T::IsCollectionProxy, std::true_type>::value>::type>
   : std::true_type {
};

/// The point here is that we can only tell at run time if a class has an associated collection proxy.
/// For compile time, in the first iteration of this PR we had an extra template argument that acted as a "tag" to
/// differentiate the RField specialization for classes with an associated collection proxy (inherits
/// `RProxiedCollectionField`) from the RField primary template definition (`RClassField`-derived), as in:
/// ```
/// auto field = std::make_unique<RField<MyClass>>("klass");
/// // vs
/// auto otherField = std::make_unique<RField<MyClass, ROOT::Experimental::TagIsCollectionProxy>>("klass");
/// ```
///
/// That is convenient only for non-nested types, i.e. it doesn't work with, e.g. `RField<std::vector<MyClass>,
/// ROOT::Experimental::TagIsCollectionProxy>`, as the tag is not forwarded to the instantiation of the inner RField
/// (that for the value type of the vector).  The following two possible solutions were considered:
/// - A wrapper type (much like `ntuple/v7/inc/ROOT/RNTupleUtil.hxx:49`), that helps to differentiate both cases.
/// There we would have:
/// ```
/// auto field = std::make_unique<RField<RProxiedCollection<MyClass>>>("klass"); // Using collection proxy
/// ```
/// - A helper `IsCollectionProxy<T>` type, that can be used in a similar way to those in the `<type_traits>` header.
/// We found this more convenient and is the implemented thing below.  Here, classes can be marked as a
/// collection proxy with either of the following two forms (whichever is more convenient for the user):
/// ```
/// template <>
/// struct IsCollectionProxy<MyClass> : std::true_type {};
/// ```
/// or by adding a member type to the class as follows:
/// ```
/// class MyClass {
/// public:
///    using IsCollectionProxy = std::true_type;
/// };
/// ```
///
/// Of course, there is another possible solution which is to have a single `RClassField` that implements both
/// the regular-class and the collection-proxy behaviors, and always chooses appropriately at run time.
/// We found that less clean and probably has more overhead, as most probably it involves an additional branch + call
/// in each of the member functions.
template <typename T, typename = void>
struct IsCollectionProxy : HasCollectionProxyMemberType<T> {
};

/// Classes behaving as a collection of elements that can be queried via the `TVirtualCollectionProxy` interface
/// The use of a collection proxy for a particular class can be enabled via:
/// ```
/// namespace ROOT::Experimental {
///    template <> struct IsCollectionProxy<Classname> : std::true_type {};
/// }
/// ```
/// Alternatively, this can be achieved by adding a member type to the class definition as follows:
/// ```
/// class Classname {
/// public:
///    using IsCollectionProxy = std::true_type;
/// };
/// ```
template <typename T>
class RField<T, typename std::enable_if<IsCollectionProxy<T>::value>::type> final : public RProxiedCollectionField {
public:
   static std::string TypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : RProxiedCollectionField(name, TypeName())
   {
      static_assert(std::is_class<T>::value, "collection proxy unsupported for fundamental types");
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::[unordered_][multi]map
////////////////////////////////////////////////////////////////////////////////

/// The generic field for a std::map<KeyType, ValueType> and std::unordered_map<KeyType, ValueType>
class RMapField : public RProxiedCollectionField {
public:
   RMapField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);
   RMapField(RMapField &&other) = default;
   RMapField &operator=(RMapField &&other) = default;
   ~RMapField() override = default;
};

template <typename KeyT, typename ValueT>
class RField<std::map<KeyT, ValueT>> final : public RMapField {
public:
   static std::string TypeName()
   {
      return "std::map<" + RField<KeyT>::TypeName() + "," + RField<ValueT>::TypeName() + ">";
   }

   explicit RField(std::string_view name)
      : RMapField(name, TypeName(), std::make_unique<RField<std::pair<KeyT, ValueT>>>("_0"))
   {
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename KeyT, typename ValueT>
class RField<std::unordered_map<KeyT, ValueT>> final : public RMapField {
public:
   static std::string TypeName()
   {
      return "std::unordered_map<" + RField<KeyT>::TypeName() + "," + RField<ValueT>::TypeName() + ">";
   }

   explicit RField(std::string_view name)
      : RMapField(name, TypeName(), std::make_unique<RField<std::pair<KeyT, ValueT>>>("_0"))
   {
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename KeyT, typename ValueT>
class RField<std::multimap<KeyT, ValueT>> final : public RMapField {
public:
   static std::string TypeName()
   {
      return "std::multimap<" + RField<KeyT>::TypeName() + "," + RField<ValueT>::TypeName() + ">";
   }

   explicit RField(std::string_view name)
      : RMapField(name, TypeName(), std::make_unique<RField<std::pair<KeyT, ValueT>>>("_0"))
   {
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename KeyT, typename ValueT>
class RField<std::unordered_multimap<KeyT, ValueT>> final : public RMapField {
public:
   static std::string TypeName()
   {
      return "std::unordered_multimap<" + RField<KeyT>::TypeName() + "," + RField<ValueT>::TypeName() + ">";
   }

   explicit RField(std::string_view name)
      : RMapField(name, TypeName(), std::make_unique<RField<std::pair<KeyT, ValueT>>>("_0"))
   {
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::[unordered_][multi]set
////////////////////////////////////////////////////////////////////////////////

/// The generic field for a std::set<Type> and std::unordered_set<Type>
class RSetField : public RProxiedCollectionField {
public:
   RSetField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);
   RSetField(RSetField &&other) = default;
   RSetField &operator=(RSetField &&other) = default;
   ~RSetField() override = default;
};

template <typename ItemT>
class RField<std::set<ItemT>> final : public RSetField {
public:
   static std::string TypeName() { return "std::set<" + RField<ItemT>::TypeName() + ">"; }

   explicit RField(std::string_view name) : RSetField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename ItemT>
class RField<std::unordered_set<ItemT>> final : public RSetField {
public:
   static std::string TypeName() { return "std::unordered_set<" + RField<ItemT>::TypeName() + ">"; }

   explicit RField(std::string_view name) : RSetField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename ItemT>
class RField<std::multiset<ItemT>> final : public RSetField {
public:
   static std::string TypeName() { return "std::multiset<" + RField<ItemT>::TypeName() + ">"; }

   explicit RField(std::string_view name) : RSetField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename ItemT>
class RField<std::unordered_multiset<ItemT>> final : public RSetField {
public:
   static std::string TypeName() { return "std::unordered_multiset<" + RField<ItemT>::TypeName() + ">"; }

   explicit RField(std::string_view name) : RSetField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

} // namespace Experimental
} // namespace ROOT

#endif
