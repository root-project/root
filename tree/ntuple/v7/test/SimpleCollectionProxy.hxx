#ifndef ROOT7_RNTuple_Test_SimpleCollectionProxy
#define ROOT7_RNTuple_Test_SimpleCollectionProxy

#include "ntuple_test.hxx"

namespace {
/// Simple collection proxy for `StructUsingCollectionProxy<T>`
template <typename CollectionT, Int_t CollectionKind = ROOT::kSTLvector>
class SimpleCollectionProxy : public TVirtualCollectionProxy {
   /// The internal representation of an iterator, which in this simple test only contains a pointer to an element
   struct IteratorData {
      typename CollectionT::ValueType *ptr;
   };

   static void
   Func_CreateIterators(void *collection, void **begin_arena, void **end_arena, TVirtualCollectionProxy * /*proxy*/)
   {
      static_assert(sizeof(IteratorData) <= TVirtualCollectionProxy::fgIteratorArenaSize);
      auto &vec = static_cast<CollectionT *>(collection)->v;
      if constexpr (CollectionKind == ROOT::kSTLvector) {
         // An iterator on an array-backed container is just a pointer; thus, it can be directly stored in `*xyz_arena`,
         // saving one dereference (see TVirtualCollectionProxy documentation)
         *begin_arena = vec.data();
         *end_arena = vec.data() + vec.size();
      } else {
         static_cast<IteratorData *>(*begin_arena)->ptr = vec.data();
         static_cast<IteratorData *>(*end_arena)->ptr = vec.data() + vec.size();
      }
   }

   static void *Func_Next(void *iter, const void *end)
   {
      auto _iter = static_cast<IteratorData *>(iter);
      auto _end = static_cast<const IteratorData *>(end);
      if (_iter->ptr >= _end->ptr)
         return nullptr;
      return _iter->ptr++;
   }

   static void Func_DeleteTwoIterators(void * /*begin*/, void * /*end*/) {}

private:
   CollectionT *fObject = nullptr;

public:
   SimpleCollectionProxy()
      : TVirtualCollectionProxy(TClass::GetClass(ROOT::Internal::GetDemangledTypeName(typeid(CollectionT)).c_str()))
   {
   }
   SimpleCollectionProxy(const SimpleCollectionProxy<CollectionT, CollectionKind> &) : SimpleCollectionProxy() {}

   TVirtualCollectionProxy *Generate() const override
   {
      return new SimpleCollectionProxy<CollectionT, CollectionKind>(*this);
   }
   Int_t GetCollectionType() const override { return CollectionKind; }
   ULong_t GetIncrement() const override { return sizeof(typename CollectionT::ValueType); }
   UInt_t Sizeof() const override { return sizeof(CollectionT); }
   bool HasPointers() const override { return false; }

   TClass *GetValueClass() const override
   {
      if constexpr (std::is_fundamental<typename CollectionT::ValueType>::value)
         return nullptr;
      return TClass::GetClass(ROOT::Internal::GetDemangledTypeName(typeid(typename CollectionT::ValueType)).c_str());
   }
   EDataType GetType() const override
   {
      if constexpr (std::is_same<typename CollectionT::ValueType, char>::value)
         return EDataType::kChar_t;
      ;
      if constexpr (std::is_same<typename CollectionT::ValueType, float>::value)
         return EDataType::kFloat_t;
      ;
      return EDataType::kOther_t;
   }

   void PushProxy(void *objectstart) override { fObject = static_cast<CollectionT *>(objectstart); }
   void PopProxy() override { fObject = nullptr; }

   void *At(UInt_t idx) override { return &fObject->v[idx]; }
   void Clear(const char * /*opt*/ = "") override { fObject->v.clear(); }
   UInt_t Size() const override { return fObject->v.size(); }
   void *Allocate(UInt_t n, bool /*forceDelete*/) override
   {
      fObject->v.resize(n);
      return fObject;
   }
   void Commit(void *) override {}
   void Insert(const void *data, void *container, size_t size) override
   {
      auto p = static_cast<const typename CollectionT::ValueType *>(data);
      for (size_t i = 0; i < size; ++i) {
         static_cast<CollectionT *>(container)->v.push_back(p[i]);
      }
   }

   TStreamerInfoActions::TActionSequence *
   GetConversionReadMemberWiseActions(TClass * /*oldClass*/, Int_t /*version*/) override
   {
      return nullptr;
   }
   TStreamerInfoActions::TActionSequence *GetReadMemberWiseActions(Int_t /*version*/) override { return nullptr; }
   TStreamerInfoActions::TActionSequence *GetWriteMemberWiseActions() override { return nullptr; }

   CreateIterators_t GetFunctionCreateIterators(bool /*read*/ = true) override { return &Func_CreateIterators; }
   CopyIterator_t GetFunctionCopyIterator(bool /*read*/ = true) override { return nullptr; }
   Next_t GetFunctionNext(bool /*read*/ = true) override { return &Func_Next; }
   DeleteIterator_t GetFunctionDeleteIterator(bool /*read*/ = true) override { return nullptr; }
   DeleteTwoIterators_t GetFunctionDeleteTwoIterators(bool /*read*/ = true) override
   {
      return &Func_DeleteTwoIterators;
   }
};
} // namespace

namespace ROOT::Experimental {
template <>
struct IsCollectionProxy<StructUsingCollectionProxy<char>> : std::true_type {
};
template <>
struct IsCollectionProxy<StructUsingCollectionProxy<float>> : std::true_type {
};
template <>
struct IsCollectionProxy<StructUsingCollectionProxy<CustomStruct>> : std::true_type {
};

template <>
struct IsCollectionProxy<StructUsingCollectionProxy<StructUsingCollectionProxy<float>>> : std::true_type {
};

// Intentionally omit `IsCollectionProxy<StructUsingCollectionProxy<int>>`
} // namespace ROOT::Experimental

#endif
