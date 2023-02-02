#ifndef ROOT7_RNTuple_Test_SimpleCollectionProxy
#define ROOT7_RNTuple_Test_SimpleCollectionProxy

#include "ntuple_test.hxx"

namespace {
/// Simple collection proxy for `StructUsingCollectionProxy<T>`
template <typename CollectionT>
struct SimpleCollectionProxy : public TVirtualCollectionProxy {
   CollectionT *fObject = nullptr;

   SimpleCollectionProxy()
      : TVirtualCollectionProxy(TClass::GetClass(ROOT::Internal::GetDemangledTypeName(typeid(CollectionT)).c_str()))
   {
   }
   SimpleCollectionProxy(const SimpleCollectionProxy &) : SimpleCollectionProxy() {}

   TVirtualCollectionProxy *Generate() const override { return new SimpleCollectionProxy<CollectionT>(*this); }
   Int_t GetCollectionType() const override { return ROOT::kSTLvector; }
   ULong_t GetIncrement() const override { return sizeof(typename CollectionT::ValueType); }
   UInt_t Sizeof() const override { return sizeof(CollectionT); }
   Bool_t HasPointers() const override { return kFALSE; }

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
   void *Allocate(UInt_t /*n*/, Bool_t /*forceDelete*/) override { return nullptr; }
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

   CreateIterators_t GetFunctionCreateIterators(Bool_t /*read*/ = kTRUE) override { return nullptr; }
   CopyIterator_t GetFunctionCopyIterator(Bool_t /*read*/ = kTRUE) override { return nullptr; }
   Next_t GetFunctionNext(Bool_t /*read*/ = kTRUE) override { return nullptr; }
   DeleteIterator_t GetFunctionDeleteIterator(Bool_t /*read*/ = kTRUE) override { return nullptr; }
   DeleteTwoIterators_t GetFunctionDeleteTwoIterators(Bool_t /*read*/ = kTRUE) override { return nullptr; }
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
