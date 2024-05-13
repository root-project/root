#include "ntuple_test.hxx"

#include <TMath.h>
#include <TObject.h>
#include <TRef.h>
#include <TRotation.h>

#include <memory>
#include <sstream>

namespace {
class RNoDictionary {};
} // namespace

namespace ROOT::Experimental {
template <>
struct IsCollectionProxy<CyclicCollectionProxy> : std::true_type {
};
} // namespace ROOT::Experimental

TEST(RNTuple, TClass) {
   auto modelFail = RNTupleModel::Create();
   EXPECT_THROW(modelFail->MakeField<RNoDictionary>("nodict"), ROOT::Experimental::RException);

   auto model = RNTupleModel::Create();
   auto ptrKlass = model->MakeField<CustomStruct>("klass");

   // TDatime would be a supported class layout but it is unsplittable due to its custom streamer
   EXPECT_THROW(model->MakeField<TDatime>("datime"), RException);

   FileRaii fileGuard("test_ntuple_tclass.root");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
}

TEST(RNTuple, CyclicClass)
{
   auto modelFail = RNTupleModel::Create();
   EXPECT_THROW(modelFail->MakeField<Cyclic>("cyclic"), ROOT::Experimental::RException);

   CyclicCollectionProxy ccp;
   auto cl = TClass::GetClass("CyclicCollectionProxy");
   cl->CopyCollectionProxy(ccp);
   EXPECT_THROW(RFieldBase::Create("f", "CyclicCollectionProxy").Unwrap(), ROOT::Experimental::RException);
}

TEST(RNTuple, DiamondInheritance)
{
   FileRaii fileGuard("test_ntuple_diamond_inheritance.root");

   {
      auto model = RNTupleModel::Create();
      auto d = model->MakeField<DuplicateBaseD>("d");
      EXPECT_THROW(model->MakeField<DiamondVirtualD>("vd"), RException);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      d->DuplicateBaseB::a = 1.0;
      d->DuplicateBaseC::a = 1.5;
      d->b = 2.0;
      d->c = 3.0;
      d->d = 4.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto d = reader->GetModel().GetDefaultEntry().GetPtr<DuplicateBaseD>("d");
   EXPECT_EQ(1u, reader->GetNEntries());

   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, d->DuplicateBaseB::a);
   EXPECT_FLOAT_EQ(1.5, d->DuplicateBaseC::a);
   EXPECT_FLOAT_EQ(2.0, d->b);
   EXPECT_FLOAT_EQ(3.0, d->c);
   EXPECT_FLOAT_EQ(4.0, d->d);
}

TEST(RTNuple, TObject)
{
   // Ensure that TObject cannot be accidentally handled through the generic RClassField field
   EXPECT_THROW(std::make_unique<ROOT::Experimental::RClassField>("obj", "TObject"), RException);

   FileRaii fileGuard("test_ntuple_tobject.root");
   {
      auto model = RNTupleModel::Create();
      model->MakeField<TObject>("obj");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto entry = writer->GetModel().CreateBareEntry();

      auto heapObj = std::make_unique<TObject>();
      EXPECT_TRUE(heapObj->TestBit(TObject::kIsOnHeap));
      EXPECT_TRUE(heapObj->TestBit(TObject::kNotDeleted));
      heapObj->SetUniqueID(137);
      entry->BindRawPtr("obj", heapObj.get());
      writer->Fill(*entry);

      // Saving a destructed object is here to verify that the RNTuple serialization does the same than
      // the TObject custom streamer (i.e., ignoring the kNotDeleted flag)
      heapObj->~TObject();
      EXPECT_FALSE(heapObj->TestBit(TObject::kNotDeleted));
      writer->Fill(*entry);

      TObject stackObj;
      EXPECT_FALSE(stackObj.TestBit(TObject::kIsOnHeap));
      entry->BindRawPtr("obj", &stackObj);
      writer->Fill(*entry);
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(3u, reader->GetNEntries());

   auto entry = reader->GetModel().CreateBareEntry();
   TObject stackObj;
   entry->BindRawPtr("obj", &stackObj);

   reader->LoadEntry(0, *entry);
   EXPECT_EQ(137u, stackObj.GetUniqueID());
   EXPECT_FALSE(stackObj.TestBit(TObject::kIsOnHeap));
   EXPECT_TRUE(stackObj.TestBit(TObject::kNotDeleted));

   reader->LoadEntry(1, *entry);
   EXPECT_EQ(137u, stackObj.GetUniqueID());
   EXPECT_FALSE(stackObj.TestBit(TObject::kIsOnHeap));
   EXPECT_TRUE(stackObj.TestBit(TObject::kNotDeleted));

   auto heapObj = std::make_unique<TObject>();
   entry->BindRawPtr("obj", heapObj.get());
   reader->LoadEntry(2, *entry);
   EXPECT_EQ(0u, heapObj->GetUniqueID());
   EXPECT_TRUE(heapObj->TestBit(TObject::kIsOnHeap));
   EXPECT_TRUE(heapObj->TestBit(TObject::kNotDeleted));
}

TEST(RTNuple, TObjectReferenced)
{
   // RNTuple does not support reading nor writing of TObject objects that are marked as referenced
   FileRaii fileGuard("test_ntuple_tobject_referenced.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrObject = model->MakeField<TObject>("obj");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      ptrObject->SetBit(TObject::kIsReferenced);
      EXPECT_THROW(writer->Fill(), RException);
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   auto ptrObject = reader->GetModel().GetDefaultEntry().GetPtr<TObject>("obj");

   reader->LoadEntry(0);
   EXPECT_EQ(0u, ptrObject->GetUniqueID());
   ptrObject->SetBit(TObject::kIsReferenced);
   EXPECT_THROW(reader->LoadEntry(0), RException);
}

TEST(RTNuple, TObjectShow)
{
   FileRaii fileGuard("test_ntuple_tobject_show.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrObject = model->MakeField<TObject>("obj");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrObject->SetUniqueID(137);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   // clang-format off
   std::string expected{
      "{\n"
      "  \"obj\": {\n"
      "    \"fUniqueID\": 137,\n"
      "    \"fBits\": 33554432\n"
      "  }\n"
      "}\n"
   };
   // clang-format on
   std::ostringstream os;
   reader->Show(0, os);
   EXPECT_EQ(expected, os.str());
}

TEST(RTNuple, TObjectDerived)
{
   FileRaii fileGuard("test_ntuple_tobject_derived.root");

   {
      auto model = RNTupleModel::Create();
      // The choice of TRotation is arbitrary; it is a simple, existing class that inherits from TObject
      // and is supported by RNTuple
      auto ptrRotation = model->MakeField<TRotation>("rotation");
      ptrRotation->RotateX(TMath::Pi());
      ptrRotation->SetUniqueID(137);
      auto ptrMultiple = model->MakeField<DerivedFromLeftAndTObject>("derived");
      ptrMultiple->SetUniqueID(137);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());

   auto ptrRotation = reader->GetModel().GetDefaultEntry().GetPtr<TRotation>("rotation");
   auto ptrMultiple = reader->GetModel().GetDefaultEntry().GetPtr<DerivedFromLeftAndTObject>("derived");
   reader->LoadEntry(0);

   EXPECT_DOUBLE_EQ(1.0, ptrRotation->XX());
   EXPECT_EQ(137u, ptrRotation->GetUniqueID());

   EXPECT_FLOAT_EQ(1.0, ptrMultiple->x);
   EXPECT_EQ(137u, ptrMultiple->GetUniqueID());
}
