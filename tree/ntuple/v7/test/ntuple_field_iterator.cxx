#include "ntuple_test.hxx"
using RFieldDescriptorRange = ROOT::Experimental::RFieldDescriptorRange;
using ENTupleMergeable = ROOT::Experimental::RNTupleDescriptor::ENTupleMergeable;

TEST(RNTupleDescriptor, FieldIterator)
{
    auto model = RNTupleModel::Create();
    auto floats = model->MakeField<std::vector<float>>("jets");
    auto bools = model->MakeField<std::vector<bool>>("bools");
    auto bool_vec_vec = model->MakeField<std::vector<std::vector<bool>>>("bool_vec_vec");
    auto ints = model->MakeField<std::int32_t>("ints");

    FileRaii fileGuard("test_field_iterator.root");
    auto modelRead = std::unique_ptr<RNTupleModel>(model->Clone());
    {
        RNTupleWriter ntuple(std::move(model),
           std::make_unique<RPageSinkFile>("ntuple", fileGuard.GetPath(), RNTupleWriteOptions()));
        ntuple.Fill();
    }

    RNTupleReader ntuple(std::move(modelRead),
       std::make_unique<RPageSourceFile>("ntuple", fileGuard.GetPath(), RNTupleReadOptions()));

    printf("-- printing top-level fields\n");
    const auto& ntuple_desc = ntuple.GetDescriptor();
    for (auto f: ntuple_desc.GetTopLevelFields()) {
       printf("top level field id: %lu\n", f);
       printf("\tname: %s\n", ntuple_desc.GetFieldDescriptor(f).GetFieldName().c_str());
    }

    printf("\n-- printing first two levels\n");
    for (auto f: ntuple_desc.GetTopLevelFields()) {
       printf("top level field id: %lu\n", f);
       for (auto child_field: ntuple_desc.GetFieldRange(f)) {
          printf("\tsecond level field id: %lu\n", child_field);
       }
    }

    // todo(max) tests
}

TEST(RNTupleDescriptor, IsMergeable)
{
    FileRaii fileGuard("test_field_iterator1.root");
    FileRaii fileGuard2("test_field_iterator2.root");

    auto model = RNTupleModel::Create();
    auto floats = model->MakeField<std::vector<float>>("jets");
    auto bools = model->MakeField<std::vector<bool>>("bools");
    auto ints = model->MakeField<std::int32_t>("ints");

    auto model2 = std::unique_ptr<RNTupleModel>(model->Clone());
    auto modelRead = std::unique_ptr<RNTupleModel>(model->Clone());
    auto modelRead2 = std::unique_ptr<RNTupleModel>(model->Clone());

    {
        RNTupleWriter ntuple(std::move(model),
            std::make_unique<RPageSinkFile>("ntuple1", fileGuard.GetPath(), RNTupleWriteOptions()));
        ntuple.Fill();

        RNTupleWriter ntuple2(std::move(model2),
            std::make_unique<RPageSinkFile>("ntuple2", fileGuard2.GetPath(), RNTupleWriteOptions()));
        ntuple2.Fill();
    }

    // mismatched model
    auto different_model = RNTupleModel::Create();
    auto different_model_read = std::unique_ptr<RNTupleModel>(different_model->Clone());
    auto different_floats = different_model->MakeField<std::vector<float>>("jorts");
    FileRaii fileGuard3("test_field_iterator3.root");
    {
        RNTupleWriter ntuple(std::move(different_model),
            std::make_unique<RPageSinkFile>("ntuple3", fileGuard3.GetPath(), RNTupleWriteOptions()));
        ntuple.Fill();
    }

    RNTupleReader ntuple(std::move(modelRead),
      std::make_unique<RPageSourceFile>("ntuple1", fileGuard.GetPath(), RNTupleReadOptions()));

    RNTupleReader ntuple_copy(std::move(modelRead2),
      std::make_unique<RPageSourceFile>("ntuple2", fileGuard2.GetPath(), RNTupleReadOptions()));

    RNTupleReader different_ntuple(std::move(different_model_read),
      std::make_unique<RPageSourceFile>("ntuple3", fileGuard3.GetPath(), RNTupleReadOptions()));

    // mergeable with itself
    EXPECT_TRUE(ENTupleMergeable::Mergeable
       == ntuple.GetDescriptor().IsMergeable(ntuple_copy.GetDescriptor()));
    // not mergeable with a ntuple with different top-level fields
    EXPECT_TRUE(ENTupleMergeable::StructureMismatch
       == ntuple.GetDescriptor().IsMergeable(different_ntuple.GetDescriptor()));
}
