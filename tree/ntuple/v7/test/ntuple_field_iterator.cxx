#include "ntuple_test.hxx"
using RFieldDescriptorRange = ROOT::Experimental::RFieldDescriptorRange;

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
}
