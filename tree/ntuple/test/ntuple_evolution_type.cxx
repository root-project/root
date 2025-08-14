#include "ntuple_test.hxx"

#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

class RNewStringField final : public ROOT::RFieldBase {
protected:
   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RNewStringField>(newName);
   }
   void ConstructValue(void *where) const final { new (where) std::string(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<std::string>>(); }

public:
   RNewStringField(std::string_view name)
      : ROOT::RFieldBase(name, "std::string", ROOT::ENTupleStructure::kLeaf, false /* isSimple */) {}
   RNewStringField(RNewStringField &&other) = default;
   RNewStringField &operator=(RNewStringField &&other) = default;
   ~RNewStringField() override = default;

   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations({{ROOT::ENTupleColumnType::kSplitIndex64}}, {});
      return representations;
   }
   // Field is only used for reading
   void GenerateColumns() final { GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(); }
   void GenerateColumns(const ROOT::RNTupleDescriptor & /* desc */) final
   {
      throw ROOT::RException(R__FAIL("no read support"));
   }

   size_t GetValueSize() const final { return sizeof(std::string); }
   size_t GetAlignment() const final { return alignof(std::string); }
};

TEST(RNTupleEvolution, CheckFieldVersion)
{
   FileRaii fileGuard("test_ntuple_evolution_field_version.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RNewStringField>("f"));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(0, reader->GetNEntries());

   reader->GetView<std::string>("f");
}
