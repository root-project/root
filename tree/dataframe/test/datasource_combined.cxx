#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>
#include <ROOT/RCombinedDS.hxx>
#include <ROOT/TSeq.hxx>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/test-util.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <gtest/gtest.h>

#include <iostream>

using namespace ROOT;
using namespace ROOT::RDF;
using namespace arrow;

std::shared_ptr<Schema> exampleSchema()
{
   return schema({field("Numbers", arrow::int64())});
}

std::shared_ptr<Table> createTestTable()
{
   auto schema_ = exampleSchema();

   std::vector<int64_t> numbers = {0, 1, 2, 3, 4, 5, 6, 7};
   std::shared_ptr<Array> arrays_[1];
   arrow::ArrayFromVector<Int64Type, int64_t>(numbers, &arrays_[0]);

   std::vector<std::shared_ptr<Column>> columns_ = {
      std::make_shared<Column>(schema_->field(0), arrays_[0])
   };

   auto table_ = Table::Make(schema_, columns_);
   return table_;
}

TEST(RCombinedDS, CrossJoin)
{
   auto table = createTestTable();
   auto left = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
   auto right = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
   auto combined = std::make_unique<RCombinedDS>(std::move(left), std::move(right));
   
   ROOT::RDataFrame rdf(std::move(combined));

   EXPECT_EQ(*rdf.Count(), 64);
   EXPECT_EQ(*rdf.Define("sum", "left_Numbers + right_Numbers").Sum("sum"), 448);
}
