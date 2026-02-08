#include "TBufferFile.h"
#include "TClass.h"
#include "TMacro.h"
#include "TNamed.h"
#include "TProtoClass.h"

#include "gtest/gtest.h"

namespace {

struct ReadResult {
   int      fError = 0;
   TObject *fObj = nullptr;
};

void Update(int &errors, std::unique_ptr<TObject> &obj, const ReadResult &res)
{
   errors += res.fError;
   obj.reset(res.fObj);
}

ReadResult ReadAndCheck(TBuffer &b, TClass *cl, TObject *ident = nullptr)
{
   ReadResult result;

   b >> result.fObj;

   EXPECT_NE(result.fObj, nullptr)
      << "Failed to read object of class '" << cl->GetName() << "'";
   if (!result.fObj) {
      result.fError = 1;
      return result;
   }

   EXPECT_EQ(result.fObj->IsA(), cl)
         << "Expected class '" << cl->GetName()
         << "' but read '" << result.fObj->IsA()->GetName() << "'";
   if (result.fObj->IsA() != cl) {
      result.fError = 2;
      return result;
   }

   if (ident) {
      EXPECT_EQ(ident, result.fObj);
      if (ident != result.fObj) {
         result.fError = 3;
         return result;
      }
   }

   return result;
}

} // anonymous namespace

TEST(TBufferFileInnerReferences, LargeOffsetsAndReferences)
{
   int errors = 0;

   auto n0 = std::make_unique<TNamed>("n0", "At start");
   auto n1 = std::make_unique<TNamed>("n1", "Below 1G");
   auto n2 = std::make_unique<TNamed>("n2", "Over 1G");
   auto m1 = std::make_unique<TMacro>("m1", "Below 1G");
   auto m2 = std::make_unique<TMacro>("m2", "Over 1G");
   auto c1 = std::make_unique<TProtoClass>(); // Only over 1G
   auto c2 = std::make_unique<TProtoClass>(); // Also over 1G

   // TBufferFile currently rejects sizes larger than 2GB.
   // SetBufferOffset does not check against the size,
   // so we can provide and use a larger buffer.
   std::vector<char> databuffer{};
   databuffer.reserve(4ull * 1024 * 1024 * 1024);
   TBufferFile b(TBuffer::kWrite, 2ull * 1024 * 1024 * 1024 - 100, databuffer.data(), false /* don't adopt */);

   b << n0.get();
   b.SetBufferOffset(512ull * 1024 * 1024);
   b << n1.get();
   b << m1.get();
   b.SetBufferOffset(1536ull * 1024 * 1024);
   b << n2.get();
   b << m2.get();
   b << c1.get();
   b << c2.get();

   // Those should all be references.
   b << n0.get();
   b << n1.get();
   b << m1.get();
   b << n2.get();
   b << m2.get();
   b << c1.get();
   b << c2.get();

   // To make a copy instead of using the const references:
   auto bytecounts = b.GetByteCounts();
   // Rewind.  Other code uses Reset instead of SetBufferOffset
   b.SetReadMode();
   b.Reset();
   b.SetByteCounts(std::move(bytecounts));

   std::unique_ptr<TObject> rn0;
   std::unique_ptr<TObject> rn1;
   std::unique_ptr<TObject> rn2;
   std::unique_ptr<TObject> rm1;
   std::unique_ptr<TObject> rm2;
   std::unique_ptr<TObject> rc1;
   std::unique_ptr<TObject> rc2;

   Update(errors, rn0, ReadAndCheck(b, n0->IsA()));

   b.SetBufferOffset(512ull * 1024 * 1024);
   Update(errors, rn1, ReadAndCheck(b, n1->IsA()));
   Update(errors, rm1, ReadAndCheck(b, m1->IsA()));

   b.SetBufferOffset(1536ull * 1024 * 1024);
   Update(errors, rn2, ReadAndCheck(b, n2->IsA()));
   Update(errors, rm2, ReadAndCheck(b, m2->IsA()));
   Update(errors, rc1, ReadAndCheck(b, c1->IsA()));
   Update(errors, rc2, ReadAndCheck(b, c2->IsA()));

   // Reference and Class name below 1G
   errors += ReadAndCheck(b, n0->IsA(), rn0.get()).fError;
   errors += ReadAndCheck(b, n1->IsA(), rn1.get()).fError;
   errors += ReadAndCheck(b, m1->IsA(), rm1.get()).fError;
   if (1) { // These require implementing proper support for long range references.
      errors += ReadAndCheck(b, n2->IsA(), rn2.get()).fError;  // Reference over 1G
      errors += ReadAndCheck(b, m2->IsA(), rm2.get()).fError;  // Reference over 1G
      errors += ReadAndCheck(b, c1->IsA(), rc1.get()).fError;  // Class and reference over 1G
      errors += ReadAndCheck(b, c2->IsA(), rc2.get()).fError;  // Class and reference over 1G
   }

   EXPECT_EQ(errors, 0);
}
