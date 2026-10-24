#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TBufferFile.h"
#include "TMemFile.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TContainerConverters.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TNamed.h"
#include "TVirtualCollectionProxy.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <iostream>

// Tests ROOT-8367
TEST(TBufferFile, ROOT_8367)
{
   std::vector<int> v{1, 2, 3, 4, 5, 6, 7};

   TClass *cl1 = TClass::GetClass("vector<int>");
   TClass *cl2 = TClass::GetClass("vector<float>");

   TBufferFile *m_buff = new TBufferFile(TBuffer::kWrite);
   m_buff->WriteObjectAny(&v, cl1);
   m_buff->SetReadMode();
   m_buff->Reset();
   void *obj = m_buff->ReadObjectAny(cl2);
   std::vector<float> &v2 = *reinterpret_cast<std::vector<float> *>(obj);
   EXPECT_FLOAT_EQ(v2[0], 1.);
   EXPECT_FLOAT_EQ(v2[1], 2.);
   EXPECT_FLOAT_EQ(v2[2], 3.);
   EXPECT_FLOAT_EQ(v2[5], 6.);
   EXPECT_FLOAT_EQ(v2[6], 7.);
   EXPECT_EQ(v2.size(), 7);
}

// https://github.com/root-project/root/issues/19371
#define MYSTRUCT0 struct MyS0 { static constexpr short Class_Version() { return 0; } };
#define MYSTRUCT1 struct MyS1 { static constexpr short Class_Version() { return 1; } };
MYSTRUCT0
MYSTRUCT1
#define TO_LITERAL(string) _QUOTE_(string)
TEST(TBufferFile, ForeignZeroVersionClass)
{
   gInterpreter->Declare(TO_LITERAL(MYSTRUCT0));
   gInterpreter->Declare(TO_LITERAL(MYSTRUCT1));
   MyS0 s0;
   MyS1 s1;
   TMemFile f("mem19371.root", "RECREATE");
   f.WriteObject(&s0, "s0");
   f.WriteObject(&s1, "s1");
   EXPECT_NE(f.Get<MyS0>("s0"), nullptr); // Before the fix, even if return was already not nullptr, this line was raising an Error thus test would fail with unexpected diagnostic
   EXPECT_NE(f.Get<MyS1>("s1"), nullptr);
   EXPECT_EQ(f.GetKey("s0")->GetObjlen(), 10);
}

// A collection size read from corrupt data must not be used to allocate memory.
#define CORRUPTINNER        \
   struct CorruptInner {    \
      int fA = 7;           \
      char fPad[1024] = {}; \
   };
#define CORRUPTOUTER                   \
   struct CorruptOuter {               \
      std::vector<CorruptInner> fObjs; \
      std::vector<int> fInts;          \
   };
// Collections inside the elements of a member-wise collection are read by TStreamerInfo::ReadBuffer.
#define CORRUPTNESTED                    \
   struct CorruptNested {                \
      std::vector<CorruptOuter> fOuters; \
   };
CORRUPTINNER
CORRUPTOUTER
CORRUPTNESTED

namespace {

// The element count of RawEvent::m_banks found in a corrupt LHCb file; negative as an Int_t.
constexpr unsigned int kCorruptSize = 0xa2c35db6;

/// Position of the big-endian Int_t equal to `expected` in `data`, which must occur exactly once.
/// Returns data.size() if it does not.
std::size_t FindInt(const std::vector<char> &data, unsigned int expected)
{
   const char pattern[4] = {char(expected >> 24), char(expected >> 16), char(expected >> 8), char(expected)};
   const auto it = std::search(data.begin(), data.end(), pattern, pattern + 4);
   if (it == data.end() || std::search(it + 1, data.end(), pattern, pattern + 4) != data.end()) {
      ADD_FAILURE() << "expected exactly one occurrence of " << expected << " in the serialized data";
      return data.size();
   }
   return it - data.begin();
}

/// Overwrite the big-endian Int_t equal to `expected` in `data` with `value`.
void CorruptInt(std::vector<char> &data, unsigned int expected, unsigned int value)
{
   const auto pos = FindInt(data, expected);
   if (pos == data.size())
      return;
   for (int i = 0; i < 4; ++i)
      data[pos + i] = char(value >> (8 * (3 - i)));
}

std::vector<char> Write(const void *obj, TClass *cl)
{
   TBufferFile buf(TBuffer::kWrite);
   buf.WriteObjectAny(obj, cl);
   return std::vector<char>(buf.Buffer(), buf.Buffer() + buf.Length());
}

void *Read(std::vector<char> &data, TClass *cl)
{
   TBufferFile buf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
   return buf.ReadObjectAny(cl);
}

template <typename T>
std::vector<char> Write(const T &obj)
{
   return Write(&obj, TClass::GetClass<T>());
}

template <typename T>
std::unique_ptr<T> Read(std::vector<char> &data)
{
   return std::unique_ptr<T>(static_cast<T *>(Read(data, TClass::GetClass<T>())));
}

/// Number of elements of the collection of type `cl` at `obj`.
UInt_t CollectionSize(TClass *cl, void *obj)
{
   TVirtualCollectionProxy::TPushPop helper(cl->GetCollectionProxy(), obj);
   return cl->GetCollectionProxy()->Size();
}

class CorruptCollectionSize : public ::testing::Test {
protected:
   static constexpr int kNObjs = 0x123;
   static constexpr int kNInts = 0x321;

   std::vector<char> fData;

   static void SetUpTestSuite()
   {
      gInterpreter->Declare(TO_LITERAL(CORRUPTINNER));
      gInterpreter->Declare(TO_LITERAL(CORRUPTOUTER));
      gInterpreter->Declare(TO_LITERAL(CORRUPTNESTED));
   }

   void SetUp() override
   {
      CorruptOuter outer;
      outer.fObjs.resize(kNObjs);
      outer.fInts.assign(kNInts, 1);
      fData = Write(outer);
   }
};

} // namespace

TEST_F(CorruptCollectionSize, MemberWise)
{
   CorruptInt(fData, kNObjs, kCorruptSize);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfoActions::ReadSTLMemberWiseSameClass", "negative number of elements",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto outer = Read<CorruptOuter>(fData);
   ASSERT_NE(outer, nullptr);
   EXPECT_TRUE(outer->fObjs.empty());
   EXPECT_EQ(outer->fInts.size(), kNInts);
}

TEST_F(CorruptCollectionSize, BasicType)
{
   CorruptInt(fData, kNInts, kCorruptSize);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfoActions::ReadCollectionBasicType", "negative number of elements",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto outer = Read<CorruptOuter>(fData);
   ASSERT_NE(outer, nullptr);
   EXPECT_EQ(outer->fObjs.size(), kNObjs);
   EXPECT_TRUE(outer->fInts.empty());
}

// Collections without a dictionary are read through TEmulatedCollectionProxy and TEmulatedMapProxy.
TEST_F(CorruptCollectionSize, Emulated)
{
   for (auto [name, location] : {std::make_pair("vector<int*>", "TEmulatedCollectionProxy::ReadBuffer"),
                                 std::make_pair("map<int,float>", "TEmulatedMapProxy::ReadBuffer")}) {
      SCOPED_TRACE(name);
      TClass *cl = TClass::GetClass(name);
      ASSERT_NE(cl, nullptr);
      ASSERT_TRUE(cl->GetCollectionProxy()->GetProperties() & TVirtualCollectionProxy::kIsEmulated);
      void *obj = cl->New();
      {
         // Emulated maps are stored like vectors of pairs, so the duplicate keys do not matter.
         TVirtualCollectionProxy::TPushPop helper(cl->GetCollectionProxy(), obj);
         cl->GetCollectionProxy()->Commit(cl->GetCollectionProxy()->Allocate(kNObjs, true));
      }
      ASSERT_EQ(CollectionSize(cl, obj), kNObjs);
      auto data = Write(obj, cl);
      cl->Destructor(obj);
      CorruptInt(data, kNObjs, kCorruptSize);

      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, location, "negative number of elements", /*matchFullMessage=*/false);
      diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
      void *read = Read(data, cl);
      ASSERT_NE(read, nullptr);
      EXPECT_EQ(CollectionSize(cl, read), 0u);
      cl->Destructor(read);
   }
}

TEST_F(CorruptCollectionSize, Nested)
{
   CorruptNested nested;
   nested.fOuters.resize(1);
   nested.fOuters[0].fObjs.resize(kNObjs);
   auto data = Write(nested);
   CorruptInt(data, kNObjs, kCorruptSize);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfo::ReadBuffer", "negative number of elements", /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<CorruptNested>(data);
   ASSERT_NE(read, nullptr);
   ASSERT_EQ(read->fOuters.size(), 1u);
   EXPECT_TRUE(read->fOuters[0].fObjs.empty());
}

// A TClonesArray on file is read into an STL collection by TConvertClonesArrayToProxy, which takes the
// absolute value of negative sizes.
TEST_F(CorruptCollectionSize, ClonesArrayToProxy)
{
   TClonesArray clones("TNamed", kNObjs);
   for (int i = 0; i < kNObjs; ++i)
      clones.ConstructedAt(i);
   TBufferFile wbuf(TBuffer::kWrite);
   clones.Streamer(wbuf);
   std::vector<char> data(wbuf.Buffer(), wbuf.Buffer() + wbuf.Length());
   CorruptInt(data, kNObjs, std::numeric_limits<Int_t>::min());

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TConvertClonesArrayToProxy::operator()", "negative number of elements",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   TClass *cl = TClass::GetClass("vector<TNamed>");
   ASSERT_NE(cl, nullptr);
   TConvertClonesArrayToProxy convert(cl->GetCollectionProxy(), /*isPointer=*/false, /*isPrealloc=*/false);
   void *read = cl->New();
   TBufferFile rbuf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
   convert(rbuf, read);
   EXPECT_EQ(CollectionSize(cl, read), 0u);
   cl->Destructor(read);
}
