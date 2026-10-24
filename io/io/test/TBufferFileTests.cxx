#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TBufferFile.h"
#include "TMemFile.h"
#include "TClass.h"
#include "TClassStreamer.h"
#include "TClonesArray.h"
#include "TContainerConverters.h"
#include "TInterpreter.h"
#include "TKey.h"
#include "TMemberStreamer.h"
#include "TNamed.h"
#include "TVirtualCollectionProxy.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
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

/// Bytes left in `data` after the big-endian Int_t equal to `expected`.
unsigned int BytesAfterInt(const std::vector<char> &data, unsigned int expected)
{
   return data.size() - std::min(data.size(), FindInt(data, expected) + 4);
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

/// Resize the collection of type `cl` at `obj` to `n` elements.
void ResizeCollection(TClass *cl, void *obj, UInt_t n)
{
   TVirtualCollectionProxy::TPushPop helper(cl->GetCollectionProxy(), obj);
   cl->GetCollectionProxy()->Commit(cl->GetCollectionProxy()->Allocate(n, true));
}

/// Write a CorruptEvoA with `nFloats` floats and twice `nElems` elements that occupy no space to a file, and rename
/// its classes to CorruptEvoB and CorruptEvoBElem, so that reading it back goes through schema evolution.
std::vector<char> WriteEvolvedFile(int nFloats, int nElems)
{
   TClass *cl = TClass::GetClass("CorruptEvoA");
   void *obj = cl->New();
   auto *floats = reinterpret_cast<std::vector<float> *>(static_cast<char *>(obj) + cl->GetDataMemberOffset("fFloats"));
   floats->assign(nFloats, 1.5f);
   ResizeCollection(TClass::GetClass("vector<CorruptEvoAElem>"),
                    static_cast<char *>(obj) + cl->GetDataMemberOffset("fElems"), nElems);
   ResizeCollection(TClass::GetClass("vector<CorruptOldElem>"),
                    static_cast<char *>(obj) + cl->GetDataMemberOffset("fRenamed"), nElems);
   std::vector<char> data;
   {
      TMemFile file("evolution.root", "RECREATE", "", /*compress=*/0);
      file.WriteObjectAny(obj, cl, "obj");
      file.Write();
      data.resize(file.GetSize());
      file.CopyTo(data.data(), data.size());
   }
   cl->Destructor(obj);
   const std::string from = "CorruptEvoA", to = "CorruptEvoB";
   for (auto it = data.begin(); (it = std::search(it, data.end(), from.begin(), from.end())) != data.end();)
      it = std::copy(to.begin(), to.end(), it);
   return data;
}

void *ReadEvolvedFile(std::vector<char> &data)
{
   TMemFile file("evolution.root", TMemFile::ZeroCopyView_t(data.data(), data.size()));
   return file.GetObjectChecked("obj", "CorruptEvoB");
}

class CorruptCollectionSize : public ::testing::Test {
protected:
   static constexpr int kNObjs = 0x123;
   static constexpr int kNInts = 0x321;
   static constexpr int kNFloats = 0x2345;

   std::vector<char> fData;

   static void SetUpTestSuite()
   {
      // Classes can only be declared and their streamers adopted once, e.g. with --gtest_repeat.
      static bool done = false;
      if (done)
         return;
      done = true;
      gInterpreter->Declare(TO_LITERAL(CORRUPTINNER));
      gInterpreter->Declare(TO_LITERAL(CORRUPTOUTER));
      gInterpreter->Declare(TO_LITERAL(CORRUPTNESTED));
      // Elements of a vector<CorruptInPlace> occupy no space on file.
      gInterpreter->Declare(R"CODE(
         struct CorruptNoBytes {};
         struct CorruptInPlace {
            CorruptNoBytes *fObj = new CorruptNoBytes; //->
         };
         struct CorruptZeroByteHolder {
            std::vector<CorruptInPlace> fInPlace;
            std::vector<int *> fPtrs;
         };
         // Version 1 is written as CorruptEvoA, and read back as CorruptEvoB after renaming it in the file.
         struct CorruptEvoAElem {
            CorruptNoBytes *fObj = new CorruptNoBytes; //->
            virtual ~CorruptEvoAElem() = default;
            ClassDef(CorruptEvoAElem, 1);
         };
         // The only data member of CorruptCustomMember is streamed by a custom member streamer that writes nothing.
         struct CorruptCustomMember : TObject {
            TObject *fObj = nullptr;
            ClassDefOverride(CorruptCustomMember, 1);
         };
         // CorruptOldElem is read as CorruptNewElem through a rename rule.
         struct CorruptOldElem {
            CorruptNoBytes *fObj = new CorruptNoBytes; //->
            virtual ~CorruptOldElem() = default;
            ClassDef(CorruptOldElem, 1);
         };
         struct CorruptNewElem {
            CorruptNoBytes *fObj = new CorruptNoBytes; //->
            int fNew = 3;
            virtual ~CorruptNewElem() = default;
            ClassDef(CorruptNewElem, 1);
         };
         struct CorruptEvoA {
            std::vector<float> fFloats;
            std::vector<CorruptEvoAElem> fElems;
            std::vector<CorruptOldElem> fRenamed;
            virtual ~CorruptEvoA() = default;
            ClassDef(CorruptEvoA, 1);
         };
         struct CorruptEvoBElem {
            CorruptNoBytes *fObj = new CorruptNoBytes; //->
            int fNew = 3;
            virtual ~CorruptEvoBElem() = default;
            ClassDef(CorruptEvoBElem, 2);
         };
         struct CorruptEvoB {
            std::vector<double> fFloats;
            std::vector<CorruptEvoBElem> fElems;
            std::vector<CorruptNewElem> fRenamed;
            virtual ~CorruptEvoB() = default;
            ClassDef(CorruptEvoB, 2);
         };
      )CODE");
      TClass::GetClass("CorruptNoBytes")->AdoptStreamer(new TClassStreamer([](TBuffer &, void *) {}));
      TClass *customMember = TClass::GetClass("CorruptCustomMember");
      customMember->BuildRealData();
      customMember->AdoptMemberStreamer("*fObj", new TMemberStreamer([](TBuffer &, void *, Int_t) {}));
      customMember->IgnoreTObjectStreamer();
      TClass::AddRule("CorruptNewElem version=[1-] sourceClass=CorruptOldElem targetClass=CorruptNewElem");
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

// A size larger than the bytes left in the buffer cannot be valid if each element occupies at least one byte.
TEST_F(CorruptCollectionSize, MemberWiseOversized)
{
   CorruptInt(fData, kNObjs, BytesAfterInt(fData, kNObjs) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfoActions::ReadSTLMemberWiseSameClass", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto outer = Read<CorruptOuter>(fData);
   ASSERT_NE(outer, nullptr);
   EXPECT_TRUE(outer->fObjs.empty());
   EXPECT_EQ(outer->fInts.size(), kNInts);
}

TEST_F(CorruptCollectionSize, NestedOversized)
{
   CorruptNested nested;
   nested.fOuters.resize(1);
   nested.fOuters[0].fObjs.resize(kNObjs);
   auto data = Write(nested);
   CorruptInt(data, kNObjs, BytesAfterInt(data, kNObjs) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfo::ReadBuffer", "bytes are left in the buffer", /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<CorruptNested>(data);
   ASSERT_NE(read, nullptr);
   ASSERT_EQ(read->fOuters.size(), 1u);
   EXPECT_TRUE(read->fOuters[0].fObjs.empty());
}

TEST_F(CorruptCollectionSize, ClonesArrayToProxyOversized)
{
   TClass *cl = TClass::GetClass("vector<TNamed>");
   ASSERT_NE(cl, nullptr);
   for (bool bypass : {true, false}) {
      SCOPED_TRACE(bypass ? "member-wise" : "object-wise");
      TClonesArray clones("TNamed", kNObjs);
      clones.BypassStreamer(bypass);
      for (int i = 0; i < kNObjs; ++i)
         clones.ConstructedAt(i);
      TBufferFile wbuf(TBuffer::kWrite);
      clones.Streamer(wbuf);
      std::vector<char> data(wbuf.Buffer(), wbuf.Buffer() + wbuf.Length());
      CorruptInt(data, kNObjs, BytesAfterInt(data, kNObjs) + 1);

      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TConvertClonesArrayToProxy::operator()", "bytes are left in the buffer",
                         /*matchFullMessage=*/false);
      diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
      TConvertClonesArrayToProxy convert(cl->GetCollectionProxy(), /*isPointer=*/false, /*isPrealloc=*/false);
      void *read = cl->New();
      TBufferFile rbuf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
      convert(rbuf, read);
      EXPECT_EQ(CollectionSize(cl, read), 0u);
      cl->Destructor(read);
   }
}

TEST_F(CorruptCollectionSize, BasicTypeOversized)
{
   CorruptInt(fData, kNInts, BytesAfterInt(fData, kNInts) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfoActions::ReadCollectionBasicType", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto outer = Read<CorruptOuter>(fData);
   ASSERT_NE(outer, nullptr);
   EXPECT_EQ(outer->fObjs.size(), kNObjs);
   EXPECT_TRUE(outer->fInts.empty());
}

// A size equal to the bytes left in the buffer is not rejected.
TEST_F(CorruptCollectionSize, BasicTypeBoundary)
{
   const auto size = BytesAfterInt(fData, kNInts);
   CorruptInt(fData, kNInts, size);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto outer = Read<CorruptOuter>(fData);
   ASSERT_NE(outer, nullptr);
   EXPECT_EQ(outer->fInts.size(), size);
}

TEST_F(CorruptCollectionSize, ObjectWiseOversized)
{
   std::vector<std::string> strings(kNObjs, "a");
   auto data = Write(strings);
   CorruptInt(data, kNObjs, BytesAfterInt(data, kNObjs) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TGenCollectionStreamer::ReadBufferGeneric", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<std::vector<std::string>>(data);
   ASSERT_NE(read, nullptr);
   EXPECT_TRUE(read->empty());
}

TEST_F(CorruptCollectionSize, EmulatedOversized)
{
   TClass *cl = TClass::GetClass("map<int,float>");
   ASSERT_NE(cl, nullptr);
   void *obj = cl->New();
   {
      TVirtualCollectionProxy::TPushPop helper(cl->GetCollectionProxy(), obj);
      cl->GetCollectionProxy()->Commit(cl->GetCollectionProxy()->Allocate(kNObjs, true));
   }
   auto data = Write(obj, cl);
   cl->Destructor(obj);
   CorruptInt(data, kNObjs, BytesAfterInt(data, kNObjs) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TEmulatedMapProxy::ReadBuffer", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   void *read = Read(data, cl);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(CollectionSize(cl, read), 0u);
   cl->Destructor(read);
}

// Valid collections whose elements occupy no space on file must still be read.
TEST_F(CorruptCollectionSize, ZeroByteElements)
{
   constexpr int kN = 1000;
   TClass *cl = TClass::GetClass("CorruptZeroByteHolder");
   ASSERT_NE(cl, nullptr);
   TClass *inPlace = TClass::GetClass("vector<CorruptInPlace>");
   TClass *ptrs = TClass::GetClass("vector<int*>");
   void *obj = cl->New();
   ResizeCollection(inPlace, static_cast<char *>(obj) + cl->GetDataMemberOffset("fInPlace"), kN);
   ResizeCollection(ptrs, static_cast<char *>(obj) + cl->GetDataMemberOffset("fPtrs"), kN);
   auto data = Write(obj, cl);
   cl->Destructor(obj);
   ASSERT_LT(data.size(), kN);
   void *read = Read(data, cl);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(CollectionSize(inPlace, static_cast<char *>(read) + cl->GetDataMemberOffset("fInPlace")), kN);
   EXPECT_EQ(CollectionSize(ptrs, static_cast<char *>(read) + cl->GetDataMemberOffset("fPtrs")), kN);
   cl->Destructor(read);
}

// The same holds for a TClonesArray read into a vector, here with objects whose only member has a custom streamer.
TEST_F(CorruptCollectionSize, ZeroByteClonesArrayToProxy)
{
   constexpr int kN = 1000;
   TClonesArray clones("CorruptCustomMember", kN);
   clones.ExpandCreate(kN);
   TBufferFile wbuf(TBuffer::kWrite);
   clones.Streamer(wbuf);
   std::vector<char> data(wbuf.Buffer(), wbuf.Buffer() + wbuf.Length());
   ASSERT_LT(data.size(), kN);

   TClass *cl = TClass::GetClass("vector<CorruptCustomMember>");
   ASSERT_NE(cl, nullptr);
   TConvertClonesArrayToProxy convert(cl->GetCollectionProxy(), /*isPointer=*/false, /*isPrealloc=*/false);
   void *read = cl->New();
   TBufferFile rbuf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
   convert(rbuf, read);
   EXPECT_EQ(CollectionSize(cl, read), kN);
   cl->Destructor(read);
}

// The same holds when reading through schema evolution, which uses the layout on file: here from vector<float>
// into vector<double>, and elements that occupy no space on file but gained a data member, of the same class or of a
// class renamed by a rule.
TEST_F(CorruptCollectionSize, SchemaEvolution)
{
   constexpr int kNElems = 1000;
   auto data = WriteEvolvedFile(kNFloats, kNElems);
   TClass *cl = TClass::GetClass("CorruptEvoB");
   auto *read = static_cast<char *>(ReadEvolvedFile(data));
   ASSERT_NE(read, nullptr);
   auto *doubles = reinterpret_cast<std::vector<double> *>(read + cl->GetDataMemberOffset("fFloats"));
   ASSERT_EQ(doubles->size(), kNFloats);
   EXPECT_EQ(doubles->front(), 1.5);
   EXPECT_EQ(CollectionSize(TClass::GetClass("vector<CorruptEvoBElem>"), read + cl->GetDataMemberOffset("fElems")),
             kNElems);
   EXPECT_EQ(CollectionSize(TClass::GetClass("vector<CorruptNewElem>"), read + cl->GetDataMemberOffset("fRenamed")),
             kNElems);
   cl->Destructor(read);
}

TEST_F(CorruptCollectionSize, SchemaEvolutionOversized)
{
   auto data = WriteEvolvedFile(kNFloats, 1);
   // The rest of the file may contain the same 4 bytes, so look for the size followed by the first value, 1.5f.
   const char pattern[8] = {
      char(kNFloats >> 24), char(kNFloats >> 16), char(kNFloats >> 8), char(kNFloats), 0x3f, char(0xc0), 0, 0};
   const auto pos = std::search(data.begin(), data.end(), pattern, pattern + 8);
   ASSERT_NE(pos, data.end());
   // One more than the bytes left after the size.
   const unsigned int size = data.end() - (pos + 4) + 1;
   for (int i = 0; i < 4; ++i)
      pos[i] = char(size >> (8 * (3 - i)));

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStreamerInfoActions::ConvertCollectionBasicType", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   TClass *cl = TClass::GetClass("CorruptEvoB");
   auto *read = static_cast<char *>(ReadEvolvedFile(data));
   ASSERT_NE(read, nullptr);
   EXPECT_TRUE(reinterpret_cast<std::vector<double> *>(read + cl->GetDataMemberOffset("fFloats"))->empty());
   cl->Destructor(read);
}
