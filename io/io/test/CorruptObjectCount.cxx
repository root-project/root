#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TArrayC.h"
#include "TArrayD.h"
#include "TArrayF.h"
#include "TArrayI.h"
#include "TArrayL.h"
#include "TArrayL64.h"
#include "TArrayS.h"
#include "TBufferFile.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TInterpreter.h"
#include "TNamed.h"
#include "TObjArray.h"
#include "TRefArray.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// A number of objects read from corrupt data must not be used to allocate memory.

namespace {

constexpr int kN = 0x123;

/// Position of the big-endian Int_t equal to `expected` in `data`, or data.end(). Unless `firstOccurrence`, it must
/// occur exactly once.
std::vector<char>::iterator FindInt(std::vector<char> &data, unsigned int expected, bool firstOccurrence)
{
   const char pattern[4] = {char(expected >> 24), char(expected >> 16), char(expected >> 8), char(expected)};
   const auto it = std::search(data.begin(), data.end(), pattern, pattern + 4);
   if (it == data.end() || (!firstOccurrence && std::search(it + 1, data.end(), pattern, pattern + 4) != data.end())) {
      ADD_FAILURE() << "expected exactly one occurrence of " << expected << " in the serialized data";
      return data.end();
   }
   return it;
}

/// Overwrite the big-endian Int_t at `it` with `value`.
void SetInt(std::vector<char>::iterator it, unsigned int value)
{
   for (int i = 0; i < 4; ++i)
      *(it + i) = char(value >> (8 * (3 - i)));
}

template <typename T>
std::vector<char> Write(T &obj)
{
   TBufferFile buf(TBuffer::kWrite);
   buf.WriteObject(&obj);
   return std::vector<char>(buf.Buffer(), buf.Buffer() + buf.Length());
}

template <typename T>
std::unique_ptr<T> Read(std::vector<char> &data)
{
   TBufferFile buf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
   return std::unique_ptr<T>(static_cast<T *>(buf.ReadObject(T::Class())));
}

template <typename T>
void CheckOversized(T &array, bool firstOccurrence = false)
{
   auto data = Write(array);
   const auto it = FindInt(data, kN, firstOccurrence);
   ASSERT_NE(it, data.end());
   // The smallest number of objects that cannot fit in the rest of the buffer. Without the check, reading them runs
   // past the end of the buffer rather than allocating gigabytes.
   SetInt(it, data.end() - (it + 4) + 1);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, std::string(T::Class_Name()) + "::Streamer", "bytes are left in the buffer",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<T>(data);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(read->GetEntriesFast(), 0);
}

template <typename T>
void CheckNegative(T &array, bool firstOccurrence = false)
{
   auto data = Write(array);
   const auto it = FindInt(data, kN, firstOccurrence);
   ASSERT_NE(it, data.end());
   SetInt(it, std::numeric_limits<Int_t>::min());

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, std::string(T::Class_Name()) + "::Streamer", "negative number of elements",
                      /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<T>(data);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(read->GetEntriesFast(), 0);
}

template <typename T>
void CheckArray(const char *location)
{
   for (bool negative : {true, false}) {
      SCOPED_TRACE(negative ? "negative" : "oversized");
      T array(kN);
      TBufferFile wbuf(TBuffer::kWrite);
      array.Streamer(wbuf);
      std::vector<char> data(wbuf.Buffer(), wbuf.Buffer() + wbuf.Length());
      const auto it = FindInt(data, kN, /*firstOccurrence=*/false);
      ASSERT_NE(it, data.end());
      SetInt(it, negative ? std::numeric_limits<Int_t>::min() : data.end() - (it + 4) + 1);

      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, location, negative ? "negative number of elements" : "bytes are left in the buffer",
                         /*matchFullMessage=*/false);
      T read;
      TBufferFile rbuf(TBuffer::kRead, data.size(), data.data(), /*adopt=*/false);
      read.Streamer(rbuf);
      EXPECT_EQ(read.GetSize(), 0);
   }
}

} // namespace

TEST(CorruptObjectCount, TObjArray)
{
   TObjArray array;
   array.SetOwner();
   for (int i = 0; i < kN; ++i)
      array.Add(new TNamed("a", ""));
   CheckOversized(array);
   CheckNegative(array);
}

TEST(CorruptObjectCount, TRefArray)
{
   std::vector<TNamed> objects(kN);
   TRefArray array;
   for (auto &obj : objects)
      array.Add(&obj);
   // The unique IDs of the objects come after their number and may include it.
   CheckOversized(array, /*firstOccurrence=*/true);
   CheckNegative(array, /*firstOccurrence=*/true);
}

TEST(CorruptObjectCount, TClonesArray)
{
   for (bool bypass : {true, false}) {
      SCOPED_TRACE(bypass ? "bypass streamer" : "object-wise");
      TClonesArray array("TNamed");
      array.BypassStreamer(bypass);
      for (int i = 0; i < kN; ++i)
         new (array[i]) TNamed("a", "");
      CheckOversized(array);
   }
}

// TClonesArray takes the absolute value of negative numbers of objects, which is undefined for INT_MIN.
TEST(CorruptObjectCount, TClonesArrayMinInt)
{
   TClonesArray array("TNamed");
   for (int i = 0; i < kN; ++i)
      new (array[i]) TNamed("a", "");
   auto data = Write(array);
   const auto it = FindInt(data, kN, /*firstOccurrence=*/false);
   ASSERT_NE(it, data.end());
   SetInt(it, std::numeric_limits<Int_t>::min());

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TClonesArray::Streamer", "negative number of elements", /*matchFullMessage=*/false);
   diags.optionalDiag(kError, "TBufferFile::CheckByteCount", "read too few bytes", /*matchFullMessage=*/false);
   auto read = Read<TClonesArray>(data);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(read->GetEntriesFast(), 0);
}

// Valid TClonesArrays whose objects occupy no space on file must still be read.
TEST(CorruptObjectCount, TClonesArrayZeroByteObjects)
{
   // The class can only be declared once, e.g. with --gtest_repeat.
   static const bool declared = gInterpreter->Declare(R"CODE(
      struct CorruptEmptyObject : TObject {
         ClassDefOverride(CorruptEmptyObject, 1);
      };
   )CODE");
   ASSERT_TRUE(declared);
   TClass *cl = TClass::GetClass("CorruptEmptyObject");
   ASSERT_NE(cl, nullptr);
   cl->IgnoreTObjectStreamer();

   constexpr int kMany = 1000;
   TClonesArray array("CorruptEmptyObject");
   array.ExpandCreate(kMany);
   auto data = Write(array);
   ASSERT_LT(data.size(), kMany);

   auto read = Read<TClonesArray>(data);
   ASSERT_NE(read, nullptr);
   EXPECT_EQ(read->GetEntriesFast(), kMany);
}

TEST(CorruptObjectCount, TArray)
{
   CheckArray<TArrayC>("TArrayC::Streamer");
   CheckArray<TArrayS>("TArrayS::Streamer");
   CheckArray<TArrayI>("TArrayI::Streamer");
   CheckArray<TArrayL>("TArrayL::Streamer");
   CheckArray<TArrayL64>("TArrayL64::Streamer");
   CheckArray<TArrayF>("TArrayF::Streamer");
   CheckArray<TArrayD>("TArrayD::Streamer");
}
