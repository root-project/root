#include <Compression.h>
#include <RZip.h>

#include <gtest/gtest.h>

#include <memory>

static void testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::EValues compressionAlgorithm)
{
   static constexpr size_t BufferSize = 256;
   static constexpr size_t MaxBytes = 128;
   static_assert(MaxBytes <= BufferSize, "MaxBytes must be smaller than BufferSize");
   static constexpr size_t StartOffset = (BufferSize - MaxBytes) / 2;
   // For extra "safety", allocate the buffers on the heap to avoid corrupting the stack should anything go wrong.
   std::unique_ptr<char[]> source(new char[BufferSize]);
   std::unique_ptr<char[]> target(new char[BufferSize]);

   // Fill the buffers with monotonically increasing numbers. This is easy to compress, but that's fine because we scan
   // through all possible sizes.
   for (size_t i = 0; i < BufferSize; i++) {
      source[i] = static_cast<char>(i);
      target[i] = static_cast<char>(i);
   }

   // Now test all possible combinations of target and source sizes. The outer loop is for the target sizes because that
   // allows us to check that nothing got overwritten.
   for (size_t targetSize = 1; targetSize <= MaxBytes; targetSize++) {
      for (size_t sourceSize = 1; sourceSize <= MaxBytes; sourceSize++) {
         for (int cxlevel = 1; cxlevel <= 9; cxlevel++) {
            int srcsize = static_cast<int>(sourceSize);
            int tgtsize = static_cast<int>(targetSize);
            int irep = -1;
            R__zipMultipleAlgorithm(cxlevel, &srcsize, source.get(), &tgtsize, target.get() + StartOffset, &irep,
                                    compressionAlgorithm);

            for (size_t i = 0; i < StartOffset; i++) {
               EXPECT_EQ(target[i], static_cast<char>(i));
            }
            for (size_t i = StartOffset + targetSize + 1; i < BufferSize; i++) {
               EXPECT_EQ(target[i], static_cast<char>(i));
            }
         }
      }
   }
}

TEST(RZip, ZipBufferSizesOld)
{
   testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::kOldCompressionAlgo);
}

TEST(RZip, ZipBufferSizesZLIB)
{
   testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::kZLIB);
}

TEST(RZip, ZipBufferSizesLZMA)
{
   testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::kLZMA);
}

TEST(RZip, ZipBufferSizesLZ4)
{
   testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::kLZ4);
}

TEST(RZip, ZipBufferSizesZSTD)
{
   testZipBufferSizes(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
}
