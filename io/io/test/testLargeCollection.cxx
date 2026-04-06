/* Test large collections

   Need to test:
   - direct store in a TBufferFile
   - store as part of an object
   - those 2 things both for numerical type and structs.
   - at least one nested test.
   - (maybe not) as part of a split TTree.
*/

#include "TBufferFile.h"
#include "TClass.h"

#include <iostream>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------
// Helper: a non-trivial struct so we exercise the object-array path
// -----------------------------------------------------------------------
struct DataPoint {
   float x{0}, y{0}, z{0};
   bool  operator==(const DataPoint &o) const { return x == o.x && y == o.y && z == o.z; }
};

// -----------------------------------------------------------------------
// Helper: an object that owns a large numerical vector and a large struct
// vector, with a minimal hand-written Streamer so we control the layout.
// -----------------------------------------------------------------------
struct LargeCollectionFixture {
   std::vector<float>     fFloats;
   std::vector<DataPoint> fPoints;

#if 0
   void Streamer(TBuffer &b)
   {
      if (b.IsReading()) {
         Long64_t nf = 0, np = 0;
         b >> nf;
         fFloats.resize(nf);
         b.ReadFastArray(fFloats.data(), nf);
         b >> np;
         fPoints.resize(np);
         b.ReadFastArray(fPoints.data(), TClass::GetClass(typeid(Point)), np);
      } else {
         Long64_t nf = fFloats.size(), np = fPoints.size();
         b << nf;
         b.WriteFastArray(fFloats.data(), nf);
         b << np;
         b.WriteFastArray(fPoints.data(), TClass::GetClass(typeid(Point)), np);
      }
   }
#endif
};

#ifdef __ROOTCLING__
#pragma link C++ class DataPoint + ;
#pragma link C++ class LargeCollectionFixture + ;
#endif

// -----------------------------------------------------------------------
// A do-nothing reallocator: we pre-allocate the buffer ourselves.
// -----------------------------------------------------------------------
static char *DoNothingAllocator(char *input, size_t, size_t)
{
   return input;
}

// -----------------------------------------------------------------------
// 1. Direct store of a numerical array in a TBufferFile
// -----------------------------------------------------------------------
int testDirectNumerical()
{
   int errors = 0;

   // Use a pre-allocated 6 GB region so TBuffer never has to realloc.
   std::vector<char> raw;
   raw.reserve(6 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 6 * 1024 * 1024 * 1024ll - 100, raw.data(), false /* don't adopt */,
                 DoNothingAllocator);

   // --- small array (stays in the <2 GB region) ---
   const Long64_t     smallN = 1000;
   std::vector<float> orig_small(smallN);
   for (Long64_t i = 0; i < smallN; ++i)
      orig_small[i] = float(i) * 0.1f;

   auto startSmall = b.GetCurrent() - b.Buffer();
   b << smallN;
   b.WriteFastArray(orig_small.data(), smallN);

   // --- large array (crosses the 2 GB boundary) ---
   // 512 M floats = 2 GB of payload
   const Long64_t     largeN = 512 * 1024 * 1024ll;
   std::vector<float> orig_large(largeN);
   for (Long64_t i = 0; i < largeN; ++i)
      orig_large[i] = float(i % 65536);

   auto startLarge = b.GetCurrent() - b.Buffer();
   b << largeN;
   b.WriteFastArray(orig_large.data(), largeN);

   // --- read back small ---
   b.SetReadMode();
   b.SetBufferOffset(startSmall);
   {
      Long64_t n = 0;
      b >> n;
      if (n != smallN) {
         std::cerr << "testDirectNumerical: small array count mismatch: got " << n << '\n';
         ++errors;
      } else {
         std::vector<float> got(n);
         b.ReadFastArray(got.data(), n);
         if (got != orig_small) {
            std::cerr << "testDirectNumerical: small array content mismatch\n";
            ++errors;
         }
      }
   }

   // --- read back large ---
   b.SetBufferOffset(startLarge);
   {
      Long64_t n = 0;
      b >> n;
      if (n != largeN) {
         std::cerr << "testDirectNumerical: large array count mismatch: got " << n << '\n';
         ++errors;
      } else {
         std::vector<float> got(n);
         b.ReadFastArray(got.data(), n);
         if (got != orig_large) {
            std::cerr << "testDirectNumerical: large array content mismatch\n";
            int nprinted = 0;
            for (Long64_t i = 0; i < n && nprinted < 10; ++i) {
               if (got[i] != orig_large[i]) {
                  std::cerr << "  [" << i << "] expected " << orig_large[i] << " got " << got[i] << '\n';
                  ++nprinted;
               }
            }
            ++errors;
         }
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// 2. Direct store of a struct array in a TBufferFile
// -----------------------------------------------------------------------
int testDirectStruct()
{
   int errors = 0;

   std::vector<char> raw;
   raw.reserve(6 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 6 * 1024 * 1024 * 1024ll - 100, raw.data(), false /* don't adopt */,
                 DoNothingAllocator);

   auto *pointClass = TClass::GetClass(typeid(DataPoint));

   // --- small struct array ---
   const Long64_t         smallN = 500;
   std::vector<DataPoint> orig_small(smallN);
   for (Long64_t i = 0; i < smallN; ++i)
      orig_small[i] = {float(i), float(i * 2), float(i * 3)};

   auto startSmall = b.GetCurrent() - b.Buffer();
   b << smallN;
   b.WriteFastArray(orig_small.data(), pointClass, smallN);

   // --- large struct array (crosses 2 GB) ---
   // ~170 M Points * 12 bytes = ~2 GB
   const Long64_t         largeN = 170 * 1024 * 1024ll;
   std::vector<DataPoint> orig_large(largeN);
   for (Long64_t i = 0; i < largeN; ++i)
      orig_large[i] = {float(i % 1000), float((i + 1) % 1000), float((i + 2) % 1000)};

   auto startLarge = b.GetCurrent() - b.Buffer();
   b << largeN;
   b.WriteFastArray(orig_large.data(), pointClass, largeN);

   // --- read back small ---
   b.SetReadMode();
   b.SetBufferOffset(startSmall);
   {
      Long64_t n = 0;
      b >> n;
      if (n != smallN) {
         std::cerr << "testDirectStruct: small array count mismatch: got " << n << '\n';
         ++errors;
      } else {
         std::vector<DataPoint> got(n);
         b.ReadFastArray(got.data(), pointClass, n);
         if (got != orig_small) {
            std::cerr << "testDirectStruct: small array content mismatch\n";
            ++errors;
         }
      }
   }

   // --- read back large ---
   b.SetBufferOffset(startLarge);
   {
      Long64_t n = 0;
      b >> n;
      if (n != largeN) {
         std::cerr << "testDirectStruct: large array count mismatch: got " << n << '\n';
         ++errors;
      } else {
         std::vector<DataPoint> got(n);
         b.ReadFastArray(got.data(), pointClass, n);
         if (got != orig_large) {
            std::cerr << "testDirectStruct: large array content mismatch\n";
            ++errors;
         }
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// 2b. Direct store of std::vector<T> via StreamObject
// -----------------------------------------------------------------------
int testDirectVector()
{
   int errors = 0;

   std::vector<char> raw;
   raw.reserve(6 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 6 * 1024 * 1024 * 1024ll - 100, raw.data(), false /* don't adopt */,
                 DoNothingAllocator);

   auto *floatVecClass = TClass::GetClass("vector<float>");
   auto *pointVecClass = TClass::GetClass("vector<DataPoint>");

   // --- small std::vector<float> ---
   std::vector<float> orig_small_f(1000);
   for (int i = 0; i < 1000; ++i)
      orig_small_f[i] = float(i) * 0.5f;

   auto startSmallF = b.GetCurrent() - b.Buffer();
   b.StreamObject(&orig_small_f, floatVecClass);

   // --- large std::vector<float> (512 M entries = 2 GB) ---
   std::vector<float> orig_large_f(512 * 1024 * 1024ll);
   for (size_t i = 0; i < orig_large_f.size(); ++i)
      orig_large_f[i] = float(i % 65536);

   auto startLargeF = b.GetCurrent() - b.Buffer();
   b.StreamObject(&orig_large_f, floatVecClass);

   // --- small std::vector<DataPoint> ---
   std::vector<DataPoint> orig_small_p(500);
   for (int i = 0; i < 500; ++i)
      orig_small_p[i] = {float(i), float(i * 2), float(i * 3)};

   auto startSmallP = b.GetCurrent() - b.Buffer();
   b.StreamObject(&orig_small_p, pointVecClass);

   // --- read back small float vector ---
   b.SetReadMode();
   b.SetBufferOffset(startSmallF);
   {
      std::vector<float> got;
      b.StreamObject(&got, floatVecClass);
      if (got != orig_small_f) {
         std::cerr << "testDirectVector: small float vector content mismatch\n";
         ++errors;
      }
   }

   // --- read back large float vector ---
   b.SetBufferOffset(startLargeF);
   {
      std::vector<float> got;
      b.StreamObject(&got, floatVecClass);
      if (got != orig_large_f) {
         std::cerr << "testDirectVector: large float vector content mismatch\n";
         ++errors;
      }
   }

   // --- read back small DataPoint vector ---
   b.SetBufferOffset(startSmallP);
   {
      std::vector<DataPoint> got;
      b.StreamObject(&got, pointVecClass);
      if (got != orig_small_p) {
         std::cerr << "testDirectVector: small DataPoint vector content mismatch\n";
         ++errors;
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// 3. Store as part of an object (numerical + struct members)
// -----------------------------------------------------------------------
static int readAndCheckFixture(const std::string &msg, TBufferFile &b, Long64_t startPos, size_t expectedFloats,
                               size_t expectedPoints)
{
   b.SetReadMode();
   b.SetBufferOffset(startPos);
   auto *obj = b.ReadObjectAny(TClass::GetClass(typeid(LargeCollectionFixture)));
   if (!obj) {
      std::cerr << msg << ": Failed to read back object\n";
      return 1;
   }
   auto *f      = static_cast<LargeCollectionFixture *>(obj);
   int   errors = 0;
   if (f->fFloats.size() != expectedFloats) {
      std::cerr << msg << ": fFloats size mismatch: got " << f->fFloats.size() << " expected " << expectedFloats
                << '\n';
      ++errors;
   }
   if (f->fPoints.size() != expectedPoints) {
      std::cerr << msg << ": fPoints size mismatch: got " << f->fPoints.size() << " expected " << expectedPoints
                << '\n';
      ++errors;
   }
   delete f;
   return errors;
}

int testAsPartOfObject()
{
   int errors = 0;

   std::vector<char> raw;
   raw.reserve(10 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 10 * 1024 * 1024 * 1024ll - 100, raw.data(), false /* don't adopt */,
                 DoNothingAllocator);

   LargeCollectionFixture fixture;

   // --- small object (well within 2 GB region) ---
   fixture.fFloats.assign(1000, 1.0f);
   fixture.fPoints.assign(500, {1.f, 2.f, 3.f});
   auto startSmall = b.GetCurrent() - b.Buffer();
   b.WriteObject(&fixture, false /* cacheReuse */);
   errors += readAndCheckFixture("small object", b, startSmall, 1000, 500);

   // --- large object (floats cross 2 GB, written in regular section) ---
   b.SetWriteMode();
   fixture.fFloats.assign(2 * 1024 * 1024 * 1024ll, 2.0f); // 2 G of floats (8GB)
   fixture.fPoints.assign(100, {4.f, 5.f, 6.f});
   auto startLarge = b.GetCurrent() - b.Buffer();
   b.WriteObject(&fixture, false /* cacheReuse */);
   errors += readAndCheckFixture("large object in regular section", b, startLarge, 2 * 1024 * 1024 * 1024ll, 100);

   // --- large object written past the 4 GB mark ---
   b.SetWriteMode();
   b.SetBufferOffset(4 * 1024 * 1024 * 1024ll + 100);
   fixture.fFloats.assign(256 * 1024 * 1024ll, 3.0f); // 1 GB of floats
   fixture.fPoints.assign(200, {7.f, 8.f, 9.f});
   auto startFar = b.GetCurrent() - b.Buffer();
   b.WriteObject(&fixture, false /* cacheReuse */);
   errors += readAndCheckFixture("large object in long-range section", b, startFar, 256 * 1024 * 1024ll, 200);

   return errors;
}

// -----------------------------------------------------------------------
// 4. Nested: a vector of LargeCollectionFixture objects
// -----------------------------------------------------------------------
int testNested()
{
   int errors = 0;

   std::vector<char> raw;
   raw.reserve(6 * 1024 * 1024 * 1024ll);
   TBufferFile b(TBuffer::kWrite, 6 * 1024 * 1024 * 1024ll - 100, raw.data(), false /* don't adopt */,
                 DoNothingAllocator);

   auto                               *fixtureClass = TClass::GetClass(typeid(LargeCollectionFixture));
   const Long64_t                      nObjects     = 3;
   std::vector<LargeCollectionFixture> objs(nObjects);
   // Give each fixture a different size so we can verify round-trip.
   objs[0].fFloats.assign(100, 1.0f);
   objs[0].fPoints.assign(10, {1, 1, 1});
   objs[1].fFloats.assign(200, 2.0f);
   objs[1].fPoints.assign(20, {2, 2, 2});
   objs[2].fFloats.assign(300, 3.0f);
   objs[2].fPoints.assign(30, {3, 3, 3});

   auto startPos = b.GetCurrent() - b.Buffer();
   b << nObjects;
   b.WriteFastArray(objs.data(), fixtureClass, nObjects);

   b.SetReadMode();
   b.SetBufferOffset(startPos);
   {
      Long64_t n = 0;
      b >> n;
      if (n != nObjects) {
         std::cerr << "testNested: object count mismatch: got " << n << '\n';
         ++errors;
      } else {
         std::vector<LargeCollectionFixture> got(n);
         b.ReadFastArray(got.data(), fixtureClass, n);
         for (Long64_t i = 0; i < n; ++i) {
            if (got[i].fFloats != objs[i].fFloats || got[i].fPoints != objs[i].fPoints) {
               std::cerr << "testNested: content mismatch at index " << i << '\n';
               ++errors;
            }
         }
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------
int testLargeCollection()
{
   int errors = 0;

   std::cerr << "testDirectNumerical ...\n";
   errors += testDirectNumerical();

   std::cerr << "testDirectStruct ...\n";
   errors += testDirectStruct();

   std::cerr << "testDirectVector ...\n";
   errors += testDirectVector();

   std::cerr << "testAsPartOfObject ...\n";
   errors += testAsPartOfObject();

   std::cerr << "testNested ...\n";
   errors += testNested();

   std::cerr << "Done. errors=" << errors << '\n';
   return errors;
}
