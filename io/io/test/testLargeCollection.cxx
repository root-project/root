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

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

// Timing helper — writes to a dedicated file, not captured by the ctest driver.
static std::ofstream &timingLog()
{
   static std::ofstream f("/tmp/testLargeCollection_timing.txt");
   return f;
}
static double now_sec()
{
   using namespace std::chrono;
   return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Returns the process peak RSS in MB.
// ru_maxrss is bytes on macOS, kilobytes on Linux.
static double peak_rss_mb()
{
   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
   return ru.ru_maxrss / (1024.0 * 1024.0);
#else
   return ru.ru_maxrss / 1024.0;
#endif
}
// Returns the available (free) physical memory in MB.
static double available_mem_mb()
{
#if defined(__APPLE__)
   uint64_t memsize = 0;
   size_t len = sizeof(memsize);
   // vm.pagesize * vm_stat free pages gives available memory
   int pagesize = 0;
   size_t plen = sizeof(pagesize);
   sysctlbyname("hw.pagesize", &pagesize, &plen, nullptr, 0);
   // Use hw.memsize for total; approximate available via sysctl vm.page_free_count
   uint64_t pagefree = 0;
   size_t pflen = sizeof(pagefree);
   if (sysctlbyname("vm.page_free_count", &pagefree, &pflen, nullptr, 0) == 0 && pagesize > 0)
      return (pagefree * (double)pagesize) / (1024.0 * 1024.0);
   // fallback: total physical memory
   sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
   return memsize / (1024.0 * 1024.0);
#elif defined(__linux__)
   struct sysinfo si;
   if (sysinfo(&si) == 0)
      return (si.freeram * (double)si.mem_unit) / (1024.0 * 1024.0);
   return -1.0;
#else
   return -1.0;
#endif
}

// Returns the available (free) swap space in MB.
static double available_swap_mb()
{
#if defined(__APPLE__)
   struct xsw_usage swapinfo;
   size_t len = sizeof(swapinfo);
   if (sysctlbyname("vm.swapusage", &swapinfo, &len, nullptr, 0) == 0)
      return swapinfo.xsu_avail / (1024.0 * 1024.0);
   return -1.0;
#elif defined(__linux__)
   struct sysinfo si;
   if (sysinfo(&si) == 0)
      return (si.freeswap * (double)si.mem_unit) / (1024.0 * 1024.0);
   return -1.0;
#elif defined(_WIN32)
   MEMORYSTATUSEX ms;
   ms.dwLength = sizeof(ms);
   if (GlobalMemoryStatusEx(&ms))
      return ms.ullAvailPageFile / (1024.0 * 1024.0);
   return -1.0;
#else
   return -1.0;
#endif
}

#define TIME_SUBTEST(timing, label, call)                                               \
   do {                                                                                 \
      std::cerr << (label) << " ...\n";                                                \
      double _t0 = (timing) ? now_sec() : 0.0;                                        \
      errors += (call);                                                                 \
      if (timing) {                                                                     \
         timingLog() << (label) << " done in " << (now_sec() - _t0)                   \
                     << " s  peak RSS: " << peak_rss_mb() << " MB\n";                 \
         timingLog().flush();                                                           \
      }                                                                                 \
   } while (0)
#define TLOG(timing, msg) do { if (timing) { timingLog() << msg << "\n"; timingLog().flush(); } } while(0)

// -----------------------------------------------------------------------
// Spot-check a large array: verify the first/last kSpot elements and
// kSpot elements centred on the 2 GB boundary, rather than comparing
// the entire array element-by-element.
// -----------------------------------------------------------------------
template <typename T>
static bool spotCheck(const std::vector<T> &got, const std::vector<T> &expected, const char *tag)
{
   if (got.size() != expected.size()) {
      std::cerr << tag << ": size mismatch " << got.size() << " vs " << expected.size() << '\n';
      return false;
   }
   constexpr Long64_t kSpot     = 1024;
   const Long64_t     n         = (Long64_t)got.size();
   const Long64_t     boundaryN = (Long64_t)(2LL * 1024 * 1024 * 1024 / sizeof(T));
   bool               ok        = true;

   auto check = [&](Long64_t i) {
      if (i < 0 || i >= n)
         return;
      if (!(got[i] == expected[i])) {
         if (ok)
            std::cerr << tag << ": mismatch at index " << i << '\n';
         ok = false;
      }
   };

   for (Long64_t i = 0; i < kSpot; ++i)
      check(i); // beginning
   for (Long64_t i = boundaryN - kSpot; i < boundaryN + kSpot; ++i)
      check(i); // around 2 GB boundary
   for (Long64_t i = n - kSpot; i < n; ++i)
      check(i); // end

   return ok;
}

// Variant of spotCheck that computes expected values on-the-fly via a
// generator, so the caller can free the source array before read-back.
template <typename T, typename Gen>
static bool spotCheckFn(const std::vector<T> &got, Long64_t expectedSize, Gen expectedAt, const char *tag)
{
   if ((Long64_t)got.size() != expectedSize) {
      std::cerr << tag << ": size mismatch " << got.size() << " vs " << expectedSize << '\n';
      return false;
   }
   constexpr Long64_t kSpot     = 1024;
   const Long64_t     n         = expectedSize;
   const Long64_t     boundaryN = (Long64_t)(2LL * 1024 * 1024 * 1024 / sizeof(T));
   bool               ok        = true;

   auto check = [&](Long64_t i) {
      if (i < 0 || i >= n)
         return;
      if (!(got[i] == expectedAt(i))) {
         if (ok)
            std::cerr << tag << ": mismatch at index " << i << '\n';
         ok = false;
      }
   };

   for (Long64_t i = 0; i < kSpot; ++i)
      check(i);
   for (Long64_t i = boundaryN - kSpot; i < boundaryN + kSpot; ++i)
      check(i);
   for (Long64_t i = n - kSpot; i < n; ++i)
      check(i);

   return ok;
}

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
//    Returns the large float vector so testDirectVector can reuse it.
// -----------------------------------------------------------------------
int testDirectNumerical(std::vector<float> &orig_large_out)
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
   const Long64_t largeN = 512 * 1024 * 1024ll;
   orig_large_out.resize(largeN);
   for (Long64_t i = 0; i < largeN; ++i)
      orig_large_out[i] = float(i & 0xFFFF); // 65536 = 2^16; & is faster than %

   auto startLarge = b.GetCurrent() - b.Buffer();
   b << largeN;
   b.WriteFastArray(orig_large_out.data(), largeN);

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
         if (!spotCheck(got, orig_large_out, "testDirectNumerical large"))
            ++errors;
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// 2. Direct store of a struct array in a TBufferFile
// -----------------------------------------------------------------------
int testDirectStruct(bool timing = false)
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
   // Use bitwise AND (power-of-2 period) instead of % 1000 to avoid
   // 510 M integer divisions in this fill loop.
   { double t0 = now_sec();
   for (Long64_t i = 0; i < largeN; ++i)
      orig_large[i] = {float(i & 0x3FF), float((i + 1) & 0x3FF), float((i + 2) & 0x3FF)};
   TLOG(timing, "  testDirectStruct fill:  " << (now_sec()-t0) << " s  peak RSS: " << peak_rss_mb() << " MB"); }

   auto startLarge = b.GetCurrent() - b.Buffer();
   b << largeN;
   { double t0 = now_sec();
   b.WriteFastArray(orig_large.data(), pointClass, largeN);
   TLOG(timing, "  testDirectStruct write: " << (now_sec()-t0) << " s  peak RSS: " << peak_rss_mb() << " MB"); }

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
         { double t0 = now_sec();
         b.ReadFastArray(got.data(), pointClass, n);
         TLOG(timing, "  testDirectStruct read:  " << (now_sec()-t0) << " s  peak RSS: " << peak_rss_mb() << " MB"); }
         if (!spotCheckFn(got, largeN,
                          [](Long64_t i) {
                             return DataPoint{float(i & 0x3FF), float((i + 1) & 0x3FF), float((i + 2) & 0x3FF)};
                          },
                          "testDirectStruct large"))
            ++errors;
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// 2b. Direct store of std::vector<T> via StreamObject
//     Takes the already-built large float vector from testDirectNumerical
//     to avoid re-allocating and re-filling 2 GB of data.
// -----------------------------------------------------------------------
int testDirectVector(std::vector<float> &orig_large_f)
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
   // orig_large_f is passed in from testDirectNumerical — no re-allocation needed.

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
      if (!spotCheckFn(got, 512 * 1024 * 1024ll,
                       [](Long64_t i) { return float(i & 0xFFFF); },
                       "testDirectVector large float"))
         ++errors;
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

   // Let TBufferFile own and manage its own buffer so it can auto-expand freely.
   // Start small; it will grow as needed (up to ~8+ GB for the 2G-float large object).
   TBufferFile b(TBuffer::kWrite, 1024);

   {
      LargeCollectionFixture fixture;

   // --- small object (well within 2 GB region) ---
   fixture.fFloats.assign(1000, 1.0f);
   fixture.fPoints.assign(500, {1.f, 2.f, 3.f});
   auto startSmall = b.GetCurrent() - b.Buffer();
   b.WriteObject(&fixture, false /* cacheReuse */);
   errors += readAndCheckFixture("small object", b, startSmall, 1000, 500);
   }
   LargeCollectionFixture fixture;

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
// 5. Minimal reproducer for two related >4 GB TBufferFile bugs:
//
//    Bug A (write-side): WriteObject on an object whose serialised content
//    crosses the 4 GB mark in the buffer.  SetByteCount fires an assert
//    because cntpos < 4 GB but fBufCur - fBuffer > 4 GB after streaming.
//    Minimal trigger: start writing just below 4 GB, payload > remaining space.
//
//    Bug B (read-side): ReadObjectAny on an object whose buffer start
//    position > kMaxUInt.  MapObject fires "offset <= kMaxUInt" assert.
//    Minimal trigger: seek to > 4 GB, WriteObject tiny fixture, ReadObjectAny.
// -----------------------------------------------------------------------
int testMapObjectLargeOffset()
{
   int errors = 0;

   // Each sub-test uses its own fresh TBufferFile so the class map is clean,
   // ensuring every read-back sees a genuine "new class" entry in ReadClass
   // and exercises the MapObject path at the relevant buffer offset.

   // --- Bug A: object START is just below 4 GB but the payload (10 M floats = 40 MB)
   //     pushes fBufCur past kMaxUInt.  SetByteCount / WriteObjectClass must handle
   //     a byte-count position that was recorded below 4 GB while the current
   //     position is above it.
   {
      const Long64_t kBufSize = (4LL * 1024 + 128) * 1024 * 1024; // 4.125 GB
      std::vector<char> raw(kBufSize, 0);
      TBufferFile b(TBuffer::kWrite, kBufSize, raw.data(), false /* don't adopt */, DoNothingAllocator);

      const Long64_t kStartA = 4LL * 1024 * 1024 * 1024 - 256;
      b.SetBufferOffset(kStartA);
      LargeCollectionFixture fixture;
      fixture.fFloats.assign(10 * 1024 * 1024, 1.0f); // 40 MB — crosses the 4 GB boundary
      fixture.fPoints.assign(3, {1.f, 2.f, 3.f});
      b.WriteObject(&fixture, false /* cacheReuse */);

      b.SetReadMode();
      b.SetBufferOffset(kStartA);
      auto *objA = b.ReadObjectAny(TClass::GetClass(typeid(LargeCollectionFixture)));
      if (!objA) {
         std::cerr << "testMapObjectLargeOffset BugA: ReadObjectAny returned null\n";
         ++errors;
      } else {
         auto *f = static_cast<LargeCollectionFixture *>(objA);
         if (f->fFloats.size() != 10u * 1024 * 1024 || f->fPoints.size() != 3) {
            std::cerr << "testMapObjectLargeOffset BugA: size mismatch\n";
            ++errors;
         }
         delete f;
      }
   }

   // --- Bug B: object written and read at a start position entirely above kMaxUInt.
   //     ReadObjectAny -> ReadClass -> MapObject(cl, startpos+kMapOffset) where
   //     startpos+kMapOffset > kMaxUInt, hitting R__ASSERT(offset <= kMaxUInt)
   //     in the TObject* overload of TBufferIO::MapObject.
   //     A fresh TBufferFile is required so that ReadClass sees a "new class"
   //     (not already cached) and calls MapObject with the >4 GB offset.
   {
      const Long64_t kBufSize = (4LL * 1024 + 128) * 1024 * 1024; // 4.125 GB
      std::vector<char> raw(kBufSize, 0);
      TBufferFile b(TBuffer::kWrite, kBufSize, raw.data(), false /* don't adopt */, DoNothingAllocator);

      const Long64_t kStartB = 4LL * 1024 * 1024 * 1024 + 100; // just above kMaxUInt
      b.SetBufferOffset(kStartB);
      LargeCollectionFixture fixture;
      fixture.fFloats.assign(10, 2.0f);
      fixture.fPoints.assign(2, {2.f, 3.f, 4.f});
      b.WriteObject(&fixture, false /* cacheReuse */);

      b.SetReadMode();
      b.SetBufferOffset(kStartB);
      auto *objB = b.ReadObjectAny(TClass::GetClass(typeid(LargeCollectionFixture)));
      if (!objB) {
         std::cerr << "testMapObjectLargeOffset BugB: ReadObjectAny returned null\n";
         ++errors;
      } else {
         auto *f = static_cast<LargeCollectionFixture *>(objB);
         if (f->fFloats.size() != 10 || f->fPoints.size() != 2) {
            std::cerr << "testMapObjectLargeOffset BugB: size mismatch\n";
            ++errors;
         }
         delete f;
      }
   }

   return errors;
}

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------
int testLargeCollection(bool timing = false, bool memoryCheck = false)
{
   int errors = 0;

   if (memoryCheck) {
      std::cerr << "Available memory at start: " << available_mem_mb() << " MB\n";
      double swap = available_swap_mb();
      if (swap >= 0.0)
         std::cerr << "Available swap at start:   " << swap << " MB\n";
   }

   std::vector<float> sharedLargeFloats;
   TIME_SUBTEST(timing, "testDirectNumerical", testDirectNumerical(sharedLargeFloats));
   TIME_SUBTEST(timing, "testDirectStruct",    testDirectStruct(timing));
   TIME_SUBTEST(timing, "testDirectVector",    testDirectVector(sharedLargeFloats));
   { std::vector<float>{}.swap(sharedLargeFloats); } // release 2 GB before the large-object test
   TIME_SUBTEST(timing, "testAsPartOfObject",  testAsPartOfObject());
   TIME_SUBTEST(timing, "testMapObjectLargeOffset", testMapObjectLargeOffset());
   TIME_SUBTEST(timing, "testNested",          testNested());

   std::cerr << "Done. errors=" << errors << '\n';
   return errors;
}
