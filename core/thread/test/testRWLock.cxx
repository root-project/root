// #include "TReentrantRWLock.h"
#include "TVirtualMutex.h"
#include "TMutex.h"
#include "TVirtualRWMutex.h"
#include "ROOT/TReentrantRWLock.hxx"
#include "ROOT/TRWSpinLock.hxx"

#include "../src/TRWMutexImp.h"

#include "TSystem.h"
#include "TROOT.h"
#include "TError.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ROOT;

void testWriteLockV(TVirtualMutex *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->Lock();
   }
}

void testWriteUnLockV(TVirtualMutex *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->UnLock();
   }
}

template <typename M>
void testWriteTLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->Lock();
   }
}

template <typename M>
TVirtualRWMutex::Hint_t *testWriteLock(M *m, size_t repetition)
{
   TVirtualRWMutex::Hint_t *hint = nullptr;
   for (size_t i = 0; i < repetition; ++i) {
      hint = m->WriteLock();
   }
   return hint;
}

template <typename M>
TVirtualRWMutex::Hint_t *testReadLock(M *m, size_t repetition)
{
   TVirtualRWMutex::Hint_t *hint = nullptr;
   for (size_t i = 0; i < repetition; ++i) {
      hint = m->ReadLock();
   }
   // hint is always the same for a given thread.
   return hint;
}

template <typename M>
void testNonReentrantLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->lock();
      m->unlock();
   }
}

template <typename M>
void testWriteTUnLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->UnLock();
   }
}

template <typename M>
void testWriteUnLock(M *m, size_t repetition, TVirtualRWMutex::Hint_t *hint)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->WriteUnLock(hint);
   }
}

template <typename M>
void testReadUnLock(M *m, size_t repetition, TVirtualRWMutex::Hint_t *hint)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->ReadUnLock(hint);
   }
}

void testWriteGuard(TVirtualMutex *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      TLockGuard guard(m);
   }
}

void testReadGuard(TVirtualRWMutex *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      auto hint = m->ReadLock();
      m->ReadUnLock(hint);
   }
}

struct Globals {
   size_t fFirst = 0;
   size_t fSecond = 0;
   size_t fThird = 0;
};

void writer(TVirtualRWMutex *m, Globals *global, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      {
         TLockGuard guard(m);
         global->fFirst++;
         // Waste some time
         for (size_t k = 0; k < 100; ++k) {
            global->fSecond += global->fThird * global->fFirst + k;
         }
         global->fThird++;
      }
      gSystem->Sleep(3 /* milliseconds */); // give sometimes to the readers
   }
}

void reader(TVirtualRWMutex *m, Globals *global, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      auto hint = m->ReadLock();
      ASSERT_EQ(global->fFirst, global->fThird);
      m->ReadUnLock(hint);
      gSystem->Sleep(1 /* milliseconds */); // give sometimes to the writers
   }
}

void concurrentReadsAndWrites(TVirtualRWMutex *m, size_t nwriters, size_t nreaders, size_t repetition)
{
   // ROOT::EnableThreadSafety();

   std::vector<std::thread> threads;

   Globals global;

   for (size_t i = 0; i < nwriters; ++i) {
      threads.push_back(std::thread([&]() { writer(m, &global, repetition); }));
   }
   for (size_t i = 0; i < nreaders; ++i) {
      threads.push_back(std::thread([&]() { reader(m, &global, repetition); }));
   }

   for (auto &&th : threads) {
      th.join();
   }
}

template <typename T>
void Reentrant(T &m)
{

   m.ReadLock();
   m.ReadLock();
   auto rhint = m.ReadLock();

   auto whint = m.WriteLock();

   m.ReadLock();
   m.ReadLock();

   m.WriteLock();

   m.ReadLock();

   m.ReadUnLock(rhint);
   m.WriteUnLock(whint);
   m.ReadUnLock(rhint);
   m.ReadUnLock(rhint);
   m.WriteUnLock(whint);
   m.ReadUnLock(rhint);
   m.ReadUnLock(rhint);
   m.ReadUnLock(rhint);
}

template <typename T>
void ResetRestore(T &m, size_t repeat = 1)
{
   do {
      auto whint0 = m.WriteLock();
      auto state = m.GetStateBefore();
      auto rhint = m.ReadLock();
      m.Apply( m.Rewind(*state.get()) );
      m.ReadUnLock(rhint);

      m.ReadLock();
      m.ReadLock();
      m.ReadLock();
      m.Apply( m.Rewind(*state.get()) );
      m.ReadUnLock(rhint);
      m.ReadUnLock(rhint);
      m.ReadUnLock(rhint);

      auto whint = m.WriteLock();
      m.Apply( m.Rewind(*state.get()) );
      m.WriteUnLock(whint);


      m.ReadLock();
      m.ReadLock();
      m.ReadLock();
      m.WriteLock();
      m.ReadLock();
      m.ReadLock();
      m.WriteLock();
      m.ReadLock();
      m.Apply( m.Rewind(*state.get()) );
      m.ReadUnLock(rhint);
      m.WriteUnLock(whint);
      m.ReadUnLock(rhint);
      m.ReadUnLock(rhint);
      m.WriteUnLock(whint);
      m.ReadUnLock(rhint);
      m.ReadUnLock(rhint);
      m.ReadUnLock(rhint);
      m.WriteUnLock(whint0);
   } while ( --repeat > 0 );
}

void concurrentResetRestore(TVirtualRWMutex *m, size_t nthreads, size_t repetition)
{
   // ROOT::EnableThreadSafety();

   std::vector<std::thread> threads;

   for (size_t i = 0; i < nthreads; ++i) {
      threads.push_back(std::thread([&]() { ResetRestore(*m, repetition); }));
   }

   for (auto &&th : threads) {
      th.join();
   }
}

constexpr size_t gRepetition = 10000000;

auto gMutex = new TMutex(kTRUE);
auto gRWMutex = new TRWMutexImp<TMutex>();
auto gRWMutexSpin = new TRWMutexImp<ROOT::TSpinMutex>();
auto gRWMutexStd = new TRWMutexImp<std::mutex>();
#ifdef R__HAS_TBB
auto gRWMutexStdTBB = new TRWMutexImp<std::mutex, ROOT::Internal::RecurseCountsTBB>();
auto gRWMutexStdTBBUnique = new TRWMutexImp<std::mutex, ROOT::Internal::RecurseCountsTBBUnique>();
#endif
auto gReentrantRWMutex = new ROOT::TReentrantRWLock<TMutex>();
auto gReentrantRWMutexSM = new ROOT::TReentrantRWLock<ROOT::TSpinMutex>();
auto gReentrantRWMutexStd = new ROOT::TReentrantRWLock<std::mutex>();
#ifdef R__HAS_TBB
auto gReentrantRWMutexStdTBB = new ROOT::TReentrantRWLock<std::mutex, ROOT::Internal::RecurseCountsTBB>();
auto gReentrantRWMutexStdTBBUnique = new ROOT::TReentrantRWLock<std::mutex, ROOT::Internal::RecurseCountsTBBUnique>();
#endif
auto gSpinMutex = new ROOT::TSpinMutex();

// Intentionally ignore the Fatal error due to the shread thread-local storage.
// In this test we need to be 'careful' to not use all those mutex at the same time.
int trigger1 = gErrorIgnoreLevel = kFatal + 1;
auto gReentrantRWMutexTL = new ROOT::TReentrantRWLock<TMutex, ROOT::Internal::UniqueLockRecurseCount>();
auto gReentrantRWMutexSMTL = new ROOT::TReentrantRWLock<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>();
auto gRWMutexTL = new TRWMutexImp<TMutex, ROOT::Internal::UniqueLockRecurseCount>();
auto gRWMutexTLSpin = new TRWMutexImp<ROOT::TSpinMutex, ROOT::Internal::UniqueLockRecurseCount>();
int trigger2 = gErrorIgnoreLevel = 0;

TEST(RWLock, MutexLockVirtual)
{
   testWriteLockV(gMutex, gRepetition);
}

TEST(RWLock, MutexUnLockVirtual)
{
   testWriteTUnLock(gMutex, gRepetition);
}

TEST(RWLock, WriteLockVirtual)
{
   testWriteLockV(gRWMutex, gRepetition);
}

TEST(RWLock, WriteUnLockVirtual)
{
   testWriteTUnLock(gRWMutex, gRepetition);
}

TEST(RWLock, WriteSpinLockVirtual)
{
   testWriteLock(gRWMutexSpin, gRepetition);
}

TEST(RWLock, WriteSpinUnLockVirtual)
{
   testWriteTUnLock(gRWMutexSpin, gRepetition);
}

TEST(RWLock, WriteLock)
{
   testWriteLock(gRWMutex, gRepetition);
}

TEST(RWLock, WriteUnLock)
{
   testWriteTUnLock(gRWMutex, gRepetition);
}

TEST(RWLock, WriteSpinLock)
{
   testWriteLock(gRWMutexSpin, gRepetition);
}

TEST(RWLock, WriteSpinUnLock)
{
   testWriteTUnLock(gRWMutexSpin, gRepetition);
}

static TVirtualRWMutex::Hint_t *gWriteHint = nullptr;
static TVirtualRWMutex::Hint_t *gReadHint = nullptr;

TEST(RWLock, WriteStdDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexStd, gRepetition);
}

TEST(RWLock, WriteStdDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexStd, gRepetition, gWriteHint);
}

#ifdef R__HAS_TBB
TEST(RWLock, WriteStdTBBDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexStdTBB, gRepetition);
}

TEST(RWLock, WriteStdTBBDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexStdTBB, gRepetition, gWriteHint);
}

TEST(RWLock, WriteStdTBBUniqueDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexStdTBBUnique, gRepetition);
}

TEST(RWLock, WriteStdTBBUniqueDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexStdTBBUnique, gRepetition, gWriteHint);
}
#endif

TEST(RWLock, WriteSpinDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, WriteSpinDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexSM, gRepetition, gWriteHint);
}

TEST(RWLock, WriteDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, WriteDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutex, gRepetition, gWriteHint);
}

TEST(RWLock, ReadLockStdDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexStd, gRepetition);
}

TEST(RWLock, ReadUnLockStdDirect)
{
   testReadUnLock(gReentrantRWMutexStd, gRepetition, gReadHint);
}

#ifdef R__HAS_TBB
TEST(RWLock, ReadLockStdTBBDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexStdTBB, gRepetition);
}

TEST(RWLock, ReadUnLockStdTBBDirect)
{
   testReadUnLock(gReentrantRWMutexStdTBB, gRepetition, gReadHint);
}

TEST(RWLock, ReadLockStdTBBUniqueDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexStdTBBUnique, gRepetition);
}

TEST(RWLock, ReadUnLockStdTBBUniqueDirect)
{
   testReadUnLock(gReentrantRWMutexStdTBBUnique, gRepetition, gReadHint);
}
#endif

TEST(RWLock, ReadLockSpinDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, ReadUnLockSpinDirect)
{
   testReadUnLock(gReentrantRWMutexSM, gRepetition, gReadHint);
}

TEST(RWLock, ReadLockDirect)
{
   gReadHint = testReadLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, ReadUnLockDirect)
{
   testReadUnLock(gReentrantRWMutex, gRepetition, gReadHint);
}

TEST(RWLock, WriteSpinTLDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, WriteSpinTLsDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexSMTL, gRepetition, gWriteHint);
}

TEST(RWLock, WriteTLDirectLock)
{
   gWriteHint = testWriteLock(gReentrantRWMutexTL, gRepetition);
}

TEST(RWLock, WriteTLDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexTL, gRepetition, gWriteHint);
}

TEST(RWLock, ReadLockSpinTLDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, ReadUnLockSpinTLDirect)
{
   testReadUnLock(gReentrantRWMutexSMTL, gRepetition, gReadHint);
}

TEST(RWLock, ReadLockTLDirect)
{
   gReadHint = testReadLock(gReentrantRWMutexTL, gRepetition);
}

TEST(RWLock, ReadUnLockTLDirect)
{
   testReadUnLock(gReentrantRWMutexTL, gRepetition, gReadHint);
}

TEST(RWLock, SpinMutexLockUnlock)
{
   testNonReentrantLock(gSpinMutex, gRepetition);
}

TEST(RWLock, MutexGuard)
{
   testWriteGuard(gMutex, gRepetition);
}

TEST(RWLock, WriteGuard)
{
   testWriteGuard(gRWMutex, gRepetition);
}

TEST(RWLock, WriteSpinGuard)
{
   testWriteGuard(gRWMutexSpin, gRepetition);
}

TEST(RWLock, ReentrantStd)
{
   Reentrant(*gReentrantRWMutexStd);
}

#ifdef R__HAS_TBB
TEST(RWLock, ReentrantStdTBB)
{
   Reentrant(*gReentrantRWMutexStdTBB);
}

TEST(RWLock, ReentrantStdTBBUnique)
{
   Reentrant(*gReentrantRWMutexStdTBBUnique);
}
#endif

TEST(RWLock, ReentrantSpin)
{
   Reentrant(*gReentrantRWMutexSM);
}

TEST(RWLock, Reentrant)
{
   Reentrant(*gReentrantRWMutex);
}

TEST(RWLock, ReentrantTLSpin)
{
   Reentrant(*gReentrantRWMutexSMTL);
}

TEST(RWLock, ReentrantTL)
{
   Reentrant(*gReentrantRWMutexTL);
}

TEST(RWLock, ResetRestoreStd)
{
   ResetRestore(*gReentrantRWMutexStd);
}

#ifdef R__HAS_TBB
TEST(RWLock, ResetRestoreStdTBB)
{
   ResetRestore(*gReentrantRWMutexStdTBB);
}

TEST(RWLock, ResetRestoreStdTBBUnique)
{
   ResetRestore(*gReentrantRWMutexStdTBBUnique);
}
#endif

TEST(RWLock, ResetRestoreSpin)
{
   ResetRestore(*gReentrantRWMutexSM);
}

TEST(RWLock, ResetRestore)
{
   ResetRestore(*gReentrantRWMutex);
}

TEST(RWLock, ResetRestoreTLSpin)
{
   ResetRestore(*gReentrantRWMutexSMTL);
}

TEST(RWLock, ResetRestoreTL)
{
   ResetRestore(*gReentrantRWMutexTL);
}


TEST(RWLock, concurrentResetRestore)
{
   concurrentResetRestore(gRWMutex, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentResetRestoreSpin)
{
   concurrentResetRestore(gRWMutexSpin, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentResetRestoreStd)
{
   concurrentResetRestore(gRWMutexStd, 2, gRepetition / 10000);
}

#ifdef R__HAS_TBB
TEST(RWLock, concurrentResetRestoreStdTBB)
{
   concurrentResetRestore(gRWMutexStdTBB, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentResetRestoreStdTBBUnique)
{
   concurrentResetRestore(gRWMutexStdTBBUnique, 2, gRepetition / 10000);
}
#endif

TEST(RWLock, LargeconcurrentResetRestore)
{
   concurrentResetRestore(gRWMutex, 20, gRepetition / 40000);
}

// TEST(RWLock, LargeconcurrentResetRestoreSpin)
// {
//    concurrentResetRestore(gRWMutexSpin,20,gRepetition / 1000);
// }

TEST(RWLock, concurrentResetRestoreTL)
{
   concurrentResetRestore(gRWMutexTL, 2, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentResetRestoreTL)
{
   concurrentResetRestore(gRWMutexTL, 20, gRepetition / 40000);
}




TEST(RWLock, concurrentReadsAndWrites)
{
   concurrentReadsAndWrites(gRWMutex, 1, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentReadsAndWritesSpin)
{
   concurrentReadsAndWrites(gRWMutexSpin, 1, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentReadsAndWritesStd)
{
   concurrentReadsAndWrites(gRWMutexStd, 1, 2, gRepetition / 10000);
}

#ifdef R__HAS_TBB
TEST(RWLock, concurrentReadsAndWritesStdTBB)
{
   concurrentReadsAndWrites(gRWMutexStdTBB, 1, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentReadsAndWritesStdTBBUnique)
{
   concurrentReadsAndWrites(gRWMutexStdTBBUnique, 1, 2, gRepetition / 10000);
}
#endif

TEST(RWLock, LargeconcurrentReadsAndWrites)
{
   concurrentReadsAndWrites(gRWMutex, 10, 20, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentReadsAndWritesStd)
{
   concurrentReadsAndWrites(gRWMutex, 10, 20, gRepetition / 10000);
}

#ifdef R__HAS_TBB
TEST(RWLock, LargeconcurrentReadsAndWritesStdTBB)
{
   concurrentReadsAndWrites(gRWMutexStdTBB, 10, 20, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentReadsAndWritesStdTBBUnique)
{
   concurrentReadsAndWrites(gRWMutexStdTBBUnique, 10, 20, gRepetition / 10000);
}
#endif

TEST(RWLock, LargeconcurrentReadsAndWritesSpin)
{
   concurrentReadsAndWrites(gRWMutexSpin,10,20,gRepetition / 100000);
}

TEST(RWLock, VeryLargeconcurrentReadsAndWrites)
{
   concurrentReadsAndWrites(gRWMutex, 10, 200, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsAndWritesStd)
{
   concurrentReadsAndWrites(gRWMutexStd, 10, 200, gRepetition / 10000);
}

#ifdef R__HAS_TBB
TEST(RWLock, VeryLargeconcurrentReadsAndWritesStdTBB)
{
   concurrentReadsAndWrites(gRWMutexStdTBB, 10, 200, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsAndWritesStdTBBUnique)
{
   concurrentReadsAndWrites(gRWMutexStdTBBUnique, 10, 200, gRepetition / 10000);
}
#endif

// this can take a few minutes in some configurations
// TEST(RWLock, VeryLargeconcurrentReadsAndWritesSpin)
// {
//    concurrentReadsAndWrites(gRWMutexSpin,10,200,gRepetition / 100000);
// }

TEST(RWLock, VeryLargeconcurrentReads)
{
   concurrentReadsAndWrites(gRWMutex, 0, 200, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsStd)
{
   concurrentReadsAndWrites(gRWMutexStd, 0, 200, gRepetition / 10000);
}

#ifdef R__HAS_TBB
TEST(RWLock, VeryLargeconcurrentReadsStdTBB)
{
   concurrentReadsAndWrites(gRWMutexStdTBB, 0, 200, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsStdTBBUnique)
{
   concurrentReadsAndWrites(gRWMutexStdTBBUnique, 0, 200, gRepetition / 10000);
}
#endif

TEST(RWLock, VeryLargeconcurrentReadsSpin)
{
   concurrentReadsAndWrites(gRWMutexSpin,0,200,gRepetition / 100000);
}

TEST(RWLock, concurrentReadsAndWritesTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 1, 2, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentReadsAndWritesTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 10, 20, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsAndWritesTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 10, 200, gRepetition / 10000);
}

TEST(RWLock, VeryLargeconcurrentReadsTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 0, 200, gRepetition / 10000);
}
