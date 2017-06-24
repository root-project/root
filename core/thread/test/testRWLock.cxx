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
void testWriteLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->WriteLock();
   }
}

template <typename M>
void testReadLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->ReadLock();
   }
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
void testWriteUnLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->WriteUnLock();
   }
}

template <typename M>
void testReadUnLock(M *m, size_t repetition)
{
   for (size_t i = 0; i < repetition; ++i) {
      m->ReadUnLock();
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
      m->ReadLock();
      m->ReadUnLock();
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
      m->ReadLock();
      ASSERT_EQ(global->fFirst, global->fThird);
      m->ReadUnLock();
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
   m.ReadLock();

   m.WriteLock();

   m.ReadLock();
   m.ReadLock();

   m.WriteLock();

   m.ReadLock();

   m.ReadUnLock();
   m.WriteUnLock();
   m.ReadUnLock();
   m.ReadUnLock();
   m.WriteUnLock();
   m.ReadUnLock();
   m.ReadUnLock();
   m.ReadUnLock();
}

constexpr size_t gRepetition = 10000000;

auto gMutex = new TMutex(kTRUE);
auto gRWMutex = new TRWMutexImp<TMutex>();
auto gRWMutexSpin = new TRWMutexImp<ROOT::TSpinMutex>();
auto gReentrantRWMutex = new ROOT::TReentrantRWLock<TMutex>();
auto gReentrantRWMutexSM = new ROOT::TReentrantRWLock<ROOT::TSpinMutex>();
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

TEST(RWLock, WriteSpinDirectLock)
{
   testWriteLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, WriteSpinDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, WriteDirectLock)
{
   testWriteLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, WriteDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, ReadLockSpinDirect)
{
   testReadLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, ReadUnLockSpinDirect)
{
   testReadUnLock(gReentrantRWMutexSM, gRepetition);
}

TEST(RWLock, ReadLockDirect)
{
   testReadLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, ReadUnLockDirect)
{
   testReadUnLock(gReentrantRWMutex, gRepetition);
}

TEST(RWLock, WriteSpinTLDirectLock)
{
   testWriteLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, WriteSpinTLsDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, WriteTLDirectLock)
{
   testWriteLock(gReentrantRWMutexTL, gRepetition);
}

TEST(RWLock, WriteTLDirectUnLock)
{
   testWriteUnLock(gReentrantRWMutexTL, gRepetition);
}

TEST(RWLock, ReadLockSpinTLDirect)
{
   testReadLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, ReadUnLockSpinTLDirect)
{
   testReadUnLock(gReentrantRWMutexSMTL, gRepetition);
}

TEST(RWLock, ReadLockTLDirect)
{
   testReadLock(gReentrantRWMutexTL, gRepetition);
}

TEST(RWLock, ReadUnLockTLDirect)
{
   testReadUnLock(gReentrantRWMutexTL, gRepetition);
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

TEST(RWLock, concurrentReadsAndWrites)
{
   concurrentReadsAndWrites(gRWMutex, 1, 2, gRepetition / 10000);
}

TEST(RWLock, concurrentReadsAndWritesSpin)
{
   concurrentReadsAndWrites(gRWMutexSpin, 1, 2, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentReadsAndWrites)
{
   concurrentReadsAndWrites(gRWMutex, 10, 20, gRepetition / 1000);
}

// TEST(RWLock, LargeconcurrentReadsAndWritesSpin)
// {
//    concurrentReadsAndWrites(gRWMutexSpin,10,20,gRepetition / 1000);
// }

TEST(RWLock, concurrentReadsAndWritesTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 1, 2, gRepetition / 10000);
}

TEST(RWLock, LargeconcurrentReadsAndWritesTL)
{
   concurrentReadsAndWrites(gRWMutexTL, 10, 20, gRepetition / 1000);
}
