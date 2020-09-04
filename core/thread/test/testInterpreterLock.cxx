#include "ROOT/TThreadExecutor.hxx"
#include "TVirtualRWMutex.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "gtest/gtest.h"

static auto gDeclarator =  gInterpreter->Declare("int f(int n) { int x = 0; for (auto i = 0u; i < n; ++i) ++x; return x; }");

constexpr bool gDebugOrder = false;

TEST(InterpreterLock, ConcurrentCalc)
{
   ASSERT_TRUE(gDeclarator);
   bool flag = true;
   auto func = [&](int) {
      gInterpreter->Calc("f(10)");
      R__LOCKGUARD(gROOTMutex);
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;

      EXPECT_TRUE(flag) << "Broken flag update on thread " << std::this_thread::get_id();
      flag = false;
      gInterpreter->Calc("f(100000)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : end" << std::endl;
      flag = true;
   };
   ROOT::TThreadExecutor pool(32);
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}


TEST(InterpreterLock, ReadLocks)
{
   ASSERT_TRUE(gDeclarator);
   auto func = [&](int) {
      R__READ_LOCKGUARD(ROOT::gCoreMutex);
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;
      R__READ_LOCKGUARD(ROOT::gCoreMutex);
      gInterpreter->Calc("f(10)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : middle" << std::endl;

      R__LOCKGUARD(gROOTMutex);
      gInterpreter->Calc("f(100000)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : end" << std::endl;

   };
   ROOT::TThreadExecutor pool;
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}

TEST(InterpreterLock, BalancedUserReadLock)
{
   ASSERT_TRUE(gDeclarator);
   auto func = [&](int) {
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;
      gInterpreter->Calc("R__READ_LOCKGUARD(ROOT::gCoreMutex); f(10)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : middle" << std::endl;

      R__LOCKGUARD(gROOTMutex);
      gInterpreter->Calc("R__READ_LOCKGUARD(ROOT::gCoreMutex); f(100000)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : end" << std::endl;

   };
   ROOT::TThreadExecutor pool;
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}

TEST(InterpreterLock, BalancedUserWriteLock)
{
   ASSERT_TRUE(gDeclarator);
   auto func = [&](int) {
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;
      gInterpreter->Calc("R__WRITE_LOCKGUARD(ROOT::gCoreMutex); f(10)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : middle" << std::endl;

      R__LOCKGUARD(gROOTMutex);
      gInterpreter->Calc("R__WRITE_LOCKGUARD(ROOT::gCoreMutex); f(100000)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : end" << std::endl;

   };
   ROOT::TThreadExecutor pool;
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}

TEST(InterpreterLock, UnBalancedUserReadLock)
{
   ASSERT_TRUE(gDeclarator);
   auto func = [&](int) {
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;
      gInterpreter->Calc("ROOT::gCoreMutex->ReadLock(); f(10)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : middle" << std::endl;

      {
            R__LOCKGUARD(gROOTMutex);
            gInterpreter->Calc("R__READ_LOCKGUARD(ROOT::gCoreMutex); f(100000)");
            if (gDebugOrder)
               std::cerr << std::this_thread::get_id() << " : end" << std::endl;
      }

      ROOT::gCoreMutex->ReadUnLock(nullptr);
   };
   ROOT::TThreadExecutor pool;
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}

TEST(InterpreterLock, UnBalancedUserWriteLock)
{
   ASSERT_TRUE(gDeclarator);
   auto func = [&](int) {
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : start" << std::endl;
      gInterpreter->Calc("ROOT::gCoreMutex->ReadLock(); f(10)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : middle" << std::endl;

      gInterpreter->Calc("ROOT::gCoreMutex->WriteLock(); f(100000)");
      if (gDebugOrder)
         std::cerr << std::this_thread::get_id() << " : end" << std::endl;

      ROOT::gCoreMutex->WriteUnLock(nullptr);
      ROOT::gCoreMutex->ReadUnLock(nullptr);
   };
   ROOT::TThreadExecutor pool;
   std::vector<int> id;
   for (auto i = 0u; i < 16; i++) id.push_back(i);
   pool.Foreach(func, id);
}
