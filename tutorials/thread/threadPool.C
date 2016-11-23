/// \file
/// \ingroup tutorial_thread
///
/// Usage:
///
/// ~~~{.cpp}
/// root [0] .L threadPool.C++
/// root [1] threadPool(10)  10 = numThreads
/// ~~~
///
/// \macro_code
///
/// \author Victor Perevovchikov

// STD
#include <iostream>
#include <iterator>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif
// ThreadPool
#include "TThreadPool.h"
// ROOT
#include "TThread.h"

//=============================================================================
using namespace std;
//=============================================================================
const size_t g_sleeptime = 1; // in secs.
const size_t g_multTasks = 50;
//=============================================================================

// define a custom parameters type for task objects
enum EProc {start, clean};

// a class defining task objects
class TTestTask: public TThreadPoolTaskImp<TTestTask, EProc>
{
public:
   bool runTask(EProc /*_param*/) {
      m_tid = TThread::SelfId();
      TThread::Sleep(g_sleeptime, 0L);
      return true;
   }
   unsigned long threadID() const {
      return m_tid;
   }

private:
   unsigned long m_tid;
};

//=============================================================================
void threadPool(size_t _numThreads = 10, bool _needDbg = false)
{
   cout << "ThreadPool: starting..." << endl;
   // number of tasks to process
   size_t numTasks(_numThreads * g_multTasks);

   // create a thread pool object
   // _numThreads - a number of threads in the pool
   // _needDbg - defines whether to show debug messages
   TThreadPool<TTestTask, EProc> threadPool(_numThreads, _needDbg);

   // create a container of tasks
   vector <TTestTask> tasksList(numTasks);

   cout << "ThreadPool: getting tasks..." << endl;
   cout << "ThreadPool: processing tasks..." << endl;
   // push tasks to the ThreadPool
   // tasks can be also pushed asynchronously
   for (size_t i = 0; i < numTasks; ++i) {
      threadPool.PushTask(tasksList[i], start);
   }

   // Stop thread pool.
   // The parameter "true" requests the calling thread to wait,
   // until the thread pool task queue is drained.
   threadPool.Stop(true);
   cout << "ThreadPool: done" << endl;
}

