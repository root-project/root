// Usage:
// root [0] .L stressThreadPool.C++
// root [1] stressThreadPool(5, true)
// where 5 is a number of Threads in the pool
// there will be then nThreads * 10 tasks pushed to the test

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
// Don't set it less than 1, otherwise autotest won't be able to detect whether tests were successful or not
const size_t g_sleeptime = 2; // in secs.
const size_t g_multTasks = 10;
//=============================================================================

enum EProc {start, clean};

class TTestTask: public TThreadPoolTaskImp<TTestTask, EProc> {
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
ostream &operator<< (ostream &_stream, const TTestTask &_task)
{
   _stream << _task.threadID();
   return _stream;
}

//=============================================================================
void stressThreadPool(size_t _numThreads = 5, bool _needDbg = false)
{
   size_t numTasks(_numThreads * g_multTasks);
   TThreadPool<TTestTask, EProc> threadPool(_numThreads, _needDbg);
   vector <TTestTask> tasksList(numTasks);
   // Pushing 4 * numTasks task in the pool
   // We want to dain the task queue before pushing a next bunch of tasks (just to show you a Drain method ;) )
   for (size_t j = 0; j < 4; ++j )
   {
      cout << "+++++++++ Starting iteration #" << j << " ++++++++++++"<< endl;
      for (size_t i = 0; i < numTasks; ++i) {
         threadPool.PushTask(tasksList[i], start);
      }

      cout << "\n ****** Drain the tasks queue ******" << endl;
      threadPool.Drain();
   }
   cout << "\n Stopping..." << endl;
   threadPool.Stop(true);

   //    ostream_iterator<TTestTask> out_it( cout, "\n" );
   //    copy( tasksList.begin(), tasksList.end(),
   //          out_it );

   typedef map<unsigned long, size_t> counter_t;
   counter_t counter;
   {
      vector <TTestTask>::const_iterator iter = tasksList.begin();
      vector <TTestTask>::const_iterator iter_end = tasksList.end();
      for (; iter != iter_end; ++iter) {
         counter_t::iterator found = counter.find(iter->threadID());
         if (found == counter.end())
            counter.insert(counter_t::value_type(iter->threadID(), 1));
         else {
            found->second = found->second + 1;
         }
      }
   }

   cout << "\n************* RESULT ****************" << endl;

   counter_t::const_iterator iter = counter.begin();
   counter_t::const_iterator iter_end = counter.end();
   bool testOK = true;
   for (; iter != iter_end; ++iter) {
      cout << "Thread " << iter->first << " was used " << iter->second << " times\n";
      // each thread suppose to be used equal amount of time,
      // exactly (g_numTasks/g_numThreads) times
      if (iter->second != g_multTasks)
         testOK = false;
   }

   cout << "ThreadPool: the simple test status: " << (testOK ? "OK" : "Failed") << endl;
}
