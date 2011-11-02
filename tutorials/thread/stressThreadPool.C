// Usage:
// root [0] .L stressThreadPool.C++
// root [1] stressThreadPool(10)   10 = numThreads

// STD
#include <iostream>
#include <iterator>
#include <vector>
#include <unistd.h>
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

enum EProc {start, clean};

class TTestTask: public TThreadPoolTaskImp<TTestTask, EProc> {
public:
   bool runTask(EProc /*_param*/) {
      m_tid = TThread::SelfId();
      sleep(g_sleeptime);
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
void stressThreadPool(size_t _numThreads, bool _needDbg = false)
{
   size_t numTasks(_numThreads * g_multTasks);
   TThreadPool<TTestTask, EProc> threadPool(_numThreads, _needDbg);
   vector <TTestTask> tasksList(numTasks);
   for (size_t i = 0; i < numTasks; ++i) {
      threadPool.PushTask(tasksList[i], start);
   }
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
   bool testOK=true;
   for (; iter != iter_end; ++iter) {
      cout << "Thread " << iter->first << " was used " << iter->second << " times\n";
      // each thread suppose to be used equal amount of time,
      // exactly (g_numTasks/g_numThreads) times
      if (iter->second != g_multTasks) 
         testOK = false;
   }

   cout << "ThreadPool: simple test - "<< (testOK? "OK": "Failed") << endl;
}
