#ifndef ROOT_TQtLockGuard
#define ROOT_TQtLockGuard

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtLockGuard - Qt-based implementation of ROOT TLockGuard class      //
//                                                                      //
// This class provides mutex resource management in a guaranteed and    //
// exception safe way. Use like this:                                   //
// {                                                                    //
//    TQtLockGuard guard(mutex);                                        //
//    ... // do something                                               //
// }                                                                    //
// when guard goes out of scope the mutex is unlocked in the TLockGuard //
// destructor. The exception mechanism takes care of calling the dtors  //
// of local objects so it is exception safe.                            //
//                                                                      //
//  The macro  Q__LOCKGUARD2(QMutex *mutex)                             //
//             creates first creates the QMutex object and then creates //
//             TQtLockGuard as above if the QT_THREAD_SUPPORT           //
//             was provided                                             //
//                                                                      //
//  NOTE: This class may be removed as soon as                          //
//  ----  ROOT TThreadImp class QThread-based implemantion              //
//        is adopted by ROOT team (one needs to convince Fons )         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "qmutex.h"

class TQtLockGuard {

private:
   QMutex *fMutex;

public:
   TQtLockGuard(QMutex *mutex);
   ~TQtLockGuard();
};

//____________________________________________________________________
inline TQtLockGuard::TQtLockGuard(QMutex *mutex)
{ fMutex = mutex; if (fMutex) fMutex->lock(); }

//____________________________________________________________________
inline TQtLockGuard::~TQtLockGuard()
{ if (fMutex) fMutex->unlock(); }


// Zero overhead macros in case not compiled with thread support
#ifdef QT_THREAD_SUPPORT

#define Q__LOCKGUARD(mutex) TQtLockGuard QR__guard(mutex)

#define Q__LOCKGUARD2(mutex) {              \
   if (qApp && !mutex) {                    \
      qApp->lock();                         \
      if (!mutex) mutex = new QMutex(true); \
      qApp->unlock();                       \
   }                                        \
   Q__LOCKGUARD(mutex);                     \
 }
#else
#define Q__LOCKGUARD(mutex)  if (mutex) { }
#define Q__LOCKGUARD2(mutex) if (mutex) { }
#endif

#endif
