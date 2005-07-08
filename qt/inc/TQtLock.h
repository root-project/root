// @(#)root/qt:$Name:  $:$Id: TQtLock.h,v 1.2 2005/07/08 00:26:35 fine Exp $
// Author: Giulio Eulisse  04/07/2005
#ifndef ROOT_TQtLock
#define ROOT_TQtLock

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtLock                                                              //
//                                                                      //
// Lock / unlock the critical section safely                            //
// To be replaced by TMutex class in future                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include <qapplication.h>
class TQtLock 
{
 public:
    TQtLock (void) { Lock();   }
   ~TQtLock (void) { Unlock(); }
    void Lock(Bool_t on=kTRUE) {
       if (qApp) {
          if (on)  qApp->lock();
          else     qApp->unlock();
       }
    }
    void Unlock(Bool_t on=kFALSE) { Lock(!on); }
};

#endif
