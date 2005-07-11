// @(#)root/qt:$Name:  $:$Id: TQtLock.h,v 1.2 2005/07/08 14:17:20 brun Exp $
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

class TQtLock {

public:
   TQtLock (void) { Lock();   }
   ~TQtLock (void) { UnLock(); }
   void Lock() { if (qApp) qApp->lock(); }
   void UnLock() { if (qApp) qApp->unlock(); }
};

#endif
