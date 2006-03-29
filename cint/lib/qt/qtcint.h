// qtcint.h

#define  __attribute__(x)  

// Lie to CINT to make it go!
#ifdef __CINT__
static const int white = 0xff;
typedef void* QTSFUNC;
class QPoint;
class QCursor {public:  const QPoint& pos(); };
typedef int Tag;
#define Q_TYPENAME 
#endif


#include <qt.h>

#include "qtclasses.h"
#include "qtglobals.h"
#include "qtfunctions.h"
