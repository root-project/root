// qtcint.h

#define  __attribute__(x)

// Lie to CINT to make it go!
#ifdef __CINT__
static const int white = 0xff;
static const int black = 0x0;
typedef void* QTSFUNC;
class QPoint;
class QCursor {public:  const QPoint& pos(); };
typedef int Tag;
#define Q_TYPENAME 
#endif


#include <qt.h>
#include <qgl.h>

#include "qtclasses.h"
#include "qtglobals.h"
#include "qtfunctions.h"
