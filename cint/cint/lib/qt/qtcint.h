/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// qtcint.h

#define  __attribute__(x)

// Lie to CINT to make it go!
#ifdef __CINT__
typedef long long __int64;
typedef __int64 Q_UINT64;
typedef unsigned int uint;
static const int white = 0xff;
static const int black = 0x0;
typedef void* QTSFUNC;
class QPoint;
class QCursor {public:  const QPoint& pos(); };
typedef int Tag;
#define UINT_MAX qtcint_U_max
#define ULONG_MAX qtcint_UL_max

static unsigned int UINT_MAX;
static unsigned long ULONG_MAX;

// const bool FALSE=false;
// const bool TRUE=true;

#ifdef  Q_TYPENAME 
#undef  Q_TYPENAME 
#endif
#define  Q_TYPENAME 

#ifdef Q_EXPORT
#undef Q_EXPORT
#endif
#define Q_EXPORT

#ifdef Q_INLINE_TEMPLATES
# undef Q_INLINE_TEMPLATES
#endif
#define Q_INLINE_TEMPLATES

#define __declspec(fake) 
#ifdef dllimport
# undef dllimport
#endif
#define dllimport
#endif

// #include "limits.h"
#include <qt.h>
// #include <qgl.h>

#include "qtclasses.h"
#include "qtglobals.h"
#include "qtfunctions.h"
