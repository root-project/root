// @(#)root/qt:$Id$
// Author: Valery Fine      18/01/2007

/****************************************************************************
** $Id$
**
** Copyright (C) 2007 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
**
*****************************************************************************/
//________________________________________________________________________
//
// The TQRootSlot singleton class introduces the global SLOT to invoke
// the  ROOT command line from the GUI signals
// Optionally one can execute TApplication::Terminate method directly
//
// It provides a Qt slot to attach the the CINT C++ interpreter
// to any Qt signal
// To execute any C++ statement from the GUI one should connect
// one's Qt signal with the Qt slot of the global instance of this class
//________________________________________________________________________

#ifndef ROOT_TQRootSlot
#define ROOT_TQRootSlot

#ifndef __CINT__
#include <qobject.h>
#else
class QObject;
#ifndef Q_OBJECT
#define Q_OBJECT
#endif
#define slots
#endif

class QString;

class TQtRootSlot : public QObject {
   Q_OBJECT
private:
   TQtRootSlot (const TQtRootSlot &);
   void operator=(const TQtRootSlot &);
protected:
   static TQtRootSlot *fgTQtRootSlot;
   TQtRootSlot () {}
public:
   static TQtRootSlot *CintSlot();
   virtual ~TQtRootSlot() {}

public slots:
   void ProcessLine(const char *);
   void ProcessLine(const QString &);
   void EndOfLine();
   void Terminate(int status) const;
   void Terminate()           const;
   void TerminateAndQuit()    const;

#ifndef __CINT__
signals:
   void Error(int error);
#endif
};

bool QConnectCint(const QObject *sender, const char *signal);
bool QConnectTerminate( const QObject *sender, const char *signal);
#endif
