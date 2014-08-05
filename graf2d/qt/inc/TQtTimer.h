// @(#)root/qt:$Id$
// Author: Valery Fine  09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtTimer
#define ROOT_TQtTimer

#include "Rtypes.h"

#ifndef __CINT__
#  include <QTimer>
#else
  class QTimer;
#endif

//
// TQtTimer is a singelton QTimer to awake the ROOT event loop from Qt event loop
//

//___________________________________________________________________
class  TQtTimer : public QTimer  {
#ifndef __CINT__
     Q_OBJECT
#endif
private:
   void operator=(const TQtTimer &);
   TQtTimer(const TQtTimer &);
protected:
   static TQtTimer *fgQTimer;
   int fCounter;
   TQtTimer (QObject *mother=0): QTimer(mother),fCounter(0)
   {}

protected slots:
   virtual void AwakeRootEvent();

public:
   virtual ~TQtTimer(){}
   static TQtTimer *Create(QObject *parent=0);
   static TQtTimer *QtTimer();
   ClassDef(TQtTimer,0) // QTimer to awake the ROOT event loop from Qt event loop
};


#endif
