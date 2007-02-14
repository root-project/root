// @(#)root/qt:$Name:  $:$Id: TQtTimer.h,v 1.5 2006/03/24 15:31:10 antcheva Exp $
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
#  include <qtimer.h>
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
  TQtTimer (QObject *parent=0, const char *name=0): QTimer(parent,name),fCounter(0){}
  
protected slots:
  virtual void AwakeRootEvent();
 
public:
  virtual ~TQtTimer(){}
  static TQtTimer *Create(QObject *parent=0, const char *name=0);
  static TQtTimer *QtTimer();
  ClassDef(TQtTimer,0) // QTimer to awake the ROOT event loop from Qt event loop
};
inline TQtTimer *TQtTimer::QtTimer(){ return fgQTimer; }


#endif
