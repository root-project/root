// @(#)root/qt:$Name:  $:$Id: TQtTimer.cxx,v 1.9 2004/08/13 06:05:17 brun Exp $
// Author: Valery Fine  09/08/2004
/****************************************************************************
** $Id: TQtTimer.cxx,v 1.2 2004/08/10 16:36:10 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

#include <qapplication.h>
#include "TQtTimer.h"
#include "TSystem.h"

////////////////////////////////////////////////////////////////////////////////
//
//  TQtClientWidget is QWidget with QPixmap double buffer
//  It designed to back the ROOT TCanvasImp class interface  and it can be used
//  as a regular Qt Widget to create Qt-based GUI with embedded TCanvas objects
//
//  This widget can be used to build a custom GUI interfaces with  Qt Designer
//
////////////////////////////////////////////////////////////////////////////////

TQtTimer *TQtTimer::fgQTimer=0;

//______________________________________________________________________________
void TQtTimer::AwakeRootEvent(){
   gSystem->DispatchOneEvent(kTRUE);
}

//______________________________________________________________________________
TQtTimer * TQtTimer::Create(QObject *parent, const char *name)
{
   qApp->lock();
   if (!fgQTimer) {
      fgQTimer = new  TQtTimer(parent,name);
      qApp->unlock();
      connect(fgQTimer,SIGNAL(timeout()),fgQTimer,SLOT(AwakeRootEvent()) );
   } else {
      qApp->unlock();
   }
   return fgQTimer;
}
