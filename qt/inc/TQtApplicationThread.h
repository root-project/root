// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtApplicationThread.h,v 1.4 2003/11/18 18:41:55 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQTAPPLICATIONTHREAD
#define ROOT_TQTAPPLICATIONTHREAD
#include "TQtRootThread.h"
#include "qobject.h"

class TQtEvent;

class TQtApplicationThread : public QObject, public TQtRootThread {
  Q_OBJECT
  protected:
    int    fArgc;
    char **fArgv;
    void  *fOption;
    int    fNumOpt;

  protected:
    friend class TQtThreadDispatcher;
    virtual void Run();
    virtual bool eventCB(TQtEvent *event);
    TQtApplicationThread() : TQtRootThread(){;}

  public:
    TQtApplicationThread(int argc, char **argv);
    virtual ~TQtApplicationThread();
public slots:
   void AboutToQuit ();
};

#endif
