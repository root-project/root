// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQClientFilter
#define ROOT_TQClientFilter

#include "GuiTypes.h"

#include <qobject.h>
#include <qptrqueue.h>
#include <qptrlist.h>
#include <qintdict.h>
#include <qapplication.h>
#include "TQtClientWidget.h"

class TQtNextEventMessage;
class TQtEventQueue;
// class TQtClientWidget;

class TQtClientFilter : public QObject {
   Q_OBJECT
   friend class TGQt;
protected:
   TQtEventQueue             *fRootEventQueue;
   TQtNextEventMessage       *fNotifyClient;
   QPtrList<TQtClientWidget>  fButtonGrabList;
   TQtClientWidget           *fPointerGrabber;
   Bool_t                     fIsGrabbing;

protected:
   bool eventFilter( QObject *o, QEvent *e );
   TQtEventQueue *Queue();
   TQtClientWidget    *GetPointerGrabber() const { return fPointerGrabber;}
   void SetPointerGrabber(TQtClientWidget *grabber) {fPointerGrabber = grabber;}
public:
   TQtClientFilter():fRootEventQueue(0),fNotifyClient(0),fPointerGrabber(0),fIsGrabbing(kFALSE ){;}
   virtual ~TQtClientFilter();
public slots:
   void AppendButtonGrab (TQtClientWidget *);
   void AppendPointerGrab(TQtClientWidget *);
   void RemoveButtonGrab (QObject *);
   void RemovePointerGrab(QObject *);
};

//______________________________________________________________________________
inline   void TQtClientFilter::AppendButtonGrab(TQtClientWidget *widget)
{  fButtonGrabList.append(widget);}
//______________________________________________________________________________
inline   void TQtClientFilter::RemoveButtonGrab(QObject *widget)
{ fButtonGrabList.remove((TQtClientWidget *)widget);}

//______________________________________________________________________________
inline   TQtEventQueue *TQtClientFilter::Queue() {
#ifdef R__QTGUITHREAD
      qApp->lock();
      TQtEventQueue *save = fRootEventQueue;
      fRootEventQueue = 0;
      qApp->unlock();
#else
      TQtEventQueue *save = fRootEventQueue;
#endif
      // fprintf(stderr," Queue %d \n", save ? save->count():-1);
      return save;
   }

#endif

