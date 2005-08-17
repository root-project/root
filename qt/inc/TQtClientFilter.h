// @(#)root/qt:$Name:  $:$Id: TQtClientFilter.h,v 1.25 2005/07/10 00:34:57 fine Exp $
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
#include "Rtypes.h"

#ifndef __CINT__
#  include <qobject.h>
#if (QT_VERSION > 0x039999)
// Added by qt3to4:
#  include <QEvent>
#endif 
#  include <qptrqueue.h>
#  include <qptrlist.h>
#  include <qintdict.h>
#  include <qapplication.h>
#else
  class QObject;
  class QPtrList<TQtClientWidget>;
#endif

#include "TQtClientWidget.h"

//
//  TQtClientFilter  is Qt "eventFilter" to map Qt event to ROOT event
//

class TQtNextEventMessage;
class TQtEventQueue;
// class TQtClientWidget;

class TQtClientFilter : public QObject {
#ifndef __CINT__
   Q_OBJECT
#endif
   friend class TGQt;
   friend class TQtClientWidget;
private:
         void operator=(const TQtClientFilter &){}
         void operator=(const TQtClientFilter &) const {}
         TQtClientFilter(const TQtClientFilter &) : QObject() {}
protected:
   TQtEventQueue             *fRootEventQueue;
   TQtNextEventMessage       *fNotifyClient;
   QPtrList<TQtClientWidget>  fButtonGrabList;
   TQtClientWidget           *fPointerGrabber;
   TQtClientWidget           *fKeyGrabber;
   Bool_t                     fIsGrabbing;

protected:
   bool eventFilter( QObject *o, QEvent *e );
   TQtEventQueue *Queue();
   TQtClientWidget    *GetPointerGrabber() const    { return fPointerGrabber;}
   void SetPointerGrabber(TQtClientWidget *grabber) { fPointerGrabber = grabber;}
   void SetKeyGrabber(TQtClientWidget *grabber)     { fKeyGrabber = grabber;}
   void UnSetKeyGrabber(TQtClientWidget *grabber)   { if (fKeyGrabber == grabber) fKeyGrabber = 0; }
public:
   TQtClientFilter():fRootEventQueue(0),fNotifyClient(0),fPointerGrabber(0),fKeyGrabber(0),fIsGrabbing(kFALSE ){;}
   virtual ~TQtClientFilter();
public slots:
   void AppendButtonGrab (TQtClientWidget *);
   void AppendPointerGrab(TQtClientWidget *);
   void RemoveButtonGrab (QObject *);
   void RemovePointerGrab(QObject *);
//MOC_SKIP_BEGIN
   ClassDef(TQtClientFilter,0) // Map Qt and ROOT event
//MOC_SKIP_END
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

