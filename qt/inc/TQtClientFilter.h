// @(#)root/qt:$Name:  $:$Id: TQtClientFilter.h,v 1.7 2006/03/24 15:31:10 antcheva Exp $
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
#  if (QT_VERSION > 0x039999)
// Added by qt3to4:
#    include <QEvent>
#    include <QMouseEvent>
#    include <q3ptrqueue.h>
#    include <q3ptrlist.h>
#    include <q3intdict.h>
#  else 
#    include <qptrqueue.h>
#    include <qptrlist.h>
#    include <qintdict.h>
#  endif  /* QT_VERSION */
#  include <qapplication.h>
#else
   class QObject;
#  if (QT_VERSION > 0x039999)
     class Q3PtrList<TQtClientWidget>;
#  else /* QT_VERSION */
     class QPtrList<TQtClientWidget>;
#  endif /* QT_VERSION */
#endif  /* CINT */ 

#include "TQtClientWidget.h"

//
//  TQtClientFilter  is Qt "eventFilter" to map Qt event to ROOT event
//

class TQtNextEventMessage;
class TQtEventQueue;
// class TQtClientWidget;
class TQtPointerGrabber;

class TQtClientFilter : public QObject {
#ifndef __CINT__
   Q_OBJECT
#endif
   friend class TGQt;
   friend class TQtClientWidget;
private:
         void operator=(const TQtClientFilter &){}
#if !defined(_MSC_VER)  || _MSC_VER >= 1310
         void operator=(const TQtClientFilter &) const {}
#endif
			TQtClientFilter(const TQtClientFilter &) : QObject() {}
protected:
   TQtEventQueue             *fRootEventQueue;
   TQtNextEventMessage       *fNotifyClient;
#if (QT_VERSION > 0x039999)
   Q3PtrList<TQtClientWidget>  fButtonGrabList;
#else /* QT_VERSION */
   QPtrList<TQtClientWidget>  fButtonGrabList;
#endif /* QT_VERSION */
   static TQtClientWidget    *fgPointerGrabber;
   static TQtClientWidget    *fgButtonGrabber;
   static TQtClientWidget    *fgActiveGrabber;
   TQtClientWidget           *fKeyGrabber;
   UInt_t                     fInputEventMask;
   static UInt_t              fgGrabPointerEventMask;
   static Bool_t              fgGrabPointerOwner;
   static QCursor            *fgGrabPointerCursor;
   // static Bool_t              fIsGrabbing;
   static TQtPointerGrabber  *fgGrabber;
protected:
   bool eventFilter( QObject *o, QEvent *e );
   void AddKeyEvent( const QKeyEvent &event, TQtClientWidget *widget);
   TQtEventQueue *Queue();
   void SetKeyGrabber(TQtClientWidget *grabber)     { fKeyGrabber = grabber;}
   void UnSetKeyGrabber(TQtClientWidget *grabber)   { if (fKeyGrabber == grabber) fKeyGrabber = 0; }
   void RestoreLostGrabbing(Event_t &event);
   static Bool_t IsGrabSelected(UInt_t selectEventMask);
   static Bool_t SelectGrab(Event_t &event, UInt_t selectEventMask, QMouseEvent &me);
public:
   TQtClientFilter():fRootEventQueue(0),fNotifyClient(0),fKeyGrabber(0){;}
   virtual ~TQtClientFilter();
   static TQtClientWidget    *GetPointerGrabber();
   static TQtClientWidget    *GetButtonGrabber();
   static void SetButtonGrabber(TQtClientWidget *grabber);
   static void GrabPointer(TQtClientWidget *grabber, UInt_t evmask, Window_t confine,
                                    QCursor *cursor, Bool_t grab = kTRUE,
                                    Bool_t owner_events = kTRUE);
   static TQtPointerGrabber *PointerGrabber() { return fgGrabber; }
public slots:
   void AppendButtonGrab (TQtClientWidget *);
   void RemoveButtonGrab (QObject *);
#ifndef Q_MOC_RUN
//MOC_SKIP_BEGIN
   ClassDef(TQtClientFilter,0) // Map Qt and ROOT event
//MOC_SKIP_END
#endif
};

//
//  TQtClientFilter is a Qt "eventFilter" to map Qt event to ROOT event
//
class QWidget;
class QCursor;

class TQtPointerGrabber {
private:
   UInt_t           fGrabPointerEventMask;
   UInt_t           fInputPointerEventMask;
   Bool_t           fGrabPointerOwner;
   QCursor         *fGrabPointerCursor;
   TQtClientWidget *fPointerGrabber;
   QWidget         *fPointerConfine;
   Bool_t           fIsActive;        // Do we active grabbing with WM
public:
   TQtPointerGrabber(TQtClientWidget *grabber, UInt_t evGrabMask, UInt_t evInputMask,
                                    QCursor *cursor, Bool_t grab = kTRUE,
                                    Bool_t owner_events = kTRUE, QWidget *confine=0);
   ~TQtPointerGrabber();
   void   ActivateGrabbing(bool on=TRUE);
   void   DisactivateGrabbing(){ ActivateGrabbing(kFALSE); }
   Bool_t IsGrabSelected(UInt_t selectEventMask) const;
   Bool_t IsGrabbing(TQtClientWidget *grabbed) const { return (grabbed == fPointerGrabber); }
   void   SetGrabPointer( TQtClientWidget *grabber, UInt_t evGrabMask, UInt_t evInputMask
                       , QCursor *cursor, Bool_t grab = kTRUE
                       , Bool_t owner_events = kTRUE, QWidget *confine=0);
   bool   SelectGrab(Event_t &event, UInt_t selectEventMask,QMouseEvent &mouse);
};

//______________________________________________________________________________
inline TQtClientWidget *TQtClientFilter::GetPointerGrabber()
{  return fgPointerGrabber;                                       }

//______________________________________________________________________________
inline TQtClientWidget *TQtClientFilter::GetButtonGrabber() 
{  return fgButtonGrabber;                                              }

//______________________________________________________________________________
inline void TQtClientFilter::SetButtonGrabber(TQtClientWidget *grabber)
{  fgButtonGrabber = grabber;                                           }
   
//______________________________________________________________________________
inline   void TQtClientFilter::AppendButtonGrab(TQtClientWidget *widget)
{  fButtonGrabList.append(widget);}
//______________________________________________________________________________
inline   void TQtClientFilter::RemoveButtonGrab(QObject *widget)
{ 
   TQtClientWidget *wid = (TQtClientWidget *)widget;
   if ((fgButtonGrabber == wid) && fgGrabber) fgGrabber->DisactivateGrabbing();
   fButtonGrabList.remove(wid);
}

//______________________________________________________________________________
inline   TQtEventQueue *TQtClientFilter::Queue() {
      TQtEventQueue *save = fRootEventQueue;
      // fprintf(stderr," Queue %d \n", save ? save->count():-1);
      return save;
   }

#endif

