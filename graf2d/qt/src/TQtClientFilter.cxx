// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQtClientFilter.h"
#include "TQtRConfig.h"

#include "TQtClientWidget.h"
#include "TGQt.h"
#include "TQtEventQueue.h"
#include "TQUserEvent.h"
#include "TQtLock.h"

#include "TSystem.h"
#include "TStopwatch.h"
#include "qevent.h"
#include <qdatetime.h>
#include <qcursor.h>
#include <qtextcodec.h>

#include <QWheelEvent>
#include <QByteArray>
#include <QFocusEvent>
#include <QPaintEvent>
#include <QCloseEvent>
#include <QMoveEvent>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QDebug>

#include <cassert>
#include "KeySymbols.h"
#define QTCLOSE_DESTROY_RESPOND 1

//______________________________________________________________________________
//
// QtClientFilter provides QOject event filter to map Qt and  ROOT events 
//   (see: http://doc.trolltech.com/4.3/qobject.html#installEventFilter )
//______________________________________________________________________________

ClassImp(TQtClientFilter)

TQtClientWidget *TQtClientFilter::fgPointerGrabber=0;
TQtClientWidget *TQtClientFilter::fgButtonGrabber =0;
TQtClientWidget *TQtClientFilter::fgActiveGrabber =0;

UInt_t           TQtClientFilter::fgGrabPointerEventMask = 0;
Bool_t           TQtClientFilter::fgGrabPointerOwner     = kFALSE;
QCursor         *TQtClientFilter::fgGrabPointerCursor    = 0;

TQtPointerGrabber *TQtClientFilter::fgGrabber = 0;
//______________________________________________________________________________________
static inline UInt_t  MapModifierState(Qt::KeyboardModifiers qState)
{
   UInt_t state = 0;
   if ( qState & Qt::ShiftModifier   ) state |= kKeyShiftMask;
   if ( qState & Qt::ControlModifier ) state |= kKeyControlMask;
   if ( qState & Qt::AltModifier     ) state |= kKeyMod1Mask;
   if ( qState & Qt::MetaModifier    ) state |= kKeyLockMask;
   return state;
}

//______________________________________________________________________________________
static inline UInt_t  MapButtonState(Qt::MouseButtons qState)
{
   UInt_t state = 0;
   if ( qState & Qt::RightButton     ) state |= kButton3Mask;
   if ( qState & Qt::MidButton       ) state |= kButton2Mask;
   if ( qState & Qt::LeftButton      ) state |= kButton1Mask;
   return state;
}

//______________________________________________________________________________________
static inline void MapEvent( QWheelEvent &qev, Event_t &ev)
{
    // Map Qt QWheelEvent (like MouseEvent) to ROOT kButton4 and kButton5 events
   ev.fX      = qev.x();
   ev.fY      = qev.y();
   ev.fXRoot  = qev.globalX();
   ev.fYRoot  = qev.globalY();

   if ( qev.delta() > 0 ) {
      ev.fCode   = kButton4;
      ev.fState |= kButton4Mask;
   } else {
      ev.fCode   = kButton5;
      ev.fState |= kButton5Mask;
   }
   ev.fState |= MapModifierState(qev.modifiers());
   ev.fState |= MapButtonState(qev.buttons());
   ev.fUser[0] = TGQt::rootwid(TGQt::wid(ev.fWindow)->childAt(ev.fX,ev.fY)) ;
   qev.ignore(); // propage the mouse event further
   // fprintf(stderr, "QEvent::Wheel %p %d child=%p\n",ev.fWindow, ev.fCode, ev.fUser[0]);
}
//______________________________________________________________________________________
Bool_t TQtClientFilter::IsGrabSelected(UInt_t selectEventMask)
{
   // return the selection by "grabButton" / "grabPointer"
   return fgGrabber ? fgGrabber->IsGrabSelected(selectEventMask) : kFALSE;
}
//______________________________________________________________________________________
static inline void MapEvent(QMouseEvent &qev, Event_t &ev)
{
   ev.fX      = qev.x();
   ev.fY      = qev.y();
   ev.fXRoot  = qev.globalX();
   ev.fYRoot  = qev.globalY();
   switch ( qev.button() )  {
      case Qt::LeftButton:
         // Set if the left button is pressed, or if this event refers to the left button.
         //(The left button may be the right button on left-handed mice.)
         ev.fCode  = kButton1;
         break;
      case Qt::MidButton:
         // the middle button.
         ev.fCode  = kButton2;
         break;
      case Qt::RightButton:
         //  the right button
         ev.fCode  = kButton3;
         break;
      default:
         if (qev.type() != QEvent::MouseMove) {
          qDebug() << "MapEvent Error ***. Unexpected event." << &qev;
          return;
         }
         break;
   };
   ev.fState |= MapModifierState(qev.modifiers());
   ev.fState |= MapButtonState(qev.buttons());   
   if (ev.fCode)
      ev.fUser[0] = TGQt::rootwid(TGQt::wid(ev.fWindow)->childAt(ev.fX,ev.fY)) ;

   qev.ignore(); // propagate the mouse event further
}


//---- Key symbol mapping
struct KeyQSymbolMap_t {
   Qt::Key fQKeySym;
   EKeySym fKeySym;
};

//---- Mapping table of all non-trivial mappings (the ASCII keys map
//---- one to one so are not included)

static KeyQSymbolMap_t gKeyQMap[] = {
   {Qt::Key_Escape,    kKey_Escape},
   {Qt::Key_Tab,       kKey_Tab},
   {Qt::Key_Backtab,   kKey_Backtab},
#if QT_VERSION < 0x40000
   {Qt::Key_BackSpace, kKey_Backspace},
#else /* QT_VERSION */
   {Qt::Key_Backspace, kKey_Backspace},
#endif /* QT_VERSION */
   {Qt::Key_Return,    kKey_Return},
   {Qt::Key_Insert,    kKey_Insert},
   {Qt::Key_Delete,    kKey_Delete},
   {Qt::Key_Pause,     kKey_Pause},
   {Qt::Key_Print,     kKey_Print},
   {Qt::Key_SysReq,    kKey_SysReq},
   {Qt::Key_Home,      kKey_Home},       // cursor movement
   {Qt::Key_End,       kKey_End},
   {Qt::Key_Left,      kKey_Left},
   {Qt::Key_Up,        kKey_Up},
   {Qt::Key_Right,     kKey_Right},
   {Qt::Key_Down,      kKey_Down},
#if QT_VERSION < 0x40000
   {Qt::Key_Prior,     kKey_Prior},
   {Qt::Key_Next,      kKey_Next},
#else /* QT_VERSION */
   {Qt::Key_PageUp,     kKey_Prior},
   {Qt::Key_PageDown,      kKey_Next},
#endif /* QT_VERSION */
   {Qt::Key_Shift,     kKey_Shift},
   {Qt::Key_Control,   kKey_Control},
   {Qt::Key_Meta,      kKey_Meta},
   {Qt::Key_Alt,       kKey_Alt},
   {Qt::Key_CapsLock,  kKey_CapsLock},
   {Qt::Key_NumLock ,  kKey_NumLock},
   {Qt::Key_ScrollLock, kKey_ScrollLock},
   {Qt::Key_Space,     kKey_Space},  // numeric keypad
   {Qt::Key_Tab,       kKey_Tab},
   {Qt::Key_Enter,     kKey_Enter},
   {Qt::Key_Equal,     kKey_Equal},
   {Qt::Key_Asterisk,  kKey_Asterisk},
   {Qt::Key_Plus,      kKey_Plus},
   {Qt::Key_Comma,     kKey_Comma},
   {Qt::Key_Minus,     kKey_Minus},
   {Qt::Key_Period,    kKey_Period},
   {Qt::Key_Slash,     kKey_Slash},
   {Qt::Key(0), (EKeySym) 0}
};
//______________________________________________________________________________________
static inline UInt_t MapKeySym(const QKeyEvent &qev)
{
   UInt_t text = 0;;
   Qt::Key key = Qt::Key(qev.key());
   for (int i = 0; gKeyQMap[i].fKeySym; i++) {	// any other keys
      if (key ==  gKeyQMap[i].fQKeySym) {
         return   UInt_t(gKeyQMap[i].fKeySym);
      }
   }
#if 0
   QByteArray oar = gQt->GetTextDecoder()->fromUnicode(qev.text());
   const char *r = oar.constData();
   qstrlcpy((char *)&text, (const char *)r,1);
   return text;
#else
   text = UInt_t(qev.text().toAscii().data()[0]);
#ifdef QT_STILL_HAS_BUG   
   // Regenerate the ascii code (Qt bug I guess)
   if  (qev.modifiers() != Qt::NoModifier)  {
      if (  ( Qt::Key_A <= key && key <= Qt::Key_Z)  ) 
            text =  (( qev.state() & Qt::ShiftModifier )?  'A' : 'a') + (key - Qt::Key_A) ;
      else if (  ( Qt::Key_0 <= key && key <= Qt::Key_9)  ) 
            text =    '0'  + (key - Qt::Key_0);
   }
#endif
    // we have to accomodate the new ROOT GUI logic.
    // the information about the "ctrl" key should be provided TWICE nowadays 12.04.2005 vf.
//   }
   return text;
#endif
}
//______________________________________________________________________________________
static inline void MapEvent(const QKeyEvent  &qev, Event_t &ev)
{
   ev.fType  = qev.type() == QEvent::KeyPress ?  kGKeyPress : kKeyRelease;
   ev.fCode  = MapKeySym(qev);
   ev.fState = MapModifierState(qev.modifiers());
   ev.fCount = qev.count();
   ev.fUser[0] = TGQt::rootwid(TGQt::wid(ev.fWindow)->childAt(ev.fX,ev.fY)) ;
   // qev.accept();
}
//______________________________________________________________________________________
static inline void MapEvent(const QMoveEvent &qev, Event_t &ev)
{
   ev.fX = qev.pos().x();
   ev.fY = qev.pos().y();
}
//______________________________________________________________________________________
static inline void MapEvent(const QResizeEvent &qev, Event_t &ev)
{
   ev.fWidth  = qev.size().width();	   // width and
   ev.fHeight = qev.size().height();	// height of exposed area
}
//______________________________________________________________________________________
static inline void MapEvent(const QPaintEvent &qev, Event_t &ev)
{
   ev.fX = qev.rect().x();
   ev.fY = qev.rect().y();
   ev.fWidth  = qev.rect().width();	  // width and
   ev.fHeight = qev.rect().height();  // height of exposed area
   ev.fCount  = 0;
   //         ev.fCount = xev.expose.count;	// number of expose events still to come
}
//______________________________________________________________________________________
static inline void MapEvent(const TQUserEvent &qev, Event_t &ev)
{
   qev.getData(ev);
}
//______________________________________________________________________________
TQtClientFilter::~TQtClientFilter()
{
   TQtLock lock;  // critical section
   if (fRootEventQueue) {
      delete fRootEventQueue;
      fRootEventQueue = 0;
   }
}
//______________________________________________________________________________
static inline bool IsMouseCursorInside()
{
   // Detect whether the mouse cursor is inside of any application widget
   bool inside = false;
   QPoint absPostion = QCursor::pos();
   QWidget *currentW = QApplication::widgetAt(absPostion);
   // fprintf(stderr,"  -0-  IsMouseCursorInside frame=%p, grabber=%p, "
   //           , currentW,QWidget::mouseGrabber());
   if (currentW) {
      QRect widgetRect = currentW->geometry();
      widgetRect.moveTopLeft(currentW->mapToGlobal(QPoint(0,0)));
      inside = widgetRect.contains(absPostion);
      // fprintf(stderr," widget x=%d, y=%d, cursor x=%d, y=%d"
      //      ,widgetRect.x(),widgetRect.y(),absPostion.x(),absPostion.y());
   }
   // fprintf(stderr," inside = %d \n",inside);
   return inside;
}

#ifdef QTCLOSE_DESTROY_RESPOND
//______________________________________________________________________________
static void SendCloseMessage(Event_t &closeEvent)
{
   // Send close message to window provided via closeEvent.
   // This method should be called just the user closes the window via WM
   // See: TGMainFrame::SendCloseMessage() 

   if (closeEvent.fType != kDestroyNotify) return;
   Event_t evt = closeEvent;

   evt.fType   = kClientMessage;
   evt.fFormat = 32;
   evt.fHandle = gWM_DELETE_WINDOW;

   // evt.fWindow  = GetId();
   evt.fUser[0] = (Long_t) gWM_DELETE_WINDOW;
   evt.fUser[1] = 0;
   evt.fUser[2] = 0;
   evt.fUser[3] = 0;
   evt.fUser[4] = 0;
   // fprintf(stderr,"SendCloseMessage Closing id=%p widget=%p\n"
   //      , (void *)evt.fWindow, (TQtClientWidget*)TGQt::wid(evt.fWindow));
   
   gVirtualX->SendEvent(evt.fWindow, &evt);
}
#endif

//______________________________________________________________________________
void DebugMe() {
   // fprintf(stderr, "Debug me please \n");
}

//______________________________________________________________________________
static inline QWidget *widgetAt(int x, int y)
{
   // Find the child window (Qt itself can not do that :( strange :)
   QWidget *w = (TQtClientWidget *)QApplication::widgetAt(x,y);
   if (w) {
      QWidget *child = w->childAt(w->mapFromGlobal(QPoint(x, y )));
      if (child) w = child;
   }
   return w;
}
//______________________________________________________________________________
void TQtClientFilter::AddKeyEvent( const QKeyEvent &keyEvent, TQtClientWidget *frame)
{
   // Map and and to the ROOT event queue Qt KeyBoard event mapped to the ROOT Event_t
   // For "dest" widget 
   if (frame) {
     Event_t &evt = *new Event_t;
     memset( &evt,0,sizeof(Event_t));
     QPaintDevice *paintDev = (QPaintDevice *)frame;
     evt.fWindow    = TGQt::rootwid(paintDev);

     evt.fSendEvent = keyEvent.spontaneous();
     evt.fTime      = QTime::currentTime().msec ();
     evt.fX         = frame->x();
     evt.fY         = frame->y();
     evt.fWidth     = frame->width();	// width and
     evt.fHeight    = frame->height();	// height excluding the frame

     QPoint pointRoot = frame->mapToGlobal(QPoint(0,0));
     evt.fXRoot     = pointRoot.x();
     evt.fYRoot     = pointRoot.y();
     MapEvent(keyEvent,evt);

     fRootEventQueue->enqueue(&evt);
  }
}

//______________________________________________________________________________
bool TQtClientFilter::SelectGrab(Event_t &event, UInt_t selectEventMask,QMouseEvent &mouse) 
{
   // Select Event:  --  04.12.2005  --
   return fgGrabber ? fgGrabber->SelectGrab(event,selectEventMask, mouse) : kFALSE;
}

//______________________________________________________________________________
bool TQtClientFilter::eventFilter( QObject *qWidget, QEvent *e ){
   // Dispatch The Qt event from event queue to Event_t structure
   // Not all of the event fields are valid for each event type,
   // except fType and fWindow.

   // Set to default event. This method however, should never be called.
   // check whether we are getting the desktop event
   UInt_t selectEventMask = 0;
   Bool_t  grabSelectEvent    = kFALSE;
   static TStopwatch *filterTime = 0;
   static int neventProcessed = 0;
   neventProcessed ++;
   Event_t &evt = *new Event_t;
   memset( &evt,0,sizeof(Event_t));
   evt.fType      = kOtherEvent;

   // Cast it carefully
   TQtClientWidget *frame = dynamic_cast<TQtClientWidget *>(qWidget);
   if (!(frame /* && gQt->IsRegistered(frame)  */) )    {
         if (filterTime) filterTime->Stop();
         delete &evt;
         return kFALSE; // it is a desktop, it is NOT ROOT gui object
   }
   QPaintDevice *paintDev = (QPaintDevice *)frame;

   // Fill the default event values

   evt.fWindow    = TGQt::rootwid(paintDev);

   evt.fSendEvent = !e->spontaneous();
   evt.fTime      = QTime::currentTime().msec ();
   evt.fX         = frame->x();
   evt.fY         = frame->y();
   evt.fWidth     = frame->width();	// width and
   evt.fHeight    = frame->height();	// height excluding the frame
  
   QPoint pointRoot = frame->mapToGlobal(QPoint(0,0));
   evt.fXRoot     = pointRoot.x();
   evt.fYRoot     = pointRoot.y();

   QMouseEvent *mouseEvent = 0;
   QWheelEvent *wheelEvent = 0; //
   QKeyEvent   *keyEvent   = 0; //     Q Event::KeyPress or QEvent::KeyRelease.
   QFocusEvent *focusEvent = 0; //     Q Event::KeyPress or QEvent::KeyRelease.
   Bool_t destroyNotify = kFALSE; // by default we have to ignore all Qt::Close events
   // DumpRootEvent(event);
   
   switch ( e->type() ) {
      case QEvent::Wheel:                // mouse wheel event
         evt.fType = kButtonPress;
         wheelEvent  = (QWheelEvent *)e;
         MapEvent(*wheelEvent,evt);
         selectEventMask |=  kButtonPressMask;
         break;
         
      case QEvent::MouseButtonPress:     // mouse button pressed
         evt.fType   = kButtonPress;
      case QEvent::MouseButtonDblClick:
         if (e->type()== QEvent::MouseButtonDblClick)
            evt.fType   = kButtonDoubleClick;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,evt);
         selectEventMask |=  kButtonPressMask;
         mouseEvent->accept();
         if (    !fgGrabber
              &&  fButtonGrabList.count(frame) >0 
              &&  frame->IsGrabbed(evt) )
         {
            GrabPointer(frame, frame->ButtonEventMask(),0,frame->GrabButtonCursor(), kTRUE,kFALSE);
            // Make sure fgButtonGrabber is Ok. GrabPointer zeros fgButtonGrabber!!!
            fgButtonGrabber = frame;
            grabSelectEvent = kTRUE;
         }
         else {
            grabSelectEvent = SelectGrab(evt,selectEventMask,*mouseEvent);

            // it was grabbed ny Qt anyway, Check it
            // fprintf(stderr," 3. -- Event::MouseButtonPress -- Qt grabber %p Check the redundant grabbing current frame=%p active grabbing frame = %p; grabSelectEvent=%d. event=%p ROOT grabber %p\n"
            //      , QWidget::mouseGrabber(), frame, fgActiveGrabber,grabSelectEvent,e,fgPointerGrabber);
         }
         break;

      case QEvent::MouseButtonRelease:   // mouse button released
         evt.fType   = kButtonRelease;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,evt);
         selectEventMask |=  kButtonReleaseMask;
         // fprintf(stderr, "QEvent::MouseButtonRelease, turn grabbing OFF id = %x widget = %p,Qt grabber =%p,button grabber%p; qt event=%p\n"
         //      , TGQt::rootwid(frame), frame, QWidget::mouseGrabber(),fgButtonGrabber,e);
         if (fgButtonGrabber) {
            grabSelectEvent =  SelectGrab(evt,selectEventMask,*mouseEvent);
            if ( !mouseEvent->buttons() ) {
                GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer
            } 
         }
         else {
            grabSelectEvent = SelectGrab(evt,selectEventMask,*mouseEvent);
          }
         break;

      case QEvent::MouseMove:            // mouse move
         evt.fType   = kMotionNotify;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,evt);
         selectEventMask |=  kPointerMotionMask;
         if ( mouseEvent->buttons()  )
         {       selectEventMask |=  kButtonMotionMask;        }
         
         grabSelectEvent = SelectGrab(evt,selectEventMask,*mouseEvent);
#if 0
         {
            TQtClientWidget *w = (TQtClientWidget*)TGQt::wid(evt.fWindow); // to print
            UInt_t eventMask = w->SelectEventMask();
            UInt_t pointerMask = w->PointerMask();
             // fprintf(stderr," 1. QMouseMove evt widget id = %x pointer =%p; frame = %p; select = %d\n", evt.fWindow, w, frame, grabSelectEvent);
             // fprintf(stderr," 2. QMouseMove pointer mask %o, evt mask = %o, current mask = %o inout selected = %d\n",pointerMask, eventMask, selectEventMask,
             //      frame->IsEventSelected(selectEventMask));
         }
#endif

         // grabSelectEvent = IsGrabSelected(selectEventMask);
         break;
      case QEvent::KeyPress:             // key pressed
         keyEvent = (QKeyEvent *)e;
         MapEvent(*keyEvent,evt);         
         selectEventMask |=  kKeyPressMask;
         ((QKeyEvent *)e)->accept();
         //  fprintf(stderr, " accepted: case QEvent::KeyPress: <%c><%d>: frame = %x; grabber = %x grabbed = %d\n",evt.fCode,evt.fCode,TGQt::wid(frame), TGQt::wid(grabber),grabEvent);
         //  fprintf(stderr, "  QEvent::KeyPress: <%s>: key = %d, key_f=%d, frame = %p\n",(const char *)keyEvent->text(),keyEvent->key(),Qt::Key_F,frame);
         //  fprintf(stderr, "  QEvent::KeyPress: Current focus %p\n",(QPaintDevice *) qApp->focusWidget () );
         //  fprintf(stderr, "---------------\n\n");
         break;
      case QEvent::KeyRelease:           // key released
         keyEvent = (QKeyEvent *)e;
         MapEvent(*keyEvent,evt);
         selectEventMask |=  kKeyReleaseMask;
         ((QKeyEvent *)e)->accept();
         break;
      case QEvent::FocusIn:              // keyboard focus received
         focusEvent   = (QFocusEvent *)e;
         evt.fCode  = kNotifyNormal;
         evt.fState = 0;
         evt.fType  = kFocusIn;
         selectEventMask |=  kFocusChangeMask;
         // fprintf(stderr, "       case IN QEvent::FocusEvent:frame = %x; \n",TGQt::wid(frame));
       break;
      case QEvent::FocusOut:             // keyboard focus lost
         focusEvent   = (QFocusEvent *)e;
         evt.fCode  = kNotifyNormal;
         evt.fState = 0;
         evt.fType  = kFocusOut;
         selectEventMask |=  kFocusChangeMask;
         // fprintf(stderr, "       case OUT QEvent::FocusEvent:frame = %x; \n",TGQt::wid(frame));
         break;
      case QEvent::Enter:                // mouse enters widget
         evt.fType      = kEnterNotify;
         selectEventMask |= kEnterWindowMask | kPointerMotionMask ;
         grabSelectEvent  = IsGrabSelected(selectEventMask);
         // fprintf(stderr,"  QEvent::Enter == frame=%p, active = %p, pointer = %p;  button =%p, grabber = %p\n"
         //       , frame, fgActiveGrabber, fgPointerGrabber, fgButtonGrabber, QWidget::mouseGrabber());
         break;
      case QEvent::Leave:                // mouse leaves widget
         evt.fType      = kLeaveNotify;
         selectEventMask |= kLeaveWindowMask | kPointerMotionMask;
         grabSelectEvent  = IsGrabSelected(selectEventMask);
         // fprintf(stderr," 1. QEvent::LEAVE == cursor = %p, frame=%p, active = %p, pointer = %p button =%p; grabber = %p  grabSelect =%d, id=%x\n"
         //       , QApplication::widgetAt(QCursor::pos() )
         //       , frame, fgActiveGrabber, fgPointerGrabber, fgButtonGrabber, QWidget::mouseGrabber(), grabSelectEvent, TGQt::rootwid(fgPointerGrabber)
         //       );
         if ( fgGrabber )fgGrabber->ActivateGrabbing();
         break;
      case QEvent::Close:
         evt.fType   = kDestroyNotify;
         selectEventMask |=  kStructureNotifyMask;
         if (fgGrabber && fgGrabber->IsGrabbing(frame) ) {
            GrabPointer(0, 0, 0, 0,kFALSE);
         }
#ifndef QTCLOSE_DESTROY_RESPOND
         if ( e->spontaneous() && frame->DeleteNotify() )
         {
            frame->SetDeleteNotify(kFALSE);
            destroyNotify = kTRUE;
            ((QCloseEvent *)e)->accept();
         }
#else
         // fprintf(stderr, " QEvent::Close spontaneous %d: for %p flag = %d\n",e->spontaneous(),frame,destroyNotify);
         frame->SetClosing(); // to avoid the double notification
 // ROOT GUI does not expect this messages to be dispatched.
         if (!e->spontaneous() ) 
         {
            if (frame->DeleteNotify() ) {
               frame->SetDeleteNotify(kFALSE);
               // Ignore this Qt event, ROOT is willing to close the widget
               ((QCloseEvent *)e)->accept();
               SendCloseMessage(evt);
            }
         }
#endif
         break;
      case QEvent::Destroy:              //  during object destruction
         evt.fType   = kDestroyNotify;
         selectEventMask |=  kStructureNotifyMask;
         if (fgGrabber && fgGrabber->IsGrabbing(frame) ) {
            GrabPointer(0, 0, 0, 0,kFALSE);
         }
         // fprintf(stderr, " QEvent::Destroy spontaneous %d: for %p  flag = %d\n",e->spontaneous(),frame,destroyNotify);
#ifdef QTCLOSE_DESTROY_RESPOND
         // ROOT GUI does not expect this messages to be dispatched.
         SendCloseMessage(evt); // nothing to do here yet 
#endif         
         break;
      case QEvent::Show:                 // Widget was shown on screen,
      case QEvent::ShowWindowRequest:    //  (obsolete)  widget's window should be mapped
         evt.fType   = kMapNotify;
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Paint:                //   widget's window should be mapped
         evt.fType   = kExpose;
         MapEvent(*(QPaintEvent *)e,evt);
         selectEventMask |=  kExposureMask;
         break;
      case QEvent::Hide:                //   widget's window should be unmapped
         evt.fType   = kUnmapNotify;
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Resize:              // window move/resize event
         evt.fType   = kConfigureNotify;
         MapEvent(*(QResizeEvent *)e,evt);
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Move:
         evt.fType   = kConfigureNotify;
         MapEvent(*(QMoveEvent *)e,evt);
         selectEventMask |=  kStructureNotifyMask;
         break;
       case QEvent::Clipboard:          // window move/resize event
         evt.fType  =  kSelectionNotify;
#ifdef R__QTX11
          // this is the platform depended part
         evt.fType   = kSelectionClear; // , kSelectionRequest, kSelectionNotify;
#endif
         // grabSelectEvent  = kTRUE; // to be revised later, It was: grabEvent = kTRUE;
         selectEventMask |=  kStructureNotifyMask;
         break;
     default:
         if ( e->type() >= TQUserEvent::Id()) {
            // evt.fType = kClientMessage; ..event type will be set by MapEvent anyway
            MapEvent(*(TQUserEvent *)e,evt);
            grabSelectEvent  = kTRUE; // to be revised later, It was: grabEvent = kTRUE;
            if (evt.fType != kClientMessage && evt.fType != kDestroyNotify)
                fprintf(stderr, "** Error ** TQUserEvent:  %d %d\n", evt.fType, kClientMessage);
            else if (evt.fType == kDestroyNotify) {
               //  remove all events related to the dead window
               // fprintf(stderr,"kClientEvent kDestroyNotify %p id=%x evt\n",((TQtClientWidget*)(TGQt::wid(evt.fWindow))), evt.fWindow);
#ifdef QTDEBUG
               int nRemoved = fRootEventQueue->RemoveItems(&evt);
               fprintf(stderr,"kClientMessage kDestroyNotify %p %d events have been removed from the queue\n",evt.fWindow,nRemoved );
#endif
            }
            // else fprintf(stderr, "TQUserEvent: %p  %d %d\n", evt.fWindow, evt.fType, kClientMessage);
         } else {
            delete &evt;
            if (filterTime) filterTime->Stop();
            return kFALSE;  // We need the standard Qt processing
         }

        // GUI event types. Later merge with EEventType in Button.h and rename to
        // EEventTypes. Also rename in that case kGKeyPress to kKeyPress.
        //enum EGEventType {
        //   kClientMessage, kSelectionClear, kSelectionRequest, kSelectionNotify,
        //   kColormapNotify
        //};
       break;
   };

   if (!fRootEventQueue) {
      fRootEventQueue = new TQtEventQueue();
      // send message to another thread
   }
#if ROOT_VERSION_CODE >= ROOT_VERSION(9,15,9)         
   if (evt.fType ==  kExpose ) {
     Bool_t keepEvent4Qt =
          (((TQtClientWidget*)(TGQt::wid(evt.fWindow)))->IsEventSelected(selectEventMask) );
     if (filterTime) filterTime->Stop();
     delete &evt;
     return !keepEvent4Qt;
   }
#endif         

   if ( destroyNotify 
       || (evt.fType == kClientMessage) || (evt.fType == kDestroyNotify)  ||
       (
           (  (grabSelectEvent && ( mouseEvent  || (evt.fType == kEnterNotify ) || (evt.fType == kLeaveNotify ) ) )
         ||
           ( (!fgGrabber || !( mouseEvent  || (evt.fType == kEnterNotify ) || (evt.fType == kLeaveNotify ) ) )
            && 
            ((TQtClientWidget*)(TGQt::wid(evt.fWindow)))->IsEventSelected(selectEventMask) ) ) ) )
   {
//---------------------------------------------------------------------------
//    QT message has been mapped to ROOT one and ready to be shipped out
//---------------------------------------------------------------------------
     fRootEventQueue->enqueue(&evt);
//---------------------------------------------------------------------------
   } else {
      delete &evt;
      if (filterTime) filterTime->Stop();
      return kFALSE;  // We need the standard Qt processing
   }

   
   // We should hold ALL events because we want to process them themsleves.
   // However non-accepted mouse event should be propagated further
   if (wheelEvent && !wheelEvent->isAccepted () ) return kFALSE;
   if (mouseEvent && !mouseEvent->isAccepted () ) return kFALSE;
   if (keyEvent   && !keyEvent->isAccepted ()   ) return kFALSE;
   if (focusEvent                               ) return kFALSE;
   switch (e->type() ) {
      case QEvent::Show:                 //  Widget was shown on screen,
      case QEvent::ShowWindowRequest:    // (obsolete)  widget's window should be mapped
      case QEvent::Hide:                 //  widget's window should be unmapped
      case QEvent::Leave:
      case QEvent::Enter:
      case QEvent::Shortcut:
         if (filterTime) filterTime->Stop();
         return kFALSE;
      default: break;
//      case QEvent::Paint:                //   widget's window should be mapped
   };
  // return frame->GetCanvasWidget()==0;
  return kTRUE; // eat event. We want the special processing via TGClient
}

//______________________________________________________________________________
void TQtClientFilter::GrabPointer(TQtClientWidget *grabber, UInt_t evmask, Window_t /*confine*/,
                                    QCursor *cursor, Bool_t grab, Bool_t owner_events)
{
    // Set the X11 style active grabbing for ROOT TG widgets
   TQtPointerGrabber *gr = fgGrabber; fgGrabber = 0; 
   if (gr) {
      if (gr->IsGrabbing(fgButtonGrabber)) fgButtonGrabber = 0;
      delete gr;
   }
   if (grab) {
        fgGrabber = new TQtPointerGrabber (grabber,evmask,grabber->SelectEventMask()
                                    , cursor, grab, owner_events);
   }
}

//______________________________________________________________________________
TQtPointerGrabber *TQtClientFilter::PointerGrabber()
{   return fgGrabber; }

//______________________________________________________________________________
TQtClientWidget *TQtClientFilter::GetPointerGrabber()
{   return fgPointerGrabber; }

//______________________________________________________________________________
TQtClientWidget *TQtClientFilter::GetButtonGrabber() 
{   return fgButtonGrabber; }

//______________________________________________________________________________
void TQtClientFilter::SetButtonGrabber(TQtClientWidget *grabber)
{   fgButtonGrabber = grabber; }
   
//______________________________________________________________________________
void TQtClientFilter::AppendButtonGrab(TQtClientWidget *widget)
{   fButtonGrabList.append(widget); }

//______________________________________________________________________________
void TQtClientFilter::RemoveButtonGrab(QObject *widget)
{ 
   TQtClientWidget *wid = (TQtClientWidget *)widget;
   if ((fgButtonGrabber == wid) && fgGrabber) fgGrabber->DisactivateGrabbing();
   fButtonGrabList.removeAll(wid);
}

//______________________________________________________________________________
//
//   class TQtPointerGrabber  to implement X11 style mouse grabbing under Qt
//______________________________________________________________________________

//______________________________________________________________________________
TQtPointerGrabber::TQtPointerGrabber(TQtClientWidget *grabber, UInt_t evGrabMask
                                    , UInt_t evInputMask, QCursor *cursor
                                    , Bool_t grab, Bool_t owner_events
                                    , QWidget *confine)
{
   fIsActive= kFALSE;
   SetGrabPointer(grabber,evGrabMask, evInputMask,cursor,grab,owner_events, confine);
}
//______________________________________________________________________________
TQtPointerGrabber::~TQtPointerGrabber()
{
   SetGrabPointer(0,0,0,0,kFALSE);
}
//______________________________________________________________________________
void TQtPointerGrabber::ActivateGrabbing(bool on)
{
   // Activate the active mouse pointer event grabbing.
   static int grabCounter = 0;
   assert (fPointerGrabber);
   QWidget *qtGrabber = QWidget::mouseGrabber();
   if (on) {
      if (qtGrabber != fPointerGrabber) {
         if (qtGrabber) qtGrabber->releaseMouse();
            if (fPointerGrabber->isVisible() ) {
            if (fGrabPointerCursor) fPointerGrabber->grabMouse(*fGrabPointerCursor);
            else                    fPointerGrabber->grabMouse();
#ifdef  QT3
            if (!QApplication::hasGlobalMouseTracking () )
                 QApplication::setGlobalMouseTracking (true);
#endif
            grabCounter++;
         }
      }
      // assert (grabCounter < 2);
   } else {
      if (fIsActive && (qtGrabber != fPointerGrabber))  {
         fprintf(stderr," ** Attention ** TQtPointerGrabber::ActivateGrabbing qtGrabber %p == fPointerGrabber %p\n", qtGrabber, fPointerGrabber);
      }
      if (qtGrabber) qtGrabber->releaseMouse();
      if (fGrabPointerCursor) fPointerGrabber->SetCursor();
#ifdef  QT3
      if (QApplication::hasGlobalMouseTracking () )
          QApplication::setGlobalMouseTracking (false);
#endif
   }
   fIsActive = on;
   // Make sure the result is correct
   QWidget *grabber = QWidget::mouseGrabber();

   assert ( !fPointerGrabber->isVisible() || (fIsActive) || (!fIsActive && !grabber) );
}
//______________________________________________________________________________
void  TQtPointerGrabber::SetGrabPointer(TQtClientWidget *grabber
                             , UInt_t evGrabMask, UInt_t evInputMask
                             , QCursor *cursor, Bool_t grab, Bool_t owner_events
                             , QWidget *confine)
{
   if (grab) {
      // Grab widget. 
      fPointerGrabber       = grabber;
      fGrabPointerEventMask = evGrabMask;
      fInputPointerEventMask= evInputMask;
      fGrabPointerOwner     = owner_events;
      fGrabPointerCursor    = cursor;
      fPointerConfine       = confine;
      // Set the mouse tracking
      fPointerGrabber->setMouseTracking( fGrabPointerEventMask & kPointerMotionMask );
 } else {
      // Restore the normal mouse tracking.
      fPointerGrabber->setMouseTracking( fInputPointerEventMask & kPointerMotionMask );
            
      // Ungrab the widget.
      DisactivateGrabbing();
      
      fPointerGrabber       = 0;
      fGrabPointerEventMask = 0;
      fGrabPointerOwner     = kFALSE;
      fGrabPointerCursor    = 0;
      fPointerConfine       = 0;
 }
 // fprintf(stderr," TQtPointerGrabber::SetGrabPointer : -- grab= %d; grabber = %p; PointerMask=%o; normal Mask = %o; owner = %d\n"
 //         ,  grab, fPointerGrabber,  fGrabPointerEventMask, fInputPointerEventMask
 //         ,  fGrabPointerOwner);

}
//______________________________________________________________________________
bool TQtPointerGrabber::SelectGrab(Event_t &evt, UInt_t selectEventMask, QMouseEvent &mouse)
{ 
  // Select Event:  --  25.11.2005  --
  TQtClientWidget *widget = (TQtClientWidget*)TGQt::wid(evt.fWindow);
  bool pass2Root = FALSE;

  QWidget *grabber = QWidget::mouseGrabber();
  TQtClientWidget *pointerGrabber =  fPointerGrabber;
  if (fIsActive && grabber && (grabber != (QWidget *)pointerGrabber) )
  {
     // This is a Qt automical grabbing we have to fight with :)
     // fprintf(stderr, "1.  TQtPointerGrabber::SelectGrab Qt grabber to fight with Qt grabber = %p, ROOT grabber = %p\n"
     //      , this, grabber );
     DisactivateGrabbing();
     grabber = QWidget::mouseGrabber();
  }
  bool inside = FALSE;
  if ( ( inside = IsMouseCursorInside() ) ) {
      if ( grabber ) {
           if ( fGrabPointerOwner ) { 
                // The cursor has "Entered" the application
                // Disable the Qt automatic grabbing
                DisactivateGrabbing();
                // Generate the Enter Event 
                // . . . to be done yet . . . 
                
                // Find the child widget there to pass event to
                widget = (TQtClientWidget *)widgetAt(evt.fXRoot,evt.fYRoot);
                if (widget == pointerGrabber) widget = 0;
           } else {
              // re-grab it as needed 
              ActivateGrabbing();
              widget = 0;
           }
      } else {
         // we remained inside
         // make sure our flag is Ok. (To disable the possible Qt interfere) 
         if (!fGrabPointerOwner) 
         { 
            ActivateGrabbing();
            widget = 0;
         } else {
            DisactivateGrabbing();
            if (widget == pointerGrabber) widget = 0;
         }
      }
      // fprintf(stderr, "2. TQtPointerGrabber::SelectGrab  grabber = %p, ROOT grabber = %p widget=%p, globalTracjing =%d, MouseGrabber=%p want=%x\n"
      //      , grabber,pointerGrabber, widget, QApplication::hasGlobalMouseTracking (),QWidget::mouseGrabber(),WantEvent(widget, selectEventMask)); 

   } else { // The mouse cursor is outside of the application
       if ( !grabber ) {
          ActivateGrabbing();
          grabber = pointerGrabber;
       } else {
         //  make sure it is our grab
         assert (grabber == (QWidget *)pointerGrabber );
       }
       widget = 0;
   }
   
   // Check the selection
   if (! (fGrabPointerOwner || inside) )
   {
      mouse.accept();
      if ( IsGrabSelected (selectEventMask) ) {
          // Grab this event.
         pointerGrabber->GrabEvent(evt);
         // fprintf(stderr," QtPointerGrabber::SelectGrab  1.1. Active grabbing %p id =%x inside = %d\n",
         //        pointerGrabber, evt.fWindow, inside);
         pass2Root = TRUE;
      }
   } else {
     if ( IsGrabSelected (selectEventMask) ) {
        // Look for the the grabbing parent.
        if (widget) {
           pass2Root = (widget->SelectEventMask() & selectEventMask);
           // fprintf(stderr," QtPointerGrabber::SelectGrab  1.3. As usual grabbing %p id =%x inside = %d, pass=%d mask %o; Qt  grabber=%p\n",
           //     widget, evt.fWindow, inside, pass2Root, selectEventMask,  QWidget::mouseGrabber() );
           if (!pass2Root) {
              TQtClientWidget *parent = (TQtClientWidget *)widget->parentWidget();
              // Look up ahead
              while (parent  && !(parent->SelectEventMask() & selectEventMask) && (parent != pointerGrabber) )
              {      parent = (TQtClientWidget *)parent->parentWidget();      }
              if (!parent || parent == pointerGrabber )  widget =0;
              else if (parent && (parent != pointerGrabber) ) {
                 // fprintf(stderr," QtPointerGrabber::SelectGrab  1.4. Expect next parent  grabbing %p mask %o; Qt grabber=%p\n",
                 //      parent, selectEventMask,  QWidget::mouseGrabber() );
              }
           }
        }
        if (!widget) {
           pointerGrabber->GrabEvent(evt);
           // fprintf(stderr," QtPointerGrabber::SelectGrab  1.5. Active grabbing %p id =%x inside = %d\n",
           //     pointerGrabber, evt.fWindow, inside);
           pass2Root = TRUE;
           mouse.accept();
        }
     } else if (widget) {
         pass2Root = widget->SelectEventMask() & selectEventMask;
         // fprintf(stderr," QtPointerGrabber::SelectGrab  1.6. As usual grabbing %p id =%x inside = %d, pass=%d mask %o\n",
         //       pointerGrabber, evt.fWindow, inside, pass2Root, selectEventMask );
         // if (pass2Root) mouse.accept();       
     }
   }

   return pass2Root;
 }
//______________________________________________________________________________
Bool_t TQtPointerGrabber::IsGrabSelected(UInt_t selectEventMask) const
{  return  fGrabPointerEventMask & selectEventMask; }
