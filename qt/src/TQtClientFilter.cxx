/****************************************************************************
** $Id: TQtClientFilter.cxx,v 1.64 2004/07/09 00:17:48 fine Exp $
**
** Copyright (C) 2003 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/
#include "TQtClientFilter.h"
#include "TQtRConfig.h"

#ifdef R__QTGUITHREAD
#   include "TQtNextEventMessage.h"
#endif
#include "TQtClientWidget.h"
#include "TGQt.h"
#include "TQtEventQueue.h"
#include "TQUserEvent.h"

#include "TSystem.h"
#include "TStopwatch.h"
#include "qevent.h"
#include <qdatetime.h>
#include <qcursor.h>
#include <qtextcodec.h> 

#include "KeySymbols.h"

//______________________________________________________________________________________
static inline UInt_t  MapModifierState(Qt::ButtonState qState) 
{
   UInt_t state = 0;
   if ( qState & Qt::ShiftButton   ) state |= kKeyShiftMask;
   if ( qState & Qt::ControlButton ) state |= kKeyControlMask;
   if ( qState & Qt::AltButton     ) state |= kKeyMod1Mask;
   if ( qState & Qt::RightButton   ) state |= kButton3Mask;
   if ( qState & Qt::MidButton     ) state |= kButton2Mask;
   if ( qState & Qt::LeftButton    ) state |= kButton1Mask;
   if ( qState & Qt::MetaButton    ) state |= kKeyLockMask;
   return state;
}

//______________________________________________________________________________________
static inline void MapEvent(QMouseEvent &qev, Event_t &ev) 
{
   ev.fX      = qev.x();
   ev.fY      = qev.y();
   ev.fXRoot  = qev.globalX();
   ev.fYRoot  = qev.globalY();
   Qt::ButtonState state = Qt::NoButton;
   switch ( state = (qev.type()== QEvent::MouseMove ? qev.state() :  qev.button() ) ) {
      case Qt::LeftButton:  
         // Set if the left button is pressed, or if this event refers to the left button. 
         //(The left button may be the right button on left-handed mice.) 
         ev.fCode  = kButton1; 
         // ev.fState = kButton1Mask;
         break;
      case Qt::MidButton: 
         // the middle button. 
         ev.fCode  = kButton2;
         // ev.fState = kButton2Mask;
         break;
      case Qt::RightButton:
         //  the right button
         ev.fCode  = kButton3;
         // ev.fState = kButton3Mask;
         break;
      default: 
         if (qev.type() != QEvent::MouseMove) {
          fprintf(stderr,"Error ***. Unexpected event. MapEvent(QMouseEvent &qev, Event_t &ev) state = %d\n",state);
          return;
         }
         break;
   };
   ev.fState |= MapModifierState(qev.state());
   if (ev.fCode) 
      ev.fUser[0] = TGQt::iwid(TGQt::wid(ev.fWindow)->childAt(ev.fX,ev.fY)) ;

   qev.ignore(); // propage the mouse event further
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
   {Qt::Key_BackSpace, kKey_Backspace},
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
   {Qt::Key_Prior,     kKey_Prior},
   {Qt::Key_Next,      kKey_Next},
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
   Qt::Key key = Qt::Key(qev.key());
   for (int i = 0; gKeyQMap[i].fKeySym; i++) {	// any other keys
      if (key ==  gKeyQMap[i].fQKeySym) {
         return   UInt_t(gKeyQMap[i].fKeySym);
      }
   }
#if 0   
   UInt_t text;
   QCString r = gQt->GetTextDecoder()->fromUnicode(qev.text());
   qstrncpy((char *)&text, (const char *)r,1);
   return text;
#else
   return UInt_t(qev.ascii());
#endif   
}
//______________________________________________________________________________________
static inline void MapEvent(const QKeyEvent  &qev, Event_t &ev)
{
   ev.fCode  = MapKeySym(qev);
   ev.fState = MapModifierState(qev.state());
   ev.fCount = qev.count(); 
   ev.fUser[0] = TGQt::iwid(TGQt::wid(ev.fWindow)->childAt(ev.fX,ev.fY)) ;
//   qev.accept();
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
//______________________________________________________________________________________
static inline TQtClientWidget *GrabEvent(Event_t &event/*,QPtrList<TQtClientWidget> &grabList*/) 
{
   // Substitute the orginal window if with the grabbed one if any
#if 1
   TQtClientWidget *grabbedWidget = (TQtClientWidget *)TGQt::wid(event.fWindow);
   while (grabbedWidget && !(grabbedWidget->IsGrabbed(event))) {
      QWidget *w = grabbedWidget->parentWidget();
      grabbedWidget = (TQtClientWidget *)w;
   }
   return grabbedWidget;
#else
   TQtClientWidget *grabbedWidget = grabList.first();
   while (grabbedWidget && !(grabbedWidget->IsGrabbed(event)))
      grabbedWidget = grabList.next();
#endif
}
//______________________________________________________________________________________
static inline bool GrabPointer(Event_t &event,TQtClientWidget * grabber) 
{
   // fprintf(stderr,"GrabPointer grabber=%p; event.Window=%d\n",grabber, event.fWindow);
   if (grabber) {
      if (grabber->IsGrabOwner()) return true;
      else if (grabber->IsPointerGrabbed(event)) {
         grabber->GrabEvent(event);
         return true;
      }
   }
   return false;
}
//______________________________________________________________________________
TQtClientFilter::~TQtClientFilter() 
{
   qApp->lock();
#ifdef R__QTGUITHREAD
   delete fNotifyClient;fNotifyClient=0;
#endif
   if (fRootEventQueue) {
      delete fRootEventQueue;
      fRootEventQueue = 0;
   }
   qApp->unlock();
}
//______________________________________________________________________________
static void SendCloseMessage(Event_t &closeEvent)
{
   // Send close message to window provided via closeEvent. 
   // This method should be called just use close window via WM

   if (closeEvent.fType != kDestroyNotify) return;
   Event_t event = closeEvent;

   event.fType   = kClientMessage;
   event.fFormat = 32;
   event.fHandle = gWM_DELETE_WINDOW;

   // event.fWindow  = GetId();
   event.fUser[0] = (Long_t) gWM_DELETE_WINDOW;
   event.fUser[1] = 0;
   event.fUser[2] = 0;
   event.fUser[3] = 0;
   event.fUser[4] = 0;
   
   gVirtualX->SendEvent(event.fWindow, &event);
}
//______________________________________________________________________________
void DebugMe() {
   // fprintf(stderr, "Debug me please \n");
}

//______________________________________________________________________________
bool TQtClientFilter::eventFilter( QObject *qWidget, QEvent *e ){
   // Dispatch The Qt event from event queue to Event_t structure
   // Not all of the event fields are valid for each event type,
   // except fType and fWindow.

   // Set to default event. This method however, should never be called.
   // check whether we are getting the desktop event
   ULong_t selectEventMask = 0;
   Bool_t  grabEvent       = kFALSE;
   static TStopwatch *filterTime = 0;
   static int neventProcessed = 0;
   neventProcessed ++;
   Event_t &event = *new Event_t;
   memset( &event,0,sizeof(Event_t));
   event.fType      = kOtherEvent;

   // Cast it carefully
   TQtClientWidget *frame = dynamic_cast<TQtClientWidget *>(qWidget);
   if (!frame)    {
         if (filterTime) filterTime->Stop();
         return kFALSE; // it is a desktop, it is NOT ROOOT gui object
   }
   QPaintDevice *paintDev = (QPaintDevice *)frame;
   
   // Fill the default event values
   
   event.fWindow    = TGQt::iwid(paintDev);

   event.fSendEvent = !e->spontaneous(); 
   event.fTime      = QTime::currentTime().msec ();
   event.fX         = frame->x();
   event.fY         = frame->y();
   event.fWidth     = frame->width();	// width and
   event.fHeight    = frame->height();	// height excluding the frame

   QMouseEvent *mouseEvent = 0;
   QKeyEvent   *keyEvent   = 0; //     Q Event::KeyPress or QEvent::KeyRelease.
   switch ( e->type() ) {
      case QEvent::MouseButtonPress:     // mouse button pressed
         event.fType   = kButtonPress;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,event);
         selectEventMask |=  kButtonPressMask;
         // Passive grab
         if ( fPointerGrabber ) {
//            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         } else if ( fButtonGrabList.findRef(frame) >=0 && frame->IsGrabbed(event) )
         { 
            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         } else {
            // delete &event;
            // return kTRUE;  // We do not need the standard Qt processing
            // return kFALSE;  // We do not need the standard Qt processing
         };
         break;

      case QEvent::MouseButtonRelease:   // mouse button released
         event.fType   = kButtonRelease;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,event);
         selectEventMask |=  kButtonReleaseMask;
         if ( fPointerGrabber ) {
//            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         } else if ( fButtonGrabList.findRef(frame) >=0 && frame->IsGrabbed(event) )
         { 
            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         } else {
            //delete &event;
            //return kTRUE;  // We do not need the standard Qt processing
         };
         break;

      case QEvent::MouseButtonDblClick: 
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,00,1)
         event.fType   = kButtonDoubleClick;
          // the rest code is taken from kButtonPress
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,event);
         selectEventMask |=  kButtonPressMask;
         if ( fPointerGrabber ) {
            //            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         } else if ( fButtonGrabList.findRef(frame) >=0 && frame->IsGrabbed(event) )
         { 
            ((QMouseEvent *)e)->accept();
            grabEvent = kTRUE;
         };
#endif
         break;

      case QEvent::MouseMove:            // mouse move
         event.fType   = kMotionNotify;
         mouseEvent = (QMouseEvent *)e;
         MapEvent(*mouseEvent,event);
         selectEventMask |=  kPointerMotionMask;
         if (event.fCode == kButton1 || event.fCode == kButton2 || event.fCode == kButton3)
         {
            selectEventMask |=  kButtonMotionMask;
         }
         //// check grabbing 
         
         if (fIsGrabbing && fPointerGrabber &&  QApplication::widgetAt(mouseEvent->globalPos())) 
         {
            fIsGrabbing = false;
            fPointerGrabber->releaseMouse(); 
         }
         // Active grab
         if ( GrabPointer(event, fPointerGrabber)) {
            grabEvent = kTRUE;
            ((QMouseEvent *)e)->accept();
         } else if ( event.fCode && fButtonGrabList.findRef(frame) >=0 && frame->IsGrabbed(event) ) { 
            grabEvent = kTRUE;
            ((QMouseEvent *)e)->accept();
         } else if (frame->GetCanvasWidget())  {
            ((QMouseEvent *)e)->accept();
         } else {
            delete &event;  // discard this event
            if (filterTime) filterTime->Stop();
            return kFALSE;  // We do need this event to be propagated to its parent by Qt
         }
         break;
      case QEvent::KeyPress:             // key pressed
         event.fType   = kGKeyPress;
         keyEvent = (QKeyEvent *)e;
         MapEvent(*keyEvent,event);
         selectEventMask |=  kKeyPressMask;
         // fprintf(stderr, " accepted: case QEvent::KeyPress: <%c>\n",event.fCode);
         ((QKeyEvent *)e)->ignore();              
         break;
      case QEvent::KeyRelease:           // key released
         event.fType   = kKeyRelease;
         keyEvent = (QKeyEvent *)e;
         MapEvent(*keyEvent,event);
         selectEventMask |=  kKeyReleaseMask;
         ((QKeyEvent *)e)->ignore();
         break;
      case QEvent::FocusIn:              // keyboard focus received
         event.fCode  = kNotifyNormal;
         event.fState = 0;
         event.fType  = kFocusIn;
         selectEventMask |=  kFocusChangeMask;
         break;
      case QEvent::FocusOut:             // keyboard focus lost
         event.fCode  = kNotifyNormal;
         event.fState = 0;
         event.fType  = kFocusOut;
         selectEventMask |=  kFocusChangeMask;
         break;
      case QEvent::Enter:                // mouse enters widget
         event.fType   = kEnterNotify;
         selectEventMask |=  kEnterWindowMask;
         break;
      case QEvent::Leave:                // mouse leaves widget
         event.fType   = kLeaveNotify;
         if (fPointerGrabber && !fIsGrabbing && fPointerGrabber->IsGrabOwner() && frame->isTopLevel() )
         {
            // Let's grab it to hook all messages
            // fprintf(stderr," QEvent::Leave frame=%p top=%d parent=%p\n", frame, frame->isTopLevel(),frame->parentWidget());
            fPointerGrabber->grabMouse();
            fIsGrabbing = true;
         }
         selectEventMask |=  kLeaveWindowMask;
         break;
      case QEvent::Close:             
         event.fType   = kDestroyNotify;
         ((QCloseEvent *)e)->accept();
         // ((QCloseEvent *)e)->ignore();
         fprintf(stderr, " QEvent::Close spontaneous %d:\n",e->spontaneous());
         if (fIsGrabbing && (fPointerGrabber == frame)) 
         {
            fIsGrabbing = false;
            gVirtualX->GrabPointer(0, 0, 0, 0,kFALSE);
         }
         SendCloseMessage(event);
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Destroy:              //  during object destruction
         event.fType   = kDestroyNotify;
         if (fIsGrabbing && (fPointerGrabber == frame)) 
         {
            fIsGrabbing = false;
            gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);
         }
         selectEventMask |=  kStructureNotifyMask;
         fprintf(stderr, " Event::Destroy: \n");
         // SendCloseMessage(event); nothing to do here
         break;
      case QEvent::Show:                 // Widget was shown on screen, 
      case QEvent::ShowWindowRequest:    //  (obsolete)  widget's window should be mapped
         event.fType   = kMapNotify;
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Paint:                //   widget's window should be mapped
         event.fType   = kExpose;
         MapEvent(*(QPaintEvent *)e,event);
         selectEventMask |=  kExposureMask;
         break;
      case QEvent::Hide:                //   widget's window should be unmapped
         event.fType   = kUnmapNotify;
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Resize:              // window move/resize event  
         event.fType   = kConfigureNotify;
         MapEvent(*(QResizeEvent *)e,event);
         selectEventMask |=  kStructureNotifyMask;
         break;
      case QEvent::Move: 
         event.fType   = kConfigureNotify;
         MapEvent(*(QMoveEvent *)e,event);
         selectEventMask |=  kStructureNotifyMask;
         break;
       case QEvent::Clipboard:          // window move/resize event  
         event.fType  =  kSelectionNotify;
#ifdef R__QTX11     
          // this is the platform depended part     
         event.fType   = kSelectionClear; // , kSelectionRequest, kSelectionNotify; 
#endif               
         grabEvent = kTRUE;
         selectEventMask |=  kStructureNotifyMask;
         break;
     default:
         if ( e->type() >= TQUserEvent::Id()) {
            // event.fType = kClientMessage; ..event type will be set by MapEvent anyway
            MapEvent(*(TQUserEvent *)e,event);
            grabEvent = kTRUE;
            if (event.fType != kClientMessage && event.fType != kDestroyNotify) 
                fprintf(stderr, "** Error ** TQUserEvent:  %d %d\n", event.fType, kClientMessage);
            else if (event.fType == kDestroyNotify) {
               //  remove all events related to the dead window
#ifdef QTDEBUG               
               int nRemoved = fRootEventQueue->RemoveItems(&event);
               fprintf(stderr,"kDestroyNotify %d %d events have been removed from the queue\n",event.fWindow,nRemoved );
#endif               
            }
            // else fprintf(stderr, "TQUserEvent: %p  %d %d\n", event.fWindow, event.fType, kClientMessage);
         } else {
            delete &event;
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
   qApp->lock();
   bool justInit =  false;
   if (!fRootEventQueue) {
      fRootEventQueue = new TQtEventQueue();
      // send message to another thread
      justInit = true;
   }

   if ( grabEvent || ((TQtClientWidget*)(TGQt::wid(event.fWindow)))->IsEventSelected(selectEventMask) )   
   {
//---------------------------------------------------------------------------
//    QT message has been mapped to ROOT one and ready to be shipped out
//---------------------------------------------------------------------------
      fRootEventQueue->enqueue(&event);
//---------------------------------------------------------------------------
   } else {   
      delete &event;
      if (filterTime) filterTime->Stop();
      return kFALSE;  // We need the standard Qt processing
   }

#ifdef R__QTGUITHREAD
   if (!fNotifyClient) fNotifyClient = new TQtNextEventMessage(); 
   if (justInit) {
      justInit = false;
      fNotifyClient->ExecCommandThread();
   }
   else {
      //fprintf(stderr,"TQtClientFilter::eventFilter %d %s\n", fRootEventQueue->count(), 
      //   (const char *)qWidget->name());
   }
#else
   // gSystem->DispatchOneEvent(kTRUE);
#endif
   qApp->unlock();
   // We should hold ALL events because we want to process them themsleves.
   // However non-accepted mouse event should be propagated further
   if (mouseEvent && !mouseEvent->isAccepted () ) return kFALSE;
   if (keyEvent   && !keyEvent->isAccepted ()   ) return kFALSE;
   switch (e->type() ) {
      case QEvent::Show:                 // Widget was shown on screen, 
      case QEvent::ShowWindowRequest:    //  (obsolete)  widget's window should be mapped
      case QEvent::Hide:                //   widget's window should be unmapped
         if (filterTime) filterTime->Stop();
         return kFALSE;
      default: break;
//      case QEvent::Paint:                //   widget's window should be mapped
   };
 // }
  return frame->GetCanvasWidget()==0;
 //  return kTRUE; // eat event. We want the special processing via TGClient
 // fClient?fClient->ProcessOneEvent(event):kFALSE;
}

//______________________________________________________________________________
void TQtClientFilter::AppendPointerGrab(TQtClientWidget *widget)
{ 
   if (fPointerGrabber) RemovePointerGrab(fPointerGrabber);
   fPointerGrabber = widget;
   connect(widget,SIGNAL(destroyed(QObject *)),this,SLOT(RemovePointerGrab(QObject *)));
}
//______________________________________________________________________________
void TQtClientFilter::RemovePointerGrab(QObject *widget)
{ 
   if (!widget && fPointerGrabber ) fPointerGrabber->UnSetPointerMask();
   else if (widget) {
      fPointerGrabber = 0;
      disconnect(widget,SIGNAL(destroyed(QObject *)),this,SLOT(RemovePointerGrab(QObject *)));
   }
}
