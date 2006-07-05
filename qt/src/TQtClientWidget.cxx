// @(#)root/qt:$Name:  $:$Id: TQtClientWidget.cxx,v 1.12 2006/05/08 13:16:56 antcheva Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQtWidget.h"
#include "TQtClientWidget.h"
#include "TQtClientFilter.h"
#include "TQtClientGuard.h"
#include "TGQt.h"
#include "TQtLock.h"

#include <qkeysequence.h>
#if QT_VERSION < 0x40000
#  include <qaccel.h>
#  include <qobjectlist.h>
#else /* QT_VERSION */
#  include <q3accel.h>
#  include <qobject.h>
#  include <QKeyEvent>
#  include <QCloseEvent>
#endif /* QT_VERSION */
#include <qevent.h>

////////////////////////////////////////////////////////////////////////////////
//
//  TQtClientWidget is QFrame designed to back the ROOT GUI TGWindow class objects
//
//
// TQtClientWidget  is a QFrame implementation backing  ROOT TGWindow objects
// It tries to mimic the X11 Widget behaviour, that kind the ROOT Gui relies on heavily.
//
// Since ROOT has a habit to destroy the widget many times, to protect the C++ QWidget
// against of double deleting all TQtClientWidgets are to be registered with a special
// "guard" container
//
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TQtClientWidget::TQtClientWidget(TQtClientGuard *guard, QWidget* parent, const char* name, Qt::WFlags f ):
          CLIENT_WIDGET_BASE_CLASS(parent,name,f),
          fGrabButtonMask(kAnyModifier),      fGrabEventPointerMask(kNoEventMask)
         ,fGrabEventButtonMask(kNoEventMask), fSelectEventMask(kNoEventMask), fSaveSelectInputMask(kNoEventMask) // ,fAttributeEventMask(0)
         ,fButton(kAnyButton),fGrabbedKey(0), fPointerOwner(kFALSE)
         ,fNormalPointerCursor(0),fGrabPointerCursor(0),fGrabButtonCursor(0)
         ,fIsClosing(false)  ,fDeleteNotify(false), fGuard(guard)
         ,fCanvasWidget(0)
{
#if QT_VERSION >= 0x40000
   setAttribute(Qt::WA_PaintOnScreen);
   setAttribute(Qt::WA_PaintOutsidePaintEvent);
#endif
}

//______________________________________________________________________________
TQtClientWidget::~TQtClientWidget()
{
   // fprintf(stderr, "TQtClientWidget::~TQtClientWidget dtor %p\n", this);
   // remove the event filter
   TQtClientFilter *f = gQt->QClientFilter();
   // Do we still grabbing anything ?
   if (f) f->GrabPointer(this, 0, 0, 0, kFALSE);  // ungrab pointer
   disconnect();
   if (fGuard) fGuard->DisconnectChildren(this);
   fNormalPointerCursor = 0; // to prevent the cursor shape restoring
   UnSetButtonMask(true);
   UnSetKeyMask();
   if (!IsClosing())
      gQt->SendDestroyEvent(this);  // notify TGClient we have been destroyed
}

//______________________________________________________________________________
void TQtClientWidget::closeEvent(QCloseEvent *ev)
{
   // This Qt QCloseEvent event handler

   // Close events are sent to widgets that the user wants to close,
   // usually by choosing "Close" from the window menu, or by clicking
   // the `X' titlebar button. They are also sent when you call QWidget::close()
   // to close a widget programmatically.

   printf("TQtClientWidget::closeEvent(QCloseEvent *ev)\n");
   QWidget::closeEvent(ev);
}
//______________________________________________________________________________
bool TQtClientWidget::IsGrabbed(Event_t &ev)
{
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // XGrabButton(3X11)         XLIB FUNCTIONS        XGrabButton(3X11)
   //   �    The pointer is not grabbed, and the specified button is logically
   //        pressed when the specified modifier keys are logically down,
   //        and no other buttons or modifier keys are logically down.
   //   �    The grab_window contains the pointer.
   //   �    The confine_to window (if any) is viewable.
   //   �    A passive grab on the same button/key combination does not exist
   //        on any ancestor of grab_window.
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   bool grab = false;
   QWidget *parent = parentWidget();
//   fprintf(stderr,"\n -1- TQtClientWidget::IsGrabbed  parent = %p mask %o register = %d "
//          , parent, ButtonEventMask(),TGQt::IsRegistered(parent));
   if (     ButtonEventMask()
         && !isHidden() 
         && !(   parent 
               && dynamic_cast<TQtClientWidget*>(parent)  // TGQt::IsRegistered(parent)
               && ((TQtClientWidget *)parent)->IsGrabbed(ev)
             )
      )
      {

        //Test whether the current button is grabbed by this window
        bool mask = (ev.fState & fGrabButtonMask) || (fGrabButtonMask & kAnyModifier);
        
        if ((fButton == kAnyButton) && mask)
           grab = true;
        else 
           grab = (fButton == EMouseButton(ev.fCode)) && mask;
        
        // Check whether this window holds the pointer coordinate
        TQtClientWidget *w = (TQtClientWidget *)TGQt::wid(ev.fWindow);
        if (grab && (w != this) ) {
           QRect absRect = geometry();
           QPoint absPos = mapToGlobal(QPoint(0,0));
           absRect.moveTopLeft(absPos);
           grab = absRect.contains(ev.fXRoot,ev.fYRoot);
        }

        if (grab)   GrabEvent(ev);
     }
   //  fprintf(stderr," this = %p grab=%d \n", this, grab);
   // TGQt::PrintEvent(ev);

   return grab;
}
//______________________________________________________________________________
TQtClientWidget *TQtClientWidget::IsKeyGrabbed(const Event_t &ev)
{
   // Check ROOT Event_t ev structure for the KeyGrab mask

   // fprintf(stderr,"Do we grab ? current window %p; event window = %p  code <%c>, grabber = %p\n",TGQt::wid(this), TGQt::rootwid(TGQt::wid(ev.fWindow)), ev.fCode,fGrabbedKey);
   TQtClientWidget *grabbed = 0;
   UInt_t modifier = ev.fState;
    
   if (SetKeyMask(ev.fCode,  modifier, kTestKey)) grabbed = this;
   if (grabbed && ( ev.fType == kKeyRelease)) {
      SetKeyMask(ev.fCode,  modifier, kRemove);
   }
   TQtClientWidget *wg = this;
   if (!grabbed) {
      // check parent 
      do {
          wg = (TQtClientWidget *)wg->parentWidget();
      }  while ( wg && (grabbed = wg->IsKeyGrabbed(ev)) );
   }
   if (!grabbed) {
      // Check children
#if QT_VERSION < 0x40000
      const QObjectList *childList = children();
      if (childList) {
         QObjectListIterator next(*childList);
         while((wg = dynamic_cast<TQtClientWidget *>(next.current())) && !(grabbed=wg->IsKeyGrabbed(ev)) ) ++next;
#else /* QT_VERSION */
      const QObjectList &childList = children();
      if (!childList.isEmpty()) {
         QListIterator<QObject*> next(childList);
         while(next.hasNext() && (wg = dynamic_cast<TQtClientWidget *>(next.next ())) && !(grabbed=wg->IsKeyGrabbed(ev)) ){;}
#endif /* QT_VERSION */
      }
   }
   return grabbed;
}
//______________________________________________________________________________
void TQtClientWidget::GrabEvent(Event_t &ev, bool /*own*/)
{
   // replace the original Windows_t  with the grabbing id and
   // re-caclulate the mouse coordinate
   // to respect the new Windows_t id if any
   TQtClientWidget *w = (TQtClientWidget *)TGQt::wid(ev.fWindow);
   if (w != this) {
      QPoint mapped = mapFromGlobal(QPoint(ev.fXRoot,ev.fYRoot));
      // Correct the event
      ev.fX      = mapped.x();
      ev.fY      = mapped.y();
      // replace the original Windows_t  with the grabbing id
      ev.fWindow          = TGQt::wid(this);
      // fprintf(stderr,"---- TQtClientWidget::GrabEvent\n");
   }
  // TGQt::PrintEvent(ev);
}
//______________________________________________________________________________
void TQtClientWidget::SelectInput (UInt_t evmask) 
{
   // Select input and chech whether qwe nat mouse tracking
   fSelectEventMask=evmask;
   assert(fSelectEventMask != (UInt_t) -1);
   setMouseTracking( fSelectEventMask & kPointerMotionMask );
}
//______________________________________________________________________________
void TQtClientWidget::SetButtonMask(UInt_t modifier,EMouseButton button)
{
   // Set the Button mask
   fGrabButtonMask  = modifier; fButton = button;
   TQtClientFilter *f = gQt->QClientFilter();
   if (f) {
      f->AppendButtonGrab(this);
      connect(this,SIGNAL(destroyed(QObject *)),f,SLOT(RemoveButtonGrab(QObject *)));
   }
}
//______________________________________________________________________________
void TQtClientWidget::UnSetButtonMask(bool dtor)
{
   // Unset the Button mask

   if (fGrabButtonMask) {
      fGrabButtonMask = 0;
      TQtClientFilter *f = gQt->QClientFilter();
      if (f) {
         if ( !dtor ) disconnect(this,SIGNAL(destroyed(QObject *)),f,SLOT(RemoveButtonGrab(QObject *)));
         f->RemoveButtonGrab(this);
      }
   }
}
//______________________________________________________________________________
Bool_t TQtClientWidget::SetKeyMask(Int_t keycode, UInt_t modifier, int insert)
{
   // Set the key button mask
   // insert   = -1 - remove
   //             0 - test
   //            +1 - insert
   Bool_t found = kTRUE;
   int key[5]= {0,0,0,0,0};
   int ikeys = 0;
   int index = 0;
   if (keycode) {
      if (modifier & kAnyModifier)  assert(!(modifier & kAnyModifier));
      else {
         if (modifier & kKeyShiftMask)   { key[index++] = Qt::SHIFT; ikeys += Qt::SHIFT;}
         if (modifier & kKeyLockMask)    { key[index++] = Qt::META;  ikeys += Qt::META; }
         if (modifier & kKeyControlMask) { key[index++] = Qt::CTRL;  ikeys += Qt::CTRL; }
         if (modifier & kKeyMod1Mask)    { key[index++] = Qt::ALT;   ikeys += Qt::ALT;  }
     }
                                           key[index++] = Qt::UNICODE_ACCEL + keycode;  ikeys += Qt::UNICODE_ACCEL + keycode; 
   }
   QKeySequence keys(ikeys);

   assert(index<=4);
   switch (insert) {
      case kInsert:
         if (keycode) {
           if (!fGrabbedKey)  {
#if QT_VERSION < 0x40000
              fGrabbedKey = new QAccel(this);
#else /* QT_VERSION */
              fGrabbedKey = new Q3Accel(this);
#endif /* QT_VERSION */
               connect(fGrabbedKey,SIGNAL(activated ( int )),this,SLOT(Accelerate(int)));
            }
            if (fGrabbedKey->findKey(keys) == -1)  {
//               int itemId = 
                fGrabbedKey->insertItem(keys);
//                fprintf(stderr,"+%p: TQtClientWidget::SetKeyMask  modifier =%d  keycode = \'%c\' item=%d enable=%d\n", TGQt::wid(this), modifier, keycode ,itemId
//              , fGrabbedKey->isEnabled() );
            }
        }
         break;
      case kRemove:
         if (!fGrabbedKey)  break;
         if (keycode) {
              int id = fGrabbedKey->findKey(keys);
            if (id != -1) { fGrabbedKey->removeItem(id); }
            if (fGrabbedKey->count() ==  0) { 
                delete fGrabbedKey; fGrabbedKey = 0; 
            }
         } else {
           // keycode ==0 - means delete all accelerators
           // fprintf(stderr,"-%p: TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' \n", this, modifier, keycode);
             delete fGrabbedKey; fGrabbedKey = 0;
         }
         break;
      case kTestKey:
         if (fGrabbedKey) {
//            found = (fGrabbedKey->findKey(QKeySequence(key[0],key[1],key[2],key[3])) != -1);
            found = (fGrabbedKey->findKey(keys) != -1);
            // fprintf(stderr,"\n+%p:testing  TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' found=%d \n", TGQt::wid(this), modifier, keycode ,found);
         }

         break;
      default: break;
  }
  return found;
}
//______________________________________________________________________________
void TQtClientWidget::SetCanvasWidget(TQtWidget *widget)
{
   // Associate this widget with the parent ROOT gui widget
   TQtLock lock;
   if (fCanvasWidget)
      disconnect(fCanvasWidget,SIGNAL(destroyed()), this, SLOT(disconnect()));
   fCanvasWidget = widget;
   if (fCanvasWidget) {
      // may be transparent
#if QT_VERSION < 0x40000
      setWFlags(getWFlags () | Qt::WRepaintNoErase | Qt:: WResizeNoErase );
#else /* QT_VERSION */
      setWindowFlags(windowFlags()  | Qt::WNoAutoErase | Qt:: WResizeNoErase );
#endif /* QT_VERSION */
      connect(fCanvasWidget,SIGNAL(destroyed()),this,SLOT(Disconnect()));
   }
}
//______________________________________________________________________________
void TQtClientWidget::UnSetKeyMask(Int_t keycode, UInt_t modifier)
{
  // Unset the key button mask

  SetKeyMask(keycode, modifier, kRemove);
}
//_____slot _________________________________________________________________________
void TQtClientWidget::Accelerate(int id)
{
  // Qt slot to respond to the "Keyboard accelerator signal"
  QKeySequence key = fGrabbedKey->key(id);
  int l = key.count();
  int keycode = key[l-1];
  uint state =0;
  
#if QT_VERSION < 0x40000
  if (keycode & Qt::SHIFT) state |=  Qt::ShiftButton;
  if (keycode & Qt::META)  state |=  Qt::MetaButton;
  if (keycode & Qt::CTRL)  state |=  Qt::ControlButton;
  if (keycode & Qt::ALT)   state |=  Qt::AltButton;
#else /* QT_VERSION */
  if (keycode & Qt::SHIFT) state |=  Qt::ShiftModifier;
  if (keycode & Qt::META)  state |=  Qt::MetaModifier;
  if (keycode & Qt::CTRL)  state |=  Qt::ControlModifier;
  if (keycode & Qt::ALT)   state |=  Qt::AltModifier;
#endif /* QT_VERSION */
        
  // Create ROOT event
  QKeyEvent ac(QEvent::KeyPress,keycode,keycode,state);
  // call Event filter directly 
  TQtClientFilter *f = gQt->QClientFilter();
  if (f) f->AddKeyEvent(ac,this); 
  QKeyEvent acRelease(QEvent::KeyRelease,keycode,keycode,state);
  if (f) f->AddKeyEvent(acRelease,this); 
}
//______________________________________________________________________________
void TQtClientWidget::Disconnect()
{
  // Disconnect the Canvas and ROOT gui widget before destroy.

   SetCanvasWidget(0);           }

//______________________________________________________________________________
void TQtClientWidget::polish()
{
   // Delayed initialization of a widget.
   // This function will be called after a widget has been fully created
   // and before it is shown the very first time.

   QWidget::polish();
   // setMouseTracking(true);
}
