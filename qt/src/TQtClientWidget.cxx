// @(#)root/qt:$Name:  $:$Id: TQtClientWidget.cxx,v 1.8 2005/08/17 20:08:37 brun Exp $
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
#include <qaccel.h>
#include <qevent.h>
#include <qobjectlist.h>

////////////////////////////////////////////////////////////////////////////////
//
//  TQtClientWidget is QFrame designed to back the ROOT GUI TGWindow class objects
//
//
// TQtClientWidget  is a QFrame implemantation backing  ROOT TGWindow objects
// It tries to mimic the X11 Widget behaviour, that kind the ROOT Gui relies on heavily.
//
// Since ROOT has a habit to destroy the widget many times, to protect the C++ QWidget
// against of double deleting all TQtClientWidgets are to be registered with a special
// "guard" container
//
////////////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
TQtClientWidget::~TQtClientWidget()
{
   // fprintf(stderr, "dtor %p\n", this);
   // remove the event filter
   TQtClientFilter *f = gQt->QClientFilter();
   removeEventFilter(f);
   f->SetPointerGrabber(0);
   disconnect();
   if (fGuard) fGuard->DisconnectChildren(this);
   fPointerCursor = 0; // to prevent the cursor shape restoring
   UnSetButtonMask(true);
   UnSetKeyMask();
   UnSetPointerMask(true);
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
   if (!fEventMask || isHidden()) return kFALSE;
   //Test whether the current button is grabbed by this window
   bool grab = false;
   bool mask = (ev.fState & fGrabButtonMask) || (fGrabButtonMask & kAnyModifier);
   if ((fButton == kAnyButton) && mask) grab = true;
   else grab = (fButton == EMouseButton(ev.fCode)) && mask;
   // Check whether this window holds the pointer coordinate
   TQtClientWidget *w = (TQtClientWidget *)TGQt::wid(ev.fWindow);
   if (grab && (w != this) ) {
      QRect absRect   = frameGeometry();
      QWidget *parent = parentWidget();
      if (parent) {
          QPoint absPos = parent->mapToGlobal(pos());
          absRect.moveTopLeft(absPos);
      }
      grab = absRect.contains(ev.fXRoot,ev.fYRoot);
   }
   if (grab)  GrabEvent(ev);
   // fprintf(stderr,"---- TQtClientWidget::IsGrabbed grab=%d \n",grab);
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
      const QObjectList *childList = children();
      if (childList) {
         QObjectListIterator next(*childList);
         while((wg = dynamic_cast<TQtClientWidget *>(next.current())) && !(grabbed=wg->IsKeyGrabbed(ev)) ) ++next;
      }
   }
   return grabbed;
}
//______________________________________________________________________________
void TQtClientWidget::GrabEvent(Event_t &ev, bool own)
{
   // replace the original Windows_t  with the grabbing id and
   // re-caclulate the mouse coordinate
   // to respect the new Windows_t id if any
   TQtClientWidget *w = (TQtClientWidget *)TGQt::wid(ev.fWindow);
   if (w != this) {
      if (own) {
         QPoint mapped = mapFromGlobal(QPoint(ev.fXRoot,ev.fYRoot));
         // Correct the event
         ev.fX      = mapped.x();
         ev.fY      = mapped.y();
      } else {
        // replace the original Windows_t  with the grabbing id
         ev.fWindow          = TGQt::wid(this);
        // fprintf(stderr,"---- TQtClientWidget::GrabEvent\n");
      }
      // TGQt::PrintEvent(ev);
   }
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
void TQtClientWidget::SetPointerMask(UInt_t modifier, Cursor_t cursor, Bool_t owner_events)
{
   // Set the pointer mask

   fGrabPointerMask = modifier;
   fPointerOwner    = owner_events;
   fPointerCursor   = (QCursor *)cursor;
   TQtClientFilter *f = gQt->QClientFilter();
   // fprintf(stderr," TQtClientWidget::SetPointerMask %p %d %d\n",this, fGrabPointerMask, fPointerOwner);
   if (f) {
      f->AppendPointerGrab(this);
   }
}
//______________________________________________________________________________
void TQtClientWidget::UnSetPointerMask(bool dtor)
{
   // Unset the pointer mask

   if (fGrabPointerMask) {
      fGrabPointerMask = 0;
      TQtClientFilter *f = gQt->QClientFilter();
      // restore the cursor shape
      if ( this == QWidget::mouseGrabber() ) {
         releaseMouse();
         SetCursor();
      }
      // fprintf(stderr," TQtClientWidget::UnSetPointerMask %p\n", this);
      if (f && !dtor) {
         f->RemovePointerGrab(this);
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
   int index = 0;
   if (keycode) {
      if (modifier & kAnyModifier)  assert(!(modifier & kAnyModifier));
      else {
         if (modifier & kKeyShiftMask)   key[index++] = SHIFT;
         if (modifier & kKeyLockMask)    key[index++] = META;
         if (modifier & kKeyControlMask) key[index++] = CTRL;
         if (modifier & kKeyMod1Mask)    key[index++] = ALT;
     }
                                         key[index++] = UNICODE_ACCEL + keycode;
   }
   assert(index<=4);
   switch (insert) {
      case kInsert:
         if (keycode) {
           if (!fGrabbedKey)  {
              fGrabbedKey = new QAccel(this);
     //         connect(fGrabbedKey,SIGNAL(activated ( int )),this,SLOT(Accelerate(int)));
            }
            QKeySequence keys(key[0],key[1],key[2],key[3]);
            if (fGrabbedKey->findKey(keys) == -1)  {
             /* int itemId = &*/ fGrabbedKey->insertItem(keys,fGrabbedKey->count()+1);
           //      fprintf(stderr,"+%p: TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' %d, evail=%d \n", TGQt::wid(this), modifier, keycode ,fGrabbedKey->count()
           //   , fGrabbedKey->isEnabled() );
            }
            TQtClientFilter *f = gQt->QClientFilter();
            f->SetKeyGrabber(this);
        }
         break;
      case kRemove:
         if (!fGrabbedKey)  break;
         if (keycode) {
              int id = fGrabbedKey->findKey(QKeySequence(key[0],key[1],key[2],key[3]));
            // fprintf(stderr,"-%p: TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' %d\n", this, modifier, keycode ,id);
            if (id != -1) fGrabbedKey->removeItem(id);
            if (fGrabbedKey->count() ==  0) { 
                delete fGrabbedKey; fGrabbedKey = 0; 
                TQtClientFilter *f = gQt->QClientFilter();
                f->UnSetKeyGrabber(this);
            }
         } else {
           // keycode ==0 - means delete all accelerators
           // fprintf(stderr,"-%p: TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' \n", this, modifier, keycode);
             delete fGrabbedKey; fGrabbedKey = 0;
             TQtClientFilter *f = gQt->QClientFilter();
             f->UnSetKeyGrabber(this);
         }
         break;
      case kTestKey:
         if (fGrabbedKey) {
            found = (fGrabbedKey->findKey(QKeySequence(key[0],key[1],key[2],key[3])) != -1);
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
      setWFlags(getWFlags () | Qt::WRepaintNoErase | Qt:: WResizeNoErase );
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
  for (int i=0; i < l;i++) {
     switch (key[i]) {
        case SHIFT:  state |= Qt::ShiftButton;   break;
        case META:   state |= Qt::MetaButton;    break;
        case CTRL:   state |= Qt::ControlButton; break;
        case ALT:    state |= Qt::AltButton;     break;
     };
  }
  QKeyEvent ac(QEvent::KeyPress,keycode,keycode,state);
  fprintf(stderr,"TQtClientWidget::Accelerate %c\n", keycode);
  QApplication::sendEvent( this, &ac );
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
   setMouseTracking(true);
}
