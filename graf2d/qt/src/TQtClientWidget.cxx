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

#include "TQtWidget.h"
#include "TQtClientWidget.h"
#include "TQtClientFilter.h"
#include "TQtClientGuard.h"
#include "TGQt.h"
#include "TQtLock.h"

#include <QKeySequence>
#include <QShortcut>
#include <QKeyEvent>
#include <QCloseEvent>
#include <QEvent>
#include <QDebug>

#include "TGClient.h"

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
TQtClientWidget::TQtClientWidget(TQtClientGuard *guard, QWidget* mother, const char* name, Qt::WFlags f ):
          QFrame(mother,f)
         ,fGrabButtonMask(kAnyModifier),      fGrabEventPointerMask(kNoEventMask)
         ,fGrabEventButtonMask(kNoEventMask), fSelectEventMask(kNoEventMask), fSaveSelectInputMask(kNoEventMask) // ,fAttributeEventMask(0)
         ,fButton(kAnyButton), fPointerOwner(kFALSE)
         ,fNormalPointerCursor(0),fGrabPointerCursor(0),fGrabButtonCursor(0)
         ,fIsClosing(false)  ,fDeleteNotify(false), fGuard(guard)
         ,fCanvasWidget(0),fMyRootWindow(0),fEraseColor(0), fErasePixmap(0)
{
   setObjectName(name);
   setAttribute(Qt::WA_PaintOnScreen);
   setAttribute(Qt::WA_PaintOutsidePaintEvent);
   setAutoFillBackground(true);
 //   fEraseColor  = new QColor("red");
//   fErasePixmap = new QPixmap(palette().brush(QPalette::Window).texture());
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
   delete fEraseColor;  fEraseColor  = 0;
   delete fErasePixmap; fErasePixmap = 0;
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
void TQtClientWidget::setEraseColor(const QColor &color)
{
   // Color to paint widget background with our PainEvent
   if (!fEraseColor)
      fEraseColor = new QColor(color);
   else
      *fEraseColor = color;
   QPalette pp = palette();
   pp.setColor(QPalette::Window, *fEraseColor);
   setPalette(pp);
//            win->setBackgroundRole(QPalette::Window);
}

//______________________________________________________________________________
void TQtClientWidget::setErasePixmap (const QPixmap &pixmap)
{
   // pixmap to paint widget background with our PainEvent
   if (!fErasePixmap)
      fErasePixmap = new QPixmap(pixmap);
   else
      *fErasePixmap = pixmap;

   QPalette pp = palette();
   pp.setBrush(QPalette::Window, QBrush(*fErasePixmap));
   setPalette(pp);
//            win->setBackgroundRole(QPalette::Window);
}

//______________________________________________________________________________
bool TQtClientWidget::IsGrabbed(Event_t &ev)
{
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // XGrabButton(3X11)         XLIB FUNCTIONS        XGrabButton(3X11)
   //   *    The pointer is not grabbed, and the specified button is logically
   //        pressed when the specified modifier keys are logically down,
   //        and no other buttons or modifier keys are logically down.
   //   *    The grab_window contains the pointer.
   //   *    The confine_to window (if any) is viewable.
   //   *    A passive grab on the same button/key combination does not exist
   //        on any ancestor of grab_window.
   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   bool grab = false;
   QWidget *mother = parentWidget();
//   fprintf(stderr,"\n -1- TQtClientWidget::IsGrabbed  parent = %p mask %o register = %d "
//          , parent, ButtonEventMask(),TGQt::IsRegistered(parent));
   if (     ButtonEventMask()
         && !isHidden()
         && !(   mother
               && dynamic_cast<TQtClientWidget*>(mother)  // TGQt::IsRegistered(parent)
               && ((TQtClientWidget *)mother)->IsGrabbed(ev)
             )
      )
      {

        //Test whether the current button is grabbed by this window
        bool msk = (ev.fState & fGrabButtonMask) || (fGrabButtonMask & kAnyModifier);

        if ((fButton == kAnyButton) && msk)
           grab = true;
        else
           grab = (fButton == EMouseButton(ev.fCode)) && msk;

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
     const QObjectList &childList = children();
      if (!childList.isEmpty()) {
         QListIterator<QObject*> next(childList);
         while(next.hasNext() && (wg = dynamic_cast<TQtClientWidget *>(next.next ())) && !(grabbed=wg->IsKeyGrabbed(ev)) ){;}
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
   int ikeys = 0;
   if (keycode) {
      if (modifier & kKeyShiftMask)   ikeys |= Qt::SHIFT;
      if (modifier & kKeyLockMask)    ikeys |= Qt::META;
      if (modifier & kKeyControlMask) ikeys |= Qt::CTRL;
      if (modifier & kKeyMod1Mask)    ikeys |= Qt::ALT;
                                      ikeys |= keycode;
   }
   QKeySequence keys(ikeys);

   std::map<QKeySequence,QShortcut*>::iterator i = fGrabbedKey.find(keys);
   switch (insert) {
      case kInsert:
         if (keycode) {
            if ( i == fGrabbedKey.end()) {
               fGrabbedKey.insert(
                     std::pair<QKeySequence,QShortcut*>(keys,new QShortcut(keys,this,SLOT(Accelerate()),SLOT(Accelerate()),Qt::ApplicationShortcut))
                     );
                // qDebug() << "TQtClientWidget::SetKeyMask()" << this << " key=" << keys;
            } else {
               (*i).second->setEnabled(true);
            }
         }
         break;
      case kRemove:
         if (keycode) {
            if ( i != fGrabbedKey.end())
               (*i).second->setEnabled(false);
        } else {
            // keycode ==0 - means delete all accelerators
            // fprintf(stderr,"-%p: TQtClientWidget::SetKeyMask modifier=%d keycode \'%c\' \n", this, modifier, keycode);
            std::map<QKeySequence,QShortcut*>::iterator j = fGrabbedKey.begin();
            while (j != fGrabbedKey.end()) {
               (*j).second->setEnabled(false);
               ++j;
           }
         }
         break;
      case kTestKey:
         found = i != fGrabbedKey.end();
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
void TQtClientWidget::Accelerate()
{
  // Qt slot to respond to the "Keyboard accelerator signal"
  QShortcut *cut = (QShortcut *)sender();
  QKeySequence key = cut->key ();
  qDebug() << "TQtClientWidget::Accelerate()" << key;
  int l = key.count();
  int keycode = key[l-1];
  Qt::KeyboardModifiers state = Qt::NoModifier;

  if (keycode & Qt::SHIFT) state |=  Qt::ShiftModifier;
  if (keycode & Qt::META)  state |=  Qt::MetaModifier;
  if (keycode & Qt::CTRL)  state |=  Qt::ControlModifier;
  if (keycode & Qt::ALT)   state |=  Qt::AltModifier;

  // Create ROOT event
  QKeyEvent ac(QEvent::KeyPress,keycode & 0x01FFFFFF,state);
  // call Event filter directly
  TQtClientFilter *f = gQt->QClientFilter();
  if (f) f->AddKeyEvent(ac,this);
  QKeyEvent acRelease(QEvent::KeyRelease,keycode & 0x01FFFFFF,state);
  if (f) f->AddKeyEvent(acRelease,this);
}
//______________________________________________________________________________
void TQtClientWidget::Disconnect()
{
  // Disconnect the Canvas and ROOT gui widget before destroy.

   SetCanvasWidget(0);           }

//______________________________________________________________________________
void TQtClientWidget::paintEvent( QPaintEvent *e )
{
   QFrame::paintEvent(e);
#if ROOT_VERSION_CODE >= ROOT_VERSION(9,15,9)
   if (gClient) {
      // Find my host ROOT TGWindow
      if (!fMyRootWindow)
         fMyRootWindow = gClient->GetWindowById(TGQt::rootwid(this));
      if (fMyRootWindow) {
         gClient->NeedRedraw(fMyRootWindow,kTRUE);
      }
   }
#endif
}
