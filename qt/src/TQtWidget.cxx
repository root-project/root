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

// Definition of TQtWidget class
// "double-buffere widget

#include <qapplication.h>

#include "TQtWidget.h"
#include "TROOT.h"
#include "TGQt.h"
#include "TCanvas.h"
#include "Buttons.h"
#include "qevent.h"
#include "qpainter.h"
#include "qpixmap.h"

#ifdef R__QTWIN32
// #include "Windows4Root.h"
#include "TWinNTSystem.h"
#include "Win32Constants.h"
#endif

////////////////////////////////////////////////////////////////////////////////
//
//  TQtWidget is QWidget with QPixmap double buffer
//  It designed to back the ROOT TCanvasImp class interface  and it can be used
//  as a regular Qt Widget to create Qt-based GUI with embedded TCanvas objects
//
//  This widget can be used to build a custom GUI interfaces with  Qt Designer
//
////////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
TCanvas  *TQtWidget::Canvas()
{
#ifdef R__QTGUITHREAD
   if (qApp->tryLock() ) {
      TCanvas  *c = 0;
      if (fCanvas)
         c = gROOT->IsLineProcessing() ? 0 : GetCanvas();
      qApp->unlock();
      return c;
   }
   return 0;
#else
   return GetCanvas();
#endif
};

//_____________________________________________________________________________
TQtWidget::TQtWidget(QWidget* parent, const char* name, WFlags f,bool embedded):QWidget(parent,name,f)
          ,fCanvas(0),fPixmapID(this),fPaint(TRUE),fSizeChanged(FALSE)
          ,fDoubleBufferOn(FALSE),fEmbedded(embedded),fWrapper(0)
{
  setFocusPolicy(QWidget::WheelFocus);
  setWFlags(getWFlags () | Qt::WRepaintNoErase | Qt:: WResizeNoErase );
  setBackgroundMode(Qt::NoBackground);
  if (fEmbedded) {
    Bool_t batch = gROOT->IsBatch();
    if (!batch) gROOT->SetBatch(kTRUE); // to avoid the recursion within TCanvas ctor
    fCanvas = new TCanvas(name, 4, 4, TGQt::iwid(this));
    // fprintf(stderr,"TQtWidget::TQtWidget fEditable %d\n", fCanvas->IsEditable());
    gROOT->SetBatch(batch);
    connect(this, SIGNAL(destroyed()),SLOT(Disconnect()));
  }
  fSizeHint = QWidget::sizeHint();
  setSizePolicy (QSizePolicy::Expanding ,QSizePolicy::Expanding );
#ifdef R__QTWIN32
   // Set the application icon for all ROOT widgets
   static HICON rootIcon = 0;
   if (!rootIcon) {
      HICON hIcon = ((TWinNTSystem *)gSystem)->GetSmallIcon(kMainROOTIcon);
      if (!hIcon) hIcon = LoadIcon(NULL, IDI_APPLICATION);
      rootIcon = hIcon;
      SetClassLong(winId(),        // handle to window
                   GCL_HICON,      // index of value to change
                   LONG(rootIcon)  // new value
      );
    }
#endif
}
//______________________________________________________________________________
TQtWidget::~TQtWidget()
{
  qApp->lock();
  fCanvas =0;
  qApp->unlock();
}

//_____________________________________________________________________________
void TQtWidget::adjustSize()
{
  // Adjusts the size of the widget to fit the contents.
  // Adjust the size of the double buffer to the
  // current Widget size
  QWidget::adjustSize ();
  AdjustBufferSize();
  update();
}
//_____________________________________________________________________________
void TQtWidget::erase ()
{
  // Erases the specified area (x, y, w, h) in the widget
  // without generating a paint event.
  QWidget::erase();
  fPixmapID.fill();
}
//_____________________________________________________________________________
void TQtWidget::cd()
{
 // [slot] to make this embedded canvas the current one
  cd(0);
}
 //______________________________________________________________________________
void TQtWidget::cd(int subpadnumber)
{
 // [slot] to make this embedded canvas / pad the current one
  qApp->lock();
  TCanvas *c = fCanvas;
  if (c) c->cd(subpadnumber);
  qApp->unlock();
}
//______________________________________________________________________________
void TQtWidget::Disconnect()
{
   // Disconnect the Qt widget from CTanvas object before deleting
   // to avoid the dead lock
  qApp->lock();
 // one has to set CanvasID = 0 to disconnect things properly.
  TCanvas *c = fCanvas; fCanvas = 0; delete c;
  qApp->unlock();
}
//_____________________________________________________________________________
void TQtWidget::Refresh()
{
   // Qt slot to allow Qt signal refreshing TOOT TCanvas if needed

   TCanvas *c = Canvas();
   if (!fPixmapID.paintingActive())  AdjustBufferSize();
   if (c) {
      c->Resize();
      c->Update();
   }
}

//_____________________________________________________________________________
void TQtWidget::resize (int w, int h)
{
   // resize the widget and its double buffer
   // fprintf(stderr,"TQtWidget::resize (int w=%d, int h=%d)\n",w,h);
   QWidget::resize(w,h);
   AdjustBufferSize();
   // fPixmapID.fill();
}

//_____________________________________________________________________________
void TQtWidget::customEvent(QCustomEvent *e)
{
   // The custom responce to the special WIN32 events
   // These events are not present with X11 systems
   switch (e->type() - QEvent::User) {
   case kEXITSIZEMOVE:
   { // WM_EXITSIZEMOVE
      fPaint = TRUE;
      setUpdatesEnabled( TRUE );
      exitSizeEvent();
         break;
   }
   case kENTERSIZEMOVE:
   {
      //  WM_ENTERSIZEMOVE
      fSizeChanged=FALSE;
      fPaint = FALSE;
      setUpdatesEnabled( FALSE );
   }
   case kFORCESIZE:
   default:
      {
         // Force resize
         fPaint       = TRUE;
         fSizeChanged = TRUE;
         setUpdatesEnabled( TRUE );
         exitSizeEvent();
         break;
      }
   };
}
//_____________________________________________________________________________
void TQtWidget::focusInEvent ( QFocusEvent *e )
{
   // The custom responce to the Qt QFocusEvent "in"
   // this imposes an extra protection to avoid TObject interaction with
   // mouse event accidently
   if (!fWrapper && e->gotFocus()) {
      setMouseTracking(TRUE);
   }
   if ( autoMask() ) updateMask();
}
//_____________________________________________________________________________
void TQtWidget::focusOutEvent ( QFocusEvent *e )
{
   // The custom responce to the Qt QFocusEvent "out"
   // this imposes an extra protection to avoid TObject interaction with
   // mouse event accidently
   if (!fWrapper && e->lostFocus()) {
      setMouseTracking(FALSE);
   }
   if ( autoMask() ) updateMask();
}

//_____________________________________________________________________________
void TQtWidget::mousePressEvent (QMouseEvent *e)
{
   // Map the Qt mouse press button event to the ROOT TCanvas events
   // Mouse events occur when a mouse button is pressed or released inside
   // a widget or when the mouse cursor is moved.

   //    kButton1Down   =  1, kButton2Down   =  2, kButton3Down   =  3,
   EEventType rootButton = kNoEvent;
   TCanvas *c = Canvas();
   if (c){
      switch (e->button ())
      {
      case Qt::LeftButton:  rootButton = kButton1Down; break;
      case Qt::RightButton: rootButton = kButton3Down; break;
      case Qt::MidButton:   rootButton = kButton2Down; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         c->HandleInput(rootButton, e->x(), e->y());
         e->accept(); return;
      }
   }
   QWidget::mousePressEvent(e);
}

//_____________________________________________________________________________
void TQtWidget::mouseMoveEvent (QMouseEvent * e)
{
   //  Map the Qt mouse move pointer event to the ROOT TCanvas events
   //  kMouseMotion   = 51,
   //  kButton1Motion = 21, kButton2Motion = 22, kButton3Motion = 23, kKeyPress = 24
   EEventType rootButton = kMouseMotion;
   TCanvas *c = Canvas();
   if (c){
      if (e->state() & LeftButton) { rootButton = kButton1Motion; }
      c->HandleInput(rootButton, e->x(), e->y());
      e->accept(); return;
   }
   QWidget::mouseMoveEvent(e);
}

//_____________________________________________________________________________
void TQtWidget::mouseReleaseEvent(QMouseEvent * e)
{
   //  Map the Qt mouse button release event to the ROOT TCanvas events
   //   kButton1Up     = 11, kButton2Up     = 12, kButton3Up     = 13
   EEventType rootButton = kNoEvent;
   TCanvas *c = Canvas();
   if (c){
      switch (e->button())
      {
      case Qt::LeftButton:  rootButton = kButton1Up; break;
      case Qt::RightButton: rootButton = kButton3Up; break;
      case Qt::MidButton:   rootButton = kButton2Up; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         c->HandleInput(rootButton, e->x(), e->y());
         gPad->Modified();
         e->accept(); return;
      }
   }
   QWidget::mouseReleaseEvent(e);
}

//_____________________________________________________________________________
void TQtWidget::mouseDoubleClickEvent(QMouseEvent * e)
{
   //  Map the Qt mouse double click button event to the ROOT TCanvas events
   //  kButton1Double = 61, kButton2Double = 62, kButton3Double = 63
   EEventType rootButton = kNoEvent;
   TCanvas *c = Canvas();
   if (c){
      switch (e->button())
      {
      case Qt::LeftButton:  rootButton = kButton1Double; break;
      case Qt::RightButton: rootButton = kButton3Double; break;
      case Qt::MidButton:   rootButton = kButton2Double; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         c->HandleInput(rootButton, e->x(), e->y());
         e->accept(); return;
      }
   }
   QWidget::mouseDoubleClickEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::keyPressEvent(QKeyEvent * e)
{
   //  Map the Qt key press event to the ROOT TCanvas events
   // kKeyDown  =  4
   TCanvas *c = Canvas();
   if (c){
      c->HandleInput(kKeyPress, e->ascii(), e->key());
   }
   QWidget::keyPressEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::keyReleaseEvent(QKeyEvent * e)
{
   // Map the Qt key release event to the ROOT TCanvas events
   // kKeyUp    = 14
   QWidget::keyReleaseEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::enterEvent(QEvent *e)
{
   // Map the Qt mouse enters widget event to the ROOT TCanvas events
   // kMouseEnter    = 52
   TCanvas *c = Canvas();
   if (c){
      c->HandleInput(kMouseEnter, 0, 0);
   }
   QWidget::enterEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::leaveEvent (QEvent *e)
{
   //  Map the Qt mouse leaves widget event to the ROOT TCanvas events
   // kMouseLeave    = 53
   TCanvas *c = Canvas();
   if (c){
      c->HandleInput(kMouseLeave, 0, 0);
   }
   QWidget::leaveEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::resizeEvent(QResizeEvent *e)
{
   // The widget will be erased and receive a paint event immediately after
   // processing the resize event.
   // No drawing need be (or should be) done inside this handler.
   if (!e) return;
   if (topLevelWidget()->isMinimized())      { fSizeChanged=FALSE; }
   else if (topLevelWidget()->isMaximized ()){
      fSizeChanged=TRUE;
      exitSizeEvent();
      fSizeChanged=TRUE;
   } else {
#ifdef R__QTWIN32
      if (!fPaint)  {
         // real resize event
         fSizeChanged=TRUE;
         stretchWidget(e);
      }
#else
      fSizeChanged=TRUE;
      fPaint = kTRUE;
      exitSizeEvent();
#endif

   }
   if ( autoMask() )
      updateMask();
}
//_____________________________________________________________________________
void TQtWidget::stretchWidget(QResizeEvent * /*s*/)
{
   // Stretch the widget during sizing

   if  (!paintingActive()) {
#ifdef R__QTWIN32
      QPainter painter( this );
      if (!StretchBlt(
         painter.handle(),    // handle of destination device context
         0,           // x-coordinate of upper-left corner of dest. rect.
         0,           // y-coordinate of upper-left corner of dest. rect.
         width(),     // width of destination rectangle
         height(),    // height of destination rectangle
         GetBuffer().handle(), // handle of source device context
         0,           // x-coordinate of upper-left corner of source rectangle
         0,           // y-coordinate of upper-left corner of source rectangle
         GetBuffer().width(),  // width of source rectangle
         GetBuffer().height(), // height of source rectangle
         SRCCOPY      // raster operation code
         )) {
            qSystemWarning("StretchBlt failed!" );
            printf("last error %d\n",GetLastError());
         }
#else
      QPainter pnt(this);
      pnt.drawPixmap(rect(),GetBuffer());
#endif
   }
}
//_____________________________________________________________________________
void TQtWidget::exitSizeEvent ()
{
   // Responce to the "exit size event"

   if (!fSizeChanged) return;
   Refresh();
}

//____________________________________________________________________________
bool TQtWidget::paintFlag(bool mode)
{
   //  Set new fPaint flag
   //  Returns: the previous version of the flag
   bool flag = fPaint;
   fPaint = mode;
   return flag;
}
//____________________________________________________________________________
void TQtWidget::showEvent ( QShowEvent *)
{
   // Custom handler of the Qt show event
   // Non-spontaneous show events are sent to widgets immediately before
   // they are shown.
   // The spontaneous show events of top-level widgets are delivered afterwards.

   if ( fPixmapID.size() != size() )
   {
      fSizeChanged = kTRUE;
      exitSizeEvent();
   }
}

//____________________________________________________________________________
void TQtWidget::paintEvent (QPaintEvent *e)
{
   // Custom handler of the Qt paint event
   // A paint event is a request to repaint all or part of the widget.
   // It can happen as a result of repaint() or update(), or because the widget
   // was obscured and has now been uncovered, or for many other reasons.

#ifdef R__QTWIN32
   if ( fEmbedded && fPixmapID.size() != size() )
   {
      fSizeChanged = kTRUE;
      exitSizeEvent();
      update();
      return;
   }
#endif
   QRect rect = e->rect();
   if ( fPaint && !rect.isEmpty() )
   {
      // fprintf(stderr,"TQtWidget::paintEvent: window = %p; buffer =  %p\n",
      //  (QPaintDevice *)this, (QPaintDevice *)&GetBuffer());
      bitBlt(this, rect.x(),rect.y(),&GetBuffer(),rect.x(), rect.y(), rect.width(), rect.height());
   }
}
//  Layout methods:
//____________________________________________________________________________
void TQtWidget::SetSizeHint (const QSize &size) {
   //  sets the preferred size of the widget.
   fSizeHint = size;
}

//____________________________________________________________________________
QSize TQtWidget::sizeHint () const{
   //  returns the preferred size of the widget.
   return QWidget::sizeHint();
}
//____________________________________________________________________________
QSize TQtWidget::minimumSizeHint () const{
   // returns the smallest size the widget can have.
   return QWidget::minimumSizeHint ();
}
//____________________________________________________________________________
QSizePolicy TQtWidget::sizePolicy () const{
   //  returns a QSizePolicy; a value describing the space requirements
   return QWidget::sizePolicy ();
}
