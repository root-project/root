// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "qevent.h"
#include "qdialog.h"
#include "qpushbutton.h"
#include "qlabel.h"
#include "qpainter.h"
#if  (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
# include "qnamespace.h"
using namespace Qt;
# include "q3dragobject.h"
typedef Q3TextDrag QTextDrag;
#endif

#include "TQRootCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TQCanvasMenu.h"

ClassImp(TQRootCanvas);

////////////////////////////////////////////////////////////////////////////////
/// set defaults

TQRootCanvas::TQRootCanvas( QWidget *parent, const char *name, TCanvas *c )
  : QWidget( parent, name ,WRepaintNoErase | WResizeNoErase ),
        fNeedResize(kTRUE)
{
   setUpdatesEnabled( kTRUE );
   setMouseTracking(kTRUE);

   //  setBackgroundMode( NoBackground );
   setFocusPolicy( TabFocus );
   setCursor( Qt::crossCursor );

   fTabWin = 0;
   // add the Qt::WinId to TGX11 interface
   fWid=gVirtualX->AddWindow((ULong_t)winId(),100,30);
   if (c == 0) {
      fIsCanvasOwned = kTRUE;
      // Window_t win=gVirtualX->GetWindowID(fWid);
      fCanvas=new TCanvas(name,width(),height(),fWid);
   }
   else {
      fIsCanvasOwned= kFALSE;
      fCanvas=c;
   }
   // create the context menu
   fContextMenu = new TQCanvasMenu( parent, fCanvas );

   // test here all the events sent to the QWidget
   // has a parent widget then install filter
   if ( parent ) {
      parent->installEventFilter( this );
      fParent = parent;
   }
   else
      fParent=0;

   // drag and drop suppurt  (M. Al-Turany)
   setAcceptDrops(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// set defaults

TQRootCanvas::TQRootCanvas( QWidget *parent, QWidget* tabWin, const char *name, TCanvas *c )
  : QWidget( tabWin, name ,WRepaintNoErase | WResizeNoErase ),
    fNeedResize(kTRUE)
{
   setUpdatesEnabled( kTRUE );
   setMouseTracking(kTRUE);

   setFocusPolicy( TabFocus );
   setCursor( Qt::crossCursor );

   fTabWin = 0;
   // add the Qt::WinId to TGX11 interface
   fWid=gVirtualX->AddWindow((ULong_t)winId(),100,30);
   if (c == 0) {
      fIsCanvasOwned = kTRUE;
      fCanvas=new TCanvas(name,width(),height(),fWid);
   }
   else {
      fIsCanvasOwned= kFALSE;
      fCanvas=c;
   }
   // create the context menu
   fContextMenu = new TQCanvasMenu( parent, tabWin, fCanvas );

   // test here all the events sent to the QWidget
   // has a parent widget then install filter
   if ( parent ) {
      parent->installEventFilter( this );
      fParent = parent;
   }
   else
      fParent=0;

   if ( tabWin )
      fTabWin = tabWin;

   // drag and drop suppurt  (M. Al-Turany)
   setAcceptDrops(TRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse move event.

void TQRootCanvas::mouseMoveEvent(QMouseEvent *e)
{
   if (fCanvas) {
      if (e->state() & LeftButton) {
         fCanvas->HandleInput(kButton1Motion, e->x(), e->y());
      }
      else {
         fCanvas->HandleInput(kMouseMotion, e->x(), e->y());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse button press event.

void TQRootCanvas::mousePressEvent( QMouseEvent *e )
{
   TPad *pad=0;
   TObjLink *pickobj=0;
   TObject *selected=0;
   Int_t px=e->x();
   Int_t py=e->y();
   TString selectedOpt;
   switch (e->button()) {
      case LeftButton :
         fCanvas->HandleInput(kButton1Down, e->x(), e->y());
         break;
      case RightButton :
         selected=fCanvas->GetSelected();
         pad = fCanvas->Pick(px, py, pickobj);
         if (pad) {
            if (!pickobj) {
               fCanvas->SetSelected(pad); selected=pad;
               selectedOpt = "";
            }
            else {
               if (!selected) {
                  selected    = pickobj->GetObject();
                  selectedOpt = pickobj->GetOption();
               }
            }
            pad->cd();
            fCanvas->SetSelectedPad(pad);
         }
         gROOT->SetSelectedPrimitive(selected);
         fContextMenu->Popup(selected, gPad->AbsPixeltoX(gPad->GetEventX()),
                             gPad->AbsPixeltoY(gPad->GetEventY()), e);

         break;
      case MidButton :
         pad = fCanvas->Pick(px, py, pickobj);  // get the selected pad and emit a Qt-Signal
         emit SelectedPadChanged(pad);          // that inform the Qt-world that tha pad is changed
                                                // and give the pointer to the new pad as argument
                                                // of the signal (M. Al-Turany)
         fCanvas->HandleInput(kButton2Down, e->x(), e->y());

         break;

      case  NoButton :
         break;
      default:
         break;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse button release event.

void TQRootCanvas::mouseReleaseEvent( QMouseEvent *e )
{
   switch (e->button()) {
      case LeftButton :
         fCanvas->HandleInput(kButton1Up, e->x(), e->y());
         break;
      case RightButton :
         fCanvas->HandleInput(kButton3Up, e->x(), e->y());
         break;
      case MidButton :
         fCanvas->HandleInput(kButton2Up, e->x(), e->y());
         break;
      case  NoButton :
         break;
      default:
         break;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse double click event.

void TQRootCanvas::mouseDoubleClickEvent( QMouseEvent *e )
{
   switch (e->button()) {
      case LeftButton :
         fCanvas->HandleInput(kButton1Double, e->x(), e->y());
         break;
      case RightButton :
         fCanvas->HandleInput(kButton3Double, e->x(), e->y());
         break;
      case MidButton :
         fCanvas->HandleInput(kButton2Double, e->x(), e->y());
         break;
      case  NoButton :
         break;
      default:
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Call QWidget resize and inform the ROOT Canvas.

void TQRootCanvas::resizeEvent( QResizeEvent *e )
{
   QWidget::resizeEvent( e );
   fNeedResize=kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle paint event of Qt.

void TQRootCanvas::paintEvent( QPaintEvent * )
{
   if (fCanvas) {
      QPainter p;
      p.begin( this);
      p.end();
      if (fNeedResize) {
         fCanvas->Resize();
         fNeedResize=kFALSE;
      }
      fCanvas->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Handle leave event.

void TQRootCanvas::leaveEvent( QEvent * /*e*/ )
{
   if (fCanvas) fCanvas->HandleInput(kMouseLeave, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Filtering of QWidget Events
/// for ressource management

Bool_t TQRootCanvas ::eventFilter( QObject *o, QEvent *e )
{
   if ( e->type() == QEvent::Close) {  // close
      if (fCanvas && (fIsCanvasOwned== kFALSE) ) {
         delete fCanvas;
         fCanvas=0;
      }
      if ( e->type() == QEvent::ChildRemoved ) {  // child is removed
      }
      return FALSE;                        // eat event
   }

   if ( e->type() == QEvent::Destroy) {  // destroy
      return FALSE;
   }

   if ( e->type() == QEvent::Paint) {  // Paint
      return FALSE;
   }
   if ( e->type() == QEvent::Move) {  // Paint
      return FALSE;
   }

   // standard event processing
   return QWidget::eventFilter( o, e );
}

////////////////////////////////////// drag and drop support

////////////////////////////////////////////////////////////////////////////////
/// Entering a drag event.

void TQRootCanvas::dragEnterEvent( QDragEnterEvent *e )
{
   if ( QTextDrag::canDecode(e))
      e->accept();
}

////////////////////////////////////////////////////////////////////////////////
/// Start a drop, for now only histogram objects can be drwon by droping.

void TQRootCanvas::dropEvent( QDropEvent *Event )
{
   QString str;
   if ( QTextDrag::decode( Event, str ) ) {
      TObject *dragedObject = gROOT->FindObject(str);
      QPoint Pos = Event->pos();
      TObject *object=0;
      TPad *pad = fCanvas->Pick(Pos.x(), Pos.y(), object);
      if (dragedObject!=0) {
         if (dragedObject->InheritsFrom("TH1")) {
            pad->cd();
            dragedObject->Draw();
            pad->Update();
         }
      }
      else
         std::cout << "object " <<
#if  (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
         str.data()
#else
         str
#endif
         <<  " not found by ROOT" << std::endl;
   }
}

/////////////////////////////////////End Drag and drop Support (Mohammad Al-Turany)

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper

void TQRootCanvas::cd(Int_t subpadnumber)
{
   fCanvas->cd(subpadnumber);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

void TQRootCanvas::Browse(TBrowser *b)
{
   fCanvas->Browse(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

void TQRootCanvas::Clear(Option_t *option)
{
   fCanvas->Clear(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

void TQRootCanvas::Close(Option_t *option)
{
   fCanvas->Close(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

void TQRootCanvas::Draw(Option_t *option)
{
   fCanvas->Draw(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

TObject *TQRootCanvas::DrawClone(Option_t *option)
{
   return  fCanvas->DrawClone(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

TObject *TQRootCanvas::DrawClonePad()
{
   return  fCanvas->DrawClonePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Just a wrapper.

void TQRootCanvas::EditorBar()
{
   fCanvas->EditorBar();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::EnterLeave(TPad *prevSelPad, TObject *prevSelObj)
{
   fCanvas->EnterLeave(prevSelPad, prevSelObj);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::FeedbackMode(Bool_t set)
{
   fCanvas->FeedbackMode(set);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Flush()
{
   fCanvas->Flush();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::UseCurrentStyle()
{
   fCanvas->UseCurrentStyle();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::ForceUpdate()
{
   fCanvas->ForceUpdate() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

const char *TQRootCanvas::GetDISPLAY()
{
   return fCanvas->GetDISPLAY() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TContextMenu *TQRootCanvas::GetContextMenu()
{
   return  fCanvas->GetContextMenu() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetDoubleBuffer()
{
   return fCanvas->GetDoubleBuffer();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetEvent()
{
   return fCanvas->GetEvent();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetEventX()
{
   return fCanvas->GetEventX() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetEventY()
{
   return fCanvas->GetEventY() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Color_t TQRootCanvas::GetHighLightColor()
{
   return fCanvas->GetHighLightColor() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TVirtualPad *TQRootCanvas::GetPadSave()
{
   return fCanvas->GetPadSave();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TObject *TQRootCanvas::GetSelected()
{
   return fCanvas->GetSelected() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Option_t *TQRootCanvas::GetSelectedOpt()
{
   return fCanvas->GetSelectedOpt();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TVirtualPad *TQRootCanvas::GetSelectedPad()
{
   return fCanvas->GetSelectedPad();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::GetShowEventStatus()
{
   return fCanvas->GetShowEventStatus() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::GetAutoExec()
{
   return fCanvas->GetAutoExec();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Size_t TQRootCanvas::GetXsizeUser()
{
   return fCanvas->GetXsizeUser();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Size_t TQRootCanvas::GetYsizeUser()
{
   return fCanvas->GetYsizeUser();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Size_t TQRootCanvas::GetXsizeReal()
{
   return fCanvas->GetXsizeReal();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Size_t TQRootCanvas::GetYsizeReal()
{
   return fCanvas->GetYsizeReal();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetCanvasID()
{
   return fCanvas->GetCanvasID();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetWindowTopX()
{
   return fCanvas->GetWindowTopX();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Int_t TQRootCanvas::GetWindowTopY()
{
   return fCanvas->GetWindowTopY();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

UInt_t TQRootCanvas::GetWindowWidth()
{
   return fCanvas->GetWindowWidth() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

UInt_t TQRootCanvas::GetWindowHeight()
{
   return fCanvas->GetWindowHeight();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

UInt_t TQRootCanvas::GetWw()
{
   return fCanvas->GetWw();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

UInt_t TQRootCanvas::GetWh()
{
   return fCanvas->GetWh() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::GetCanvasPar(Int_t &wtopx, Int_t &wtopy, UInt_t &ww, UInt_t &wh)
{
   fCanvas->GetCanvasPar(wtopx, wtopy, ww, wh);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::HandleInput(EEventType button, Int_t x, Int_t y)
{
   fCanvas->HandleInput(button, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::HasMenuBar()
{
   return fCanvas->HasMenuBar() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Iconify()
{
   fCanvas->Iconify();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::IsBatch()
{
   return fCanvas->IsBatch() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::IsRetained()
{
   return fCanvas->IsRetained();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::ls(Option_t *option)
{
   fCanvas->ls(option);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::MoveOpaque(Int_t set)
{
   fCanvas->MoveOpaque(set);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::OpaqueMoving()
{
   return fCanvas->OpaqueMoving();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

Bool_t TQRootCanvas::OpaqueResizing()
{
   return fCanvas->OpaqueResizing();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Paint(Option_t *option)
{
   fCanvas->Paint(option);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TPad *TQRootCanvas::Pick(Int_t px, Int_t py, TObjLink *&pickobj)
{
   return fCanvas->Pick(px, py, pickobj);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

TPad *TQRootCanvas::Pick(Int_t px, Int_t py, TObject *prevSelObj)
{
   return fCanvas->Pick(px, py, prevSelObj);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Resize(Option_t *option)
{
   fCanvas->Resize(option);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::ResizeOpaque(Int_t set)
{
   fCanvas->ResizeOpaque(set);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SaveSource(const char *filename, Option_t *option)
{
   fCanvas->SaveSource(filename, option);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetCursor(ECursor cursor)
{
   fCanvas->SetCursor(cursor);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetDoubleBuffer(Int_t mode)
{
   fCanvas->SetDoubleBuffer(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetWindowPosition(Int_t x, Int_t y)
{
   fCanvas->SetWindowPosition(x, y) ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetWindowSize(UInt_t ww, UInt_t wh)
{
   fCanvas->SetWindowSize(ww,wh) ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   fCanvas->SetCanvasSize(ww, wh);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetHighLightColor(Color_t col)
{
   fCanvas->SetHighLightColor(col);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetSelected(TObject *obj)
{
   fCanvas->SetSelected(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetSelectedPad(TPad *pad)
{
   fCanvas->SetSelectedPad(pad);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Show()
{
   fCanvas->Show() ;
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Size(Float_t xsizeuser, Float_t ysizeuser)
{
   fCanvas->Size(xsizeuser, ysizeuser);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetBatch(Bool_t batch)
{
   fCanvas->SetBatch(batch);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetRetained(Bool_t retained)
{
   fCanvas->SetRetained(retained);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::SetTitle(const char *title)
{
   fCanvas->SetTitle(title);
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::ToggleEventStatus()
{
   fCanvas->ToggleEventStatus();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::ToggleAutoExec()
{
   fCanvas->ToggleAutoExec();
}

////////////////////////////////////////////////////////////////////////////////
/// just a wrapper

void TQRootCanvas::Update()
{
   fCanvas->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Close.

void  TQRootCanvas::closeEvent( QCloseEvent * e)
{
   if ( fIsCanvasOwned ) {
      delete fCanvas;
      fCanvas = 0;
   }
   e->accept();
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TQRootCanvas::~TQRootCanvas()
{
   if (fContextMenu) {
      delete fContextMenu;
      fContextMenu=0;
   }
   if ( fIsCanvasOwned && fCanvas ) {
      delete fCanvas;
      fCanvas = 0;
   }
}



