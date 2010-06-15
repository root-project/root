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

ClassImp(TQRootCanvas)

//______________________________________________________________________________
TQRootCanvas::TQRootCanvas( QWidget *parent, const char *name, TCanvas *c )
  : QWidget( parent, name ,WRepaintNoErase | WResizeNoErase ),
        fNeedResize(kTRUE)
{
   // set defaults
   setUpdatesEnabled( kTRUE );
   setMouseTracking(kTRUE);

   //  setBackgroundMode( NoBackground );
   setFocusPolicy( TabFocus );
   setCursor( Qt::crossCursor );

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

//______________________________________________________________________________
TQRootCanvas::TQRootCanvas( QWidget *parent, QWidget* tabWin, const char *name, TCanvas *c )
  : QWidget( tabWin, name ,WRepaintNoErase | WResizeNoErase ),
    fNeedResize(kTRUE)
{
   // set defaults
   setUpdatesEnabled( kTRUE );
   setMouseTracking(kTRUE);

   setFocusPolicy( TabFocus );
   setCursor( Qt::crossCursor );

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

//______________________________________________________________________________
void TQRootCanvas::mouseMoveEvent(QMouseEvent *e)
{
   // Handle mouse move event.

   if (fCanvas) {
      if (e->state() & LeftButton) {
         fCanvas->HandleInput(kButton1Motion, e->x(), e->y());
      }
      else {
         fCanvas->HandleInput(kMouseMotion, e->x(), e->y());
      }
   }
}

//______________________________________________________________________________
void TQRootCanvas::mousePressEvent( QMouseEvent *e )
{
   // Handle mouse button press event.

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

//______________________________________________________________________________
void TQRootCanvas::mouseReleaseEvent( QMouseEvent *e )
{
   // Handle mouse button release event.

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

//______________________________________________________________________________
void TQRootCanvas::mouseDoubleClickEvent( QMouseEvent *e )
{
   // Handle mouse double click event.

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

//______________________________________________________________________________
void TQRootCanvas::resizeEvent( QResizeEvent *e )
{
   // Call QWidget resize and inform the ROOT Canvas.

   QWidget::resizeEvent( e );
   fNeedResize=kTRUE;
}

//______________________________________________________________________________
void TQRootCanvas::paintEvent( QPaintEvent * )
{
   // Handle paint event of Qt.

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

//______________________________________________________________________________
void TQRootCanvas::leaveEvent( QEvent * /*e*/ )
{
   // Handle leave event.

   if (fCanvas) fCanvas->HandleInput(kMouseLeave, 0, 0);
}

//______________________________________________________________________________
Bool_t TQRootCanvas ::eventFilter( QObject *o, QEvent *e )
{
   // Filtering of QWidget Events
   // for ressource management

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

//______________________________________________________________________________
void TQRootCanvas::dragEnterEvent( QDragEnterEvent *e )
{
   // Entering a drag event.

   if ( QTextDrag::canDecode(e))
      e->accept();
}

//______________________________________________________________________________
void TQRootCanvas::dropEvent( QDropEvent *Event )
{
   // Start a drop, for now only histogram objects can be drwon by droping.

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
	cout << "object " << 
#if  (QT_VERSION > 0x039999) // Added by cholm@nbi.dk - for Qt 4
	  str.data() 
#else 
	  str
#endif
	     <<  " not found by ROOT" << endl;
   }
}

/////////////////////////////////////End Drag and drop Support (Mohammad Al-Turany)

//______________________________________________________________________________
void TQRootCanvas::cd(Int_t subpadnumber)
{
   // Just a wrapper

   fCanvas->cd(subpadnumber);
}

//______________________________________________________________________________
void TQRootCanvas::Browse(TBrowser *b)
{
   // Just a wrapper.

   fCanvas->Browse(b);
}

//______________________________________________________________________________
void TQRootCanvas::Clear(Option_t *option)
{
   // Just a wrapper.

   fCanvas->Clear(option);
}

//______________________________________________________________________________
void TQRootCanvas::Close(Option_t *option)
{
   // Just a wrapper.

   fCanvas->Close(option);
}

//______________________________________________________________________________
void TQRootCanvas::Draw(Option_t *option)
{
   // Just a wrapper.

   fCanvas->Draw(option);
}

//______________________________________________________________________________
TObject *TQRootCanvas::DrawClone(Option_t *option)
{
   // Just a wrapper.

   return  fCanvas->DrawClone(option);
}

//______________________________________________________________________________
TObject *TQRootCanvas::DrawClonePad()
{
   // Just a wrapper.

   return  fCanvas->DrawClonePad();
}

//______________________________________________________________________________
void TQRootCanvas::EditorBar()
{
   // Just a wrapper.

   fCanvas->EditorBar();
}

//______________________________________________________________________________
void TQRootCanvas::EnterLeave(TPad *prevSelPad, TObject *prevSelObj)
{
   // just a wrapper
   fCanvas->EnterLeave(prevSelPad, prevSelObj);
}

//______________________________________________________________________________
void TQRootCanvas::FeedbackMode(Bool_t set)
{
   // just a wrapper
   fCanvas->FeedbackMode(set);
}

//______________________________________________________________________________
void TQRootCanvas::Flush()
{
   // just a wrapper
   fCanvas->Flush();
}

//______________________________________________________________________________
void TQRootCanvas::UseCurrentStyle()
{
   // just a wrapper
   fCanvas->UseCurrentStyle();
}

//______________________________________________________________________________
void TQRootCanvas::ForceUpdate()
{
   // just a wrapper
   fCanvas->ForceUpdate() ;
}

//______________________________________________________________________________
const char *TQRootCanvas::GetDISPLAY()
{
   // just a wrapper
   return fCanvas->GetDISPLAY() ;
}

//______________________________________________________________________________
TContextMenu *TQRootCanvas::GetContextMenu()
{
   // just a wrapper
   return  fCanvas->GetContextMenu() ;
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetDoubleBuffer()
{
   // just a wrapper
   return fCanvas->GetDoubleBuffer();
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetEvent()
{
   // just a wrapper
   return fCanvas->GetEvent();
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetEventX()
{
   // just a wrapper
   return fCanvas->GetEventX() ;
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetEventY()
{
   // just a wrapper
   return fCanvas->GetEventY() ;
}

//______________________________________________________________________________
Color_t TQRootCanvas::GetHighLightColor()
{
   // just a wrapper
   return fCanvas->GetHighLightColor() ;
}

//______________________________________________________________________________
TVirtualPad *TQRootCanvas::GetPadSave()
{
   // just a wrapper
   return fCanvas->GetPadSave();
}

//______________________________________________________________________________
TObject *TQRootCanvas::GetSelected()
{
   // just a wrapper
   return fCanvas->GetSelected() ;
}

//______________________________________________________________________________
Option_t *TQRootCanvas::GetSelectedOpt()
{
   // just a wrapper
   return fCanvas->GetSelectedOpt();
}

//______________________________________________________________________________
TVirtualPad *TQRootCanvas::GetSelectedPad()
{
   // just a wrapper
   return fCanvas->GetSelectedPad();
}

//______________________________________________________________________________
Bool_t TQRootCanvas::GetShowEventStatus()
{
   // just a wrapper
   return fCanvas->GetShowEventStatus() ;
}

//______________________________________________________________________________
Bool_t TQRootCanvas::GetAutoExec()
{
   // just a wrapper
   return fCanvas->GetAutoExec();
}

//______________________________________________________________________________
Size_t TQRootCanvas::GetXsizeUser()
{
   // just a wrapper
   return fCanvas->GetXsizeUser();
}

//______________________________________________________________________________
Size_t TQRootCanvas::GetYsizeUser()
{
   // just a wrapper
   return fCanvas->GetYsizeUser();
}

//______________________________________________________________________________
Size_t TQRootCanvas::GetXsizeReal()
{
   // just a wrapper
   return fCanvas->GetXsizeReal();
}

//______________________________________________________________________________
Size_t TQRootCanvas::GetYsizeReal()
{
   // just a wrapper
   return fCanvas->GetYsizeReal();
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetCanvasID()
{
   // just a wrapper
   return fCanvas->GetCanvasID();
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetWindowTopX()
{
   // just a wrapper
   return fCanvas->GetWindowTopX();
}

//______________________________________________________________________________
Int_t TQRootCanvas::GetWindowTopY()
{
   // just a wrapper
   return fCanvas->GetWindowTopY();
}

//______________________________________________________________________________
UInt_t TQRootCanvas::GetWindowWidth()
{
   // just a wrapper
   return fCanvas->GetWindowWidth() ;
}

//______________________________________________________________________________
UInt_t TQRootCanvas::GetWindowHeight()
{
   // just a wrapper
   return fCanvas->GetWindowHeight();
}

//______________________________________________________________________________
UInt_t TQRootCanvas::GetWw()
{
   // just a wrapper
   return fCanvas->GetWw();
}

//______________________________________________________________________________
UInt_t TQRootCanvas::GetWh()
{
   // just a wrapper
   return fCanvas->GetWh() ;
}

//______________________________________________________________________________
void TQRootCanvas::GetCanvasPar(Int_t &wtopx, Int_t &wtopy, UInt_t &ww, UInt_t &wh)
{
   // just a wrapper
   fCanvas->GetCanvasPar(wtopx, wtopy, ww, wh);
}

//______________________________________________________________________________
void TQRootCanvas::HandleInput(EEventType button, Int_t x, Int_t y)
{
   // just a wrapper
   fCanvas->HandleInput(button, x, y);
}

//______________________________________________________________________________
Bool_t TQRootCanvas::HasMenuBar()
{
   // just a wrapper
   return fCanvas->HasMenuBar() ;
}

//______________________________________________________________________________
void TQRootCanvas::Iconify()
{
   // just a wrapper
   fCanvas->Iconify();
}

//______________________________________________________________________________
Bool_t TQRootCanvas::IsBatch()
{
   // just a wrapper
   return fCanvas->IsBatch() ;
}

//______________________________________________________________________________
Bool_t TQRootCanvas::IsRetained()
{
   // just a wrapper
   return fCanvas->IsRetained();
}

//______________________________________________________________________________
void TQRootCanvas::ls(Option_t *option)
{
   // just a wrapper
   fCanvas->ls(option);
}

//______________________________________________________________________________
void TQRootCanvas::MoveOpaque(Int_t set)
{
   // just a wrapper
   fCanvas->MoveOpaque(set);
}

//______________________________________________________________________________
Bool_t TQRootCanvas::OpaqueMoving()
{
   // just a wrapper
   return fCanvas->OpaqueMoving();
}

//______________________________________________________________________________
Bool_t TQRootCanvas::OpaqueResizing()
{
   // just a wrapper
   return fCanvas->OpaqueResizing();
}

//______________________________________________________________________________
void TQRootCanvas::Paint(Option_t *option)
{
   // just a wrapper
   fCanvas->Paint(option);
}

//______________________________________________________________________________
TPad *TQRootCanvas::Pick(Int_t px, Int_t py, TObjLink *&pickobj)
{
   // just a wrapper
   return fCanvas->Pick(px, py, pickobj);
}

//______________________________________________________________________________
TPad *TQRootCanvas::Pick(Int_t px, Int_t py, TObject *prevSelObj)
{
   // just a wrapper
   return fCanvas->Pick(px, py, prevSelObj);
}

//______________________________________________________________________________
void TQRootCanvas::Resize(Option_t *option)
{
   // just a wrapper
   fCanvas->Resize(option);
}

//______________________________________________________________________________
void TQRootCanvas::ResizeOpaque(Int_t set)
{
   // just a wrapper
   fCanvas->ResizeOpaque(set);
}

//______________________________________________________________________________
void TQRootCanvas::SaveSource(const char *filename, Option_t *option)
{
   // just a wrapper
   fCanvas->SaveSource(filename, option);
}

//______________________________________________________________________________
void TQRootCanvas::SetCursor(ECursor cursor)
{
   // just a wrapper
   fCanvas->SetCursor(cursor);
}

//______________________________________________________________________________
void TQRootCanvas::SetDoubleBuffer(Int_t mode)
{
   // just a wrapper
   fCanvas->SetDoubleBuffer(mode);
}

//______________________________________________________________________________
void TQRootCanvas::SetWindowPosition(Int_t x, Int_t y)
{
   // just a wrapper
   fCanvas->SetWindowPosition(x, y) ;
}

//______________________________________________________________________________
void TQRootCanvas::SetWindowSize(UInt_t ww, UInt_t wh)
{
   // just a wrapper
   fCanvas->SetWindowSize(ww,wh) ;
}

//______________________________________________________________________________
void TQRootCanvas::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   // just a wrapper
   fCanvas->SetCanvasSize(ww, wh);
}

//______________________________________________________________________________
void TQRootCanvas::SetHighLightColor(Color_t col)
{
   // just a wrapper
   fCanvas->SetHighLightColor(col);
}

//______________________________________________________________________________
void TQRootCanvas::SetSelected(TObject *obj)
{
   // just a wrapper
   fCanvas->SetSelected(obj);
}

//______________________________________________________________________________
void TQRootCanvas::SetSelectedPad(TPad *pad)
{
   // just a wrapper
   fCanvas->SetSelectedPad(pad);
}

//______________________________________________________________________________
void TQRootCanvas::Show()
{
   // just a wrapper
   fCanvas->Show() ;
}

//______________________________________________________________________________
void TQRootCanvas::Size(Float_t xsizeuser, Float_t ysizeuser)
{
   // just a wrapper
   fCanvas->Size(xsizeuser, ysizeuser);
}

//______________________________________________________________________________
void TQRootCanvas::SetBatch(Bool_t batch)
{
   // just a wrapper
   fCanvas->SetBatch(batch);
}

//______________________________________________________________________________
void TQRootCanvas::SetRetained(Bool_t retained)
{
  // just a wrapper
   fCanvas->SetRetained(retained);
}

//______________________________________________________________________________
void TQRootCanvas::SetTitle(const char *title)
{
   // just a wrapper
   fCanvas->SetTitle(title);
}

//______________________________________________________________________________
void TQRootCanvas::ToggleEventStatus()
{
   // just a wrapper
   fCanvas->ToggleEventStatus();
}

//______________________________________________________________________________
void TQRootCanvas::ToggleAutoExec()
{
   // just a wrapper
   fCanvas->ToggleAutoExec();
}

//______________________________________________________________________________
void TQRootCanvas::Update()
{
   // just a wrapper
   fCanvas->Update();
}

//______________________________________________________________________________
void  TQRootCanvas::closeEvent( QCloseEvent * e)
{
   // Close.

   if ( fIsCanvasOwned ) {
      delete fCanvas;
      fCanvas = 0;
   }
   e->accept();
   return;
}

//______________________________________________________________________________
TQRootCanvas::~TQRootCanvas()
{
   // dtor

   if (fContextMenu) {
      delete fContextMenu;
      fContextMenu=0;
   }
   if ( fIsCanvasOwned && fCanvas ) {
      delete fCanvas;
      fCanvas = 0;
   }
}



