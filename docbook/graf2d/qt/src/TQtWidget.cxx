// @(#)root/qt:$Id$
// Author: Valeri Fine   23/01/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Definition of TQtWidget class
// "double-buffere widget

#include <qapplication.h>

#if QT_VERSION >= 0x40000
#  include <QFocusEvent>
#  include <QPaintEvent>
#  include <QKeyEvent>
#  include <QShowEvent>
#  include <QResizeEvent>
#  include <QMouseEvent>
#  include <QCustomEvent>
#  include <QImage>
#  include <QDebug>
#endif /* QT_VERSION */

#include "TQtWidget.h"
#include "TQtTimer.h"

#include "TROOT.h"
#include "TEnv.h"
#include "TRint.h"
#include "TSystem.h"
#include "Getline.h"
#include "TGQt.h"
#include "TCanvas.h"
#include "Buttons.h"
#include <qevent.h>
#include <qpainter.h>
#include <qpixmap.h>
#include <qfileinfo.h>

#ifdef R__QTWIN32
// #include "Windows4Root.h"
#include "TWinNTSystem.h"
#include "Win32Constants.h"
#endif

// Class to adjust buffer within the method scope if needed

//___________________________________________________________________
TQtWidgetBuffer::TQtWidgetBuffer(const QWidget *w, bool clear)
: fWidget(w),fBuffer(0), fIsImage(clear)
{
   if (fIsImage) {
      fBuffer = new  QImage(fWidget?fWidget->size():QSize(0,0),QImage::Format_ARGB32_Premultiplied);
//      ((QImage*)fBuffer)->fill(0);
   } else {
      fBuffer = new  QPixmap(fWidget?fWidget->size():QSize(0,0));
   }
}
//___________________________________________________________________
TQtWidgetBuffer::TQtWidgetBuffer(const TQtWidgetBuffer &b)
: fWidget(b.fWidget),fBuffer(0), fIsImage(b.fIsImage)
{
   // Copy ctor. Create the copy and account the new widget size
   if (fWidget && (fWidget->size() != QSize(0,0))) { 
      if (fIsImage) {
         QImage resized =((QImage*)b.fBuffer)->scaled (fWidget->size());
         fBuffer = new  QImage(resized);
      } else {
         QPixmap resized =((QPixmap*) b.fBuffer)->scaled (fWidget->size());
         fBuffer = new  QPixmap(resized);
      }
   }
} 
//___________________________________________________________________
TQtWidgetBuffer:: ~TQtWidgetBuffer()
{
   // dtor
   delete fBuffer;
   fBuffer = 0;
}

//___________________________________________________________________
void TQtWidgetBuffer::Clear()
{
   // Clear the buffer with the transparent color
   if (fBuffer &&  !fIsImage ) {
#ifdef R__WIN32   
         ((QPixmap*)fBuffer)->fill(Qt::transparent);
#else
         QPainter p(fBuffer);
         p.fillRect(QRect(0,0,fBuffer->width(), fBuffer->height())
            ,Qt::transparent);
#endif
   }
}


//___________________________________________________________________

ClassImp(TQtWidget)

////////////////////////////////////////////////////////////////////////////////
//
//  TQtWidget is a QWidget with the QPixmap double buffer
//  It is designed to back the ROOT TCanvasImp class interface  and it can be used
//  as a regular Qt Widget to create Qt-based GUI with the embedded TCanvas objects
//
//           This widget can be used as a Qt "custom widget"
//         to build a custom GUI interfaces with  Qt Designer
//
// The class emits the Qt signals and has Qt public slots
//
//  Public slots:  (Qt)
//
//   virtual void cd();  // make the associated TCanvas the current one (shortcut to TCanvas::cd())
//   virtual void cd(int subpadnumber); // as above - shortcut to Canvas::cd(int subpadnumber)
//   void Disconnect(); // disconnect the QWidget from the ROOT TCanvas (used in the class dtor)
//   void Refresh();    // force the associated TCanvas::Update to be called
//   virtual bool Save(const QString &fileName) const;  // Save the widget image with some ppixmap file
//   virtual bool Save(const char    *fileName) const;
//   virtual bool Save(const QString &fileName,const char *format,int quality=60) const;
//   virtual bool Save(const char    *fileName,const char *format,int quality=60) const;
//
//  signals        (Qt)
//
//    CanvasPainted();  // Signal the TCanvas has been painted onto the screen
//    Saved(bool ok);   // Signal the TCanvas has been saved into the file
//    RootEventProcessed(TObject *selected, unsigned int event, TCanvas *c);
//                      // Signal the Qt mouse/keyboard event has been process by ROOT
//                      // This "signal" is emitted by the enabled mouse events only.
//                      // See: EnableSignalEvents
//                      // ---  DisableSignalEvents
//
//  public methods:
//    The methods below define whether the TQtWidget object emits "RootEventProcessed" Qt signals
//     (By default no  RootEventProcessed Qt signal is emitted )
//     void EnableSignalEvents (UInt_t f)
//     void DisableSignalEvents(UInt_t f),
//         where f is a bitwise OR of the mouse event flags:
//                  kMousePressEvent       // TCanvas processed QEvent mousePressEvent
//                  kMouseMoveEvent        // TCanvas processed QEvent mouseMoveEvent
//                  kMouseReleaseEvent     // TCanvas processed QEvent mouseReleaseEvent
//                  kMouseDoubleClickEvent // TCanvas processed QEvent mouseDoubleClickEvent
//                  kKeyPressEvent         // TCanvas processed QEvent keyPressEvent
//                  kEnterEvent            // TCanvas processed QEvent enterEvent
//                  kLeaveEvent            // TCanvas processed QEvent leaveEvent
//
//  For example to create the custom responce to the mouse crossing TCanvas
//  connect the RootEventProsecced signal with your qt slot:
//
// connect(tQtWidget,SIGNAL(RootEventProcessed(TObject *, unsigned int, TCanvas *))
//          ,this,SLOT(CanvasEvent(TObject *, unsigned int, TCanvas *)));
//  . . .
//
//void qtrootexample1::CanvasEvent(TObject *obj, unsigned int event, TCanvas *)
//{
//  TQtWidget *tipped = (TQtWidget *)sender();
//  const char *objectInfo =
//        obj->GetObjectInfo(tipped->GetEventX(),tipped->GetEventY());
//  QString tipText ="You have ";
//  if  (tipped == tQtWidget1)
//     tipText +="clicked";
//  else
//     tipText +="passed";
//  tipText += " the object <";
//  tipText += obj->GetName();
//  tipText += "> of class ";
//  tipText += obj->ClassName();
//  tipText += " : ";
//  tipText += objectInfo;
//
//   QWhatsThis::display(tipText)
// }
//
////////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
TQtWidget::TQtWidget(QWidget* mother, const char* name, Qt::WFlags f,bool embedded) :
      QWidget(mother,f)
        ,fBits(0),fNeedStretch(false),fCanvas(0),fPixmapID(0),fPixmapScreen(0)
        ,fPaint(TRUE),fSizeChanged(FALSE),fDoubleBufferOn(FALSE),fEmbedded(embedded)
        ,fWrapper(0),fSaveFormat("PNG"),fInsidePaintEvent(false),fOldMousePos(-1,-1)
        ,fIgnoreLeaveEnter(0),fRefreshTimer(0)
{
   if (name && name[0]) setObjectName(name);
   Init() ;
}

//_____________________________________________________________________________
TQtWidget::TQtWidget(QWidget* mother, Qt::WFlags f,bool embedded) :
      QWidget(mother,f)
     ,fBits(0),fNeedStretch(false),fCanvas(0),fPixmapID(0)
     ,fPixmapScreen(0),fPaint(TRUE),fSizeChanged(FALSE)
     ,fDoubleBufferOn(FALSE),fEmbedded(embedded),fWrapper(0),fSaveFormat("PNG")
     ,fInsidePaintEvent(false),fOldMousePos(-1,-1),fIgnoreLeaveEnter(0),fRefreshTimer(0) 
{ setObjectName("tqtwidget"); Init() ;}

//_____________________________________________________________________________
void TQtWidget::Init()
{
  setFocusPolicy(Qt::WheelFocus);
  setAttribute(Qt::WA_NoSystemBackground);
  setAutoFillBackground(false);
  QPalette  p = palette();
  p.setBrush(QPalette::Window, Qt::transparent);
  setPalette(p);

  if (fEmbedded) {
    if (!gApplication) InitRint();
    int minw = 10;
    int minh = 10;
    setMinimumSize(minw,minh);
     Bool_t batch = gROOT->IsBatch();
    if (!batch) gROOT->SetBatch(kTRUE); // to avoid the recursion within TCanvas ctor
    TGQt::RegisterWid(this);
    fCanvas = new TCanvas(objectName().toStdString().c_str(),minw,minh, TGQt::RegisterWid(this));
    gROOT->SetBatch(batch);
    //   schedule the flush operation fCanvas->Flush(); via timer
    Refresh();
  }
  fSizeHint = QWidget::sizeHint();
  setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding));
#ifdef R__QTWIN32
   // Set the application icon for all ROOT widgets
   static HICON rootIcon = 0;
   if (!rootIcon) {
      HICON hIcon = ::LoadIcon(::GetModuleHandle(NULL), MAKEINTRESOURCE(101));
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
   TCanvas *c = 0;
   // to block the double deleting from
   gVirtualX->SelectWindow(-1);
   TGQt::UnRegisterWid(this);
   if (fEmbedded) {
      // one has to set CanvasID = 0 to disconnect things properly.
      c = fCanvas;
      ResetCanvas();
      delete c;
   } else {
      ResetCanvas();
   }
   delete fPixmapID;     fPixmapID = 0;
   delete fPixmapScreen; fPixmapScreen = 0;
}

//______________________________________________________________________________
void TQtWidget::AdjustBufferSize() 
{
   // Adjust the widget buffer size
   TQtWidgetBuffer &buf = SetBuffer();
   QSize s(buf.Width(),buf.Height());
   if ( s != size() )  {
#if 0
       qDebug() << "TQtWidget::AdjustBufferSize(): " 
             << this 
             << s << size();
#endif
      if (fPixmapID) {
         TQtWidgetBuffer *bf = new TQtWidgetBuffer(*fPixmapID);
         delete  fPixmapID;     fPixmapID = bf;
      }
      if (fPixmapScreen) {
         TQtWidgetBuffer *bf = new TQtWidgetBuffer(*fPixmapScreen);
         delete  fPixmapScreen; fPixmapScreen = bf;
      }
   }
}
//_____________________________________________________________________________
TCanvas  *TQtWidget::Canvas()
{
   // Alias for GetCanvas method
   return GetCanvas();
}

//_____________________________________________________________________________
TCanvas   *TQtWidget::Canvas(TQtWidget *widget)
{
    // static: return TCanvas by TQtWidget pointer
   return widget ? widget->Canvas() : 0 ;
}

//_____________________________________________________________________________
TQtWidget *TQtWidget::Canvas(const TCanvas *canvas)
{
   // static: return the TQtWidget backend for TCanvas *canvas object
   return canvas ? Canvas(canvas->GetCanvasID()) : 0;
}
//_____________________________________________________________________________
TQtWidget *TQtWidget::Canvas(Int_t id)
{
   // static: return TQtWidget by TCanvas id
   return dynamic_cast<TQtWidget *>(TGQt::iwid(id));
}

//_____________________________________________________________________________
TApplication *TQtWidget::InitRint( Bool_t /*prompt*/, const char *appClassName, int *argc, char **argv,
          void *options, int numOptions, Bool_t noLogo)
{
   //
   // Instantiate ROOT from within Qt application if needed
   // Return the TRint pointer
   // Most parametrs are passed to TRint class ctor
   //
   // Bool_t prompt = kTRUE;  Instantiate ROOT with ROOT command prompt
   //                 kFALSE; No ROOT prompt. The default for Qt GUI applications
   //
   //  The prompt option can be defined via ROOT parameter file ".rootrc"
   // .rootrc:
   //    . . .
   //  Gui.Prompt   yes
   //
   static int localArgc      =0;
   static char **localArgv  =0;
   if (!gApplication) {
       QStringList args  = QCoreApplication::arguments ();
       localArgc = argc ? *argc : args.size();
       // check the Gui.backend and Factory
       TString guiBackend(gEnv->GetValue("Gui.Backend", "native"));
       guiBackend.ToLower();
       // Enforce Qt-base Gui.Backend and Gui.Factory from within ROOT-based Qt-application
       if (!guiBackend.BeginsWith("qt",TString::kIgnoreCase)) {
         gEnv->SetValue("Gui.Backend", "qt");
       }
       TString guiFactory(gEnv->GetValue("Gui.Factory", "native"));
       guiFactory.ToLower();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,16,0)
       TApplication::NeedGraphicsLibs() ;
#endif
       if (!guiFactory.BeginsWith("qt",TString::kIgnoreCase )){
         // Check for the extention
         char *extLib = gSystem->DynamicPathName("libQtRootGui",kTRUE);
         if (extLib) {
            gEnv->SetValue("Gui.Factory", "qtgui");
         } else {
            gEnv->SetValue("Gui.Factory", "qt");
         }
         delete [] extLib;
       }
       if (!argc && !argv ) {
          localArgv  = new char*[args.size()]; // leaking :-(
          for (int i = 0; i < args.size(); ++i) {
             QString nextarg = args.at(i);
             Int_t nchi = nextarg.length()+1;
             localArgv[i]= new char[nchi]; 
             memcpy(localArgv[i], nextarg.toAscii().constData(),nchi-1);
             localArgv[nchi-1]=0;
          } 
       } else {
         localArgv  = argv;
       }

       TRint *rint = new TRint(appClassName, &localArgc, localArgv, options,numOptions,noLogo);
       // To mimic what TRint::Run(kTRUE) does.
       Int_t prompt= gEnv->GetValue("Gui.Prompt", (Int_t)0);
       if (prompt) {
           Getlinem(kInit, rint->GetPrompt());
       } else {
           // disable the TTermInputHandler too to avoid the crash under X11
           // to get the pure "GUI" application
           TSeqCollection* col = gSystem->GetListOfFileHandlers();
           TIter next(col);
           TFileHandler* o=0;
           while ( ( o=(TFileHandler*) next() ) ) {
              if ( o->GetFd()== 0 ) {
                o->Remove();
                break;
              }
           }
           // Remove Ctrl-C, there will be ROOT prompt anyway
           gSystem->RemoveSignalHandler(rint->GetSignalHandler());
       }
       TQtTimer::Create()->start(0);
   }
   return gApplication;
}

//_____________________________________________________________________________
void TQtWidget::Erase()
{
  // Erases the entire widget and its double buffer
 
  SetBuffer();
//  buf.fill(this,QPoint(0,0));
  if (fPixmapScreen)  fPixmapScreen->Clear();
  if (fPixmapID)      fPixmapID->Clear();
  // erase();
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
  TCanvas *c = fCanvas;
  if (c) {
     c->cd(subpadnumber);
  }
}
//______________________________________________________________________________
void TQtWidget::Disconnect()
{
   // [slot] Disconnect the Qt widget from TCanvas object before deleting
   // to avoid the dead lock
   // one has to set CanvasID = 0 to disconnect things properly.
   fCanvas = 0;
}
//_____________________________________________________________________________
void TQtWidget::Refresh()
{
   // [slot]  to allow Qt signal refreshing the ROOT TCanvas if needed
   // use the permanent single shot timer to eliminate 
   // the redundand refreshing for the sake of the performance
   if (!fRefreshTimer) {
      fRefreshTimer  = new QTimer(this);
      fRefreshTimer->setSingleShot(true);
      fRefreshTimer->setInterval(0);
      connect(fRefreshTimer, SIGNAL(timeout()), this, SLOT(RefreshCB()));
   }
   fRefreshTimer->start();
}
//_____________________________________________________________________________
void TQtWidget::RefreshCB()
{
   // [slot]  to allow Qt signal refreshing the ROOT TCanvas if needed

   TCanvas *c = Canvas();
   if (c) {
      c->Modified();
      c->Resize();
      c->Update();
   }
   if (!fInsidePaintEvent) { update(); }
   else {
      qDebug() << " TQtWidget::Refresh() update inside of paintEvent !!!" << this; 
   }
}
//_____________________________________________________________________________
void TQtWidget::SetCanvas(TCanvas *c) 
{ 
   //  remember my host TCanvas and adopt its name
   fCanvas = c;
   // qDebug() << "TQtWidget::SetCanvas(TCanvas *c)" << fCanvas << fCanvas->GetName() ;
   setObjectName(fCanvas->GetName());
}

//_____________________________________________________________________________
void
TQtWidget::customEvent(QEvent *e)
{
   // The custom response to the special WIN32 events
   // These events are not present with X11 systems
   switch (e->type() - QEvent::User) {
   case kEXITSIZEMOVE:
      { // WM_EXITSIZEMOVE
         fPaint = TRUE;
         exitSizeEvent();
         break;
      }
   case kENTERSIZEMOVE:
      {
         //  WM_ENTERSIZEMOVE
         fSizeChanged=FALSE;
         fPaint = FALSE;
         break;
      }
   case kFORCESIZE:
   default:
      {
         // Force resize
         fPaint       = TRUE;
         fSizeChanged = TRUE;
         exitSizeEvent();
         break;
      }
   };
}
 //_____________________________________________________________________________
void TQtWidget::contextMenuEvent(QContextMenuEvent *e)
{
   // The custom response to the Qt QContextMenuEvent
   // Map QContextMenuEvent to the ROOT kButton3Down = 3  event
   TCanvas *c = Canvas();
   if (e && c && (e->reason() != QContextMenuEvent::Mouse) ) {
      e->accept();
      c->HandleInput(kButton3Down, e->x(), e->y());
   }
}
//_____________________________________________________________________________
void TQtWidget::focusInEvent ( QFocusEvent *e )
{
   // The custom response to the Qt QFocusEvent "in"
   // this imposes an extra protection to avoid TObject interaction with
   // mouse event accidently
   if (!fWrapper && e->gotFocus()) {
      setMouseTracking(TRUE);
   }
}
//_____________________________________________________________________________
void TQtWidget::focusOutEvent ( QFocusEvent *e )
{
   // The custom response to the Qt QFocusEvent "out"
   // this imposes an extra protection to avoid TObject interaction with
   // mouse event accidently
   if (!fWrapper && e->lostFocus()) {
      setMouseTracking(FALSE);
   }
}

//_____________________________________________________________________________
void TQtWidget::mousePressEvent (QMouseEvent *e)
{
   // Map the Qt mouse press button event to the ROOT TCanvas events
   // Mouse events occur when a mouse button is pressed or released inside
   // a widget or when the mouse cursor is moved.

   //    kButton1Down   =  1, kButton2Down   =  2, kButton3Down   =  3,

   EEventType rootButton = kNoEvent;
   Qt::ContextMenuPolicy currentPolicy = contextMenuPolicy();
   fOldMousePos = e->pos();
   TCanvas *c = Canvas();
   if (c && !fWrapper ){
      switch (e->button ())
      {
      case Qt::LeftButton:  rootButton = kButton1Down; break;
      case Qt::RightButton: {
         // respect the QWidget::contextMenuPolicy
         // treat this event as QContextMenuEvent
         if ( currentPolicy == Qt::DefaultContextMenu) {
            e->accept();
            QContextMenuEvent evt(QContextMenuEvent::Other, e->pos() );
            QApplication::sendEvent(this, &evt);
         } else {
            rootButton = kButton3Down;
         }
         break;
      }
      case Qt::MidButton:   rootButton = kButton2Down; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         e->accept();
	 if (rootButton == kButton3Down) {
           bool lastvalue = c->TestBit(kNoContextMenu);
           c->SetBit(kNoContextMenu);
	   c->HandleInput(rootButton, e->x(), e->y());
           c->SetBit(kNoContextMenu, lastvalue);
         } else {
	   c->HandleInput(rootButton, e->x(), e->y());
         }
         EmitSignal(kMousePressEvent);
         return;
      }
   } else {
      e->ignore();
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
   if ( fOldMousePos != e->pos() && fIgnoreLeaveEnter < 2  ) { // workaround of Qt 4.5.x bug
      fOldMousePos = e->pos(); 
      TCanvas *c = Canvas();
      if (c && !fWrapper){
         if (e->buttons() & Qt::LeftButton) { rootButton = kButton1Motion; }
         e->accept();
         c->HandleInput(rootButton, e->x(), e->y());
         EmitSignal(kMouseMoveEvent);
         return;
      } else {
         e->ignore();
      }
   }
   QWidget::mouseMoveEvent(e);
}

//_____________________________________________________________________________
void TQtWidget::mouseReleaseEvent(QMouseEvent * e)
{
   //  Map the Qt mouse button release event to the ROOT TCanvas events
   //   kButton1Up     = 11, kButton2Up     = 12, kButton3Up     = 13

   EEventType rootButton = kNoEvent;
   fOldMousePos = QPoint(-1,-1);
   TCanvas *c = Canvas();
   if (c && !fWrapper){
      switch (e->button())
      {
      case Qt::LeftButton:  rootButton = kButton1Up; break;
      case Qt::RightButton: rootButton = kButton3Up; break;
      case Qt::MidButton:   rootButton = kButton2Up; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         e->accept();
         c->HandleInput(rootButton, e->x(), e->y());
         gPad->Modified();
         EmitSignal(kMouseReleaseEvent);
         return;
      }
   } else {
      e->ignore();
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
   if (c && !fWrapper){
      switch (e->button())
      {
      case Qt::LeftButton:  rootButton = kButton1Double; break;
      case Qt::RightButton: rootButton = kButton3Double; break;
      case Qt::MidButton:   rootButton = kButton2Double; break;
      default: break;
      };
      if (rootButton != kNoEvent) {
         e->accept();
         c->HandleInput(rootButton, e->x(), e->y());
         EmitSignal(kMouseDoubleClickEvent);return;
      }
   }  else {
      e->ignore();
   }
   QWidget::mouseDoubleClickEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::keyPressEvent(QKeyEvent * e)
{
   // Map the Qt key press event to the ROOT TCanvas events
   // kKeyDown  =  4
   TCanvas *c = Canvas();
   if (c && !fWrapper){
      c->HandleInput(kKeyPress, e->text().toStdString().c_str()[0], e->key());
      EmitSignal(kKeyPressEvent);
   } else {
      e->ignore();
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
   if (c && !fIgnoreLeaveEnter && !fWrapper){
      c->HandleInput(kMouseEnter, 0, 0);
      EmitSignal(kEnterEvent);
   }
   QWidget::enterEvent(e);
}
//_____________________________________________________________________________
void TQtWidget::leaveEvent (QEvent *e)
{
   //  Map the Qt mouse leaves widget event to the ROOT TCanvas events
   // kMouseLeave    = 53
   TCanvas *c = Canvas();
   if (c && !fIgnoreLeaveEnter && !fWrapper){
      c->HandleInput(kMouseLeave, 0, 0);
      EmitSignal(kLeaveEvent);
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
         fNeedStretch=true;
      } else {
#else
      {
         fSizeChanged=TRUE;
#if 0
         if (Qt::LeftButton == QApplication::mouseButtons()) 
         {
            fNeedStretch=true;
            fPaint = false;
         } else 
#endif
         {
            fPaint = kTRUE;
            exitSizeEvent();
         }
#endif
      } }
}
//____________________________________________________________________________
void TQtWidget::SetSaveFormat(const char *format)
{
     // Set the default save format for the widget
   fSaveFormat = TGQt::QtFileFormat(format);
}
//____________________________________________________________________________
bool TQtWidget::Save(const char *fileName) const
{
   //
   //  TQtWidget::Save(const QString &fileName) is a public Qt slot.
   //  it saves the double buffer of this object using the default save
   //  format  defined the file extension
   //  If the "fileName" has no extension the "default" format is to be used instead
   //  The deafult format is "PNG".
   //  It can be changed with the TQtWidget::SetSaveFormat method
   //
    return Save(QString(fileName));
}
//____________________________________________________________________________
bool TQtWidget::Save(const QString &fileName) const
{
   //
   //  TQtWidget::Save(const QString &fileName) is a public Qt slot.
   //  it saves the double buffer of this object using the default save
   //  format  defined the file extension
   //  If the "fileName" has no extension the "default" format is to be used instead
   //  The deafult format is "PNG".
   //  It can be changed with the TQtWidget::SetSaveFormat method
   //
   QString fileNameExtension = QFileInfo(fileName).suffix().toUpper();
   QString saveFormat;
   if (fileNameExtension.isEmpty() ) {
      saveFormat = fSaveFormat; // this is default
   } else {
      saveFormat = TGQt::QtFileFormat(fileNameExtension);
   }
   return Save(fileName,saveFormat.toStdString().c_str());
}

//____________________________________________________________________________
bool TQtWidget::Save(const char *fileName,const char *format,int quality)const
{
   return Save(QString(fileName),format,quality);
}
//____________________________________________________________________________
bool TQtWidget::Save(const QString &fileName,const char *format,int quality)const
{
   //  TQtWidget::save is a public Qt slot.
   //  it saves the double buffer of this object using QPixmap facility
   bool Ok = false;
   bool rootFormatFound=kTRUE;
   QString saveType =  TGQt::RootFileFormat(format);
   if (saveType.isEmpty() )  {
      rootFormatFound = false;
      saveType = TGQt::QtFileFormat(format);
   }
   TCanvas *c = GetCanvas();
   if (rootFormatFound && c) {
      c->Print(fileName.toStdString().c_str(),saveType.toStdString().c_str());
      Ok = true;
   } else if (GetOffScreenBuffer()) {
      // Since the "+" is a legal part of the file name and it is used by Onuchin
      // to indicate  the "animation" mode, we have to proceed very carefully
      int dot = fileName.lastIndexOf('.');
      int plus = 0;
      if (dot > -1) {
         plus = fileName.indexOf('+',dot+1);
      }
      QString fln = (plus > -1) ? TGQt::GetNewFileName(fileName.left(plus)) : fileName;
      if (fCanvasDecorator.isNull()) {
         Ok = GetOffScreenBuffer()->save(fln,saveType.toStdString().c_str(),quality);
      } else {
         /// add decoration
      }
   }
   emit ((TQtWidget *)this)->Saved(Ok);
   return Ok;
}
//_____________________________________________________________________________
void  TQtWidget::SetDoubleBuffer(bool on)
{
     // Set the double buffered mode on/off
   if (fDoubleBufferOn != on ) {
      fDoubleBufferOn = on;
      if (on) SetBuffer();
   }
}
//_____________________________________________________________________________
void TQtWidget::stretchWidget(QResizeEvent * /*s*/)
{
   // Stretch the widget during sizing

   if  (!paintingActive() && fPixmapID) {
      QPainter pnt(this);
      pnt.drawPixmap(rect(),*GetOffScreenBuffer());
   }
   fNeedStretch = false;
}
//_____________________________________________________________________________
void TQtWidget::exitSizeEvent ()
{
   // Response to the "exit size event"

   if (!fSizeChanged ) return;
   {
      AdjustBufferSize();
   }
   //Refresh();
   TCanvas *c = Canvas();
   if (c)   c->Resize();
   // One more time to catch the last size
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
   TQtWidgetBuffer &buf = SetBuffer();
   QSize s(buf.Width(),buf.Height());
   if (s != size() )
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
   fInsidePaintEvent = true;
   if (fNeedStretch) {
      stretchWidget((QResizeEvent *)0);
   } else {
#ifdef R__QTWIN32
      TQtWidgetBuffer &buf = SetBuffer();
      QSize s(buf.Width(),buf.Height());
      if ( fEmbedded && (s != size()) )
      {
         fSizeChanged = kTRUE;
         exitSizeEvent();
         fInsidePaintEvent = false;
         return;
      }
#endif
      QRegion region = e->region();
      if ( ( fPaint && !region.isEmpty() ) )
      {
         //  fprintf(stderr,"TQtWidget::paintEvent: window = %p; buffer =  %p\n",
         //  (QPaintDevice *)this, (QPaintDevice *)&GetBuffer());
         //  qDebug() << "1. TQtWidget::paintEvent this =" << (QPaintDevice *)this  << " buffer = " << fPixmapID << "redirected = " << QPainter::redirected(this)
         //    <<" IsDoubleBuffered()=" << IsDoubleBuffered() ;
         // qDebug() << "2. TQtWidget::paintEvent this =" << (QPaintDevice *)this  << " buffer = " << fPixmapID << " IsDoubleBuffered()=" << IsDoubleBuffered() ;
         QPainter screen(this);
         screen.setClipRegion(region);
         // paint the the TCanvas double buffer
         if (fPixmapID)  screen.drawPixmap(0,0,*GetOffScreenBuffer());
         if (!fCanvasDecorator.isNull()) fCanvasDecorator->paintEvent(screen,e);
      }
   }
   fInsidePaintEvent = false;
}
//  Layout methods:
//____________________________________________________________________________
void TQtWidget::SetSizeHint (const QSize &sz) {
   //  sets the preferred size of the widget.
   fSizeHint = sz;
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
//____________________________________________________________________________
void  TQtWidget::EmitTestedSignal()
{
   TCanvas *c        = GetCanvas();
   TObject *selected = GetSelected();
   UInt_t evt      = GetEvent();
   emit RootEventProcessed(selected, evt, c);
}
//____________________________________________________________________________
void  TQtWidget::SetBit(UInt_t f, Bool_t set)
{
   // Set or unset the user status bits as specified in f.

   if (set)
      SetBit(f);
   else
      ResetBit(f);
}
//____________________________________________________________________________
TQtWidgetBuffer  &TQtWidget::SetBuffer() {
   // Create (if needed) and return the buffer
   TQtWidgetBuffer *buf = 0;
   if (IsDoubleBuffered() ) {
      if (!fPixmapID) fPixmapID = new TQtWidgetBuffer(this);
      buf = fPixmapID;
   } else {
      if (!fPixmapScreen) fPixmapScreen = new TQtWidgetBuffer(this,true);
      // qDebug() << "TQtWidget::SetBuffer() " << fPixmapScreen;
      buf = fPixmapScreen;
   }
   return  *buf;
}
//______________________________________________________________________________
QPixmap  *TQtWidget::GetOffScreenBuffer()  const { 
   //  return the current widget buffer;
   return fPixmapID ? (QPixmap  *)fPixmapID->Buffer():0;
}
