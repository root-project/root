// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TGQt.cxx,v 1.62 2004/07/21 21:55:42 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*-*-*The   T G Q t  class*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
//*-*
//*-*  Basic interface to the Qt graphics system
//*-*
//*-*  This code was initially developped in the context of HIGZ and PAW
//*-*  by Valery Fine to port the package X11INT (by Olivie Couet)
//*-*  to Windows NT.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#ifdef R__QTWIN32
#include <process.h>
#endif

#include <assert.h>

//  Qt include files
  
#include <qapplication.h> 
#if (QT_VERSION < 0x030200)
# include <qthread.h> 
#endif
#include <qwidget.h>
#include <qpixmap.h>
#include <qcursor.h>
#include <qpen.h>
#include <qpicture.h>
#include <qdesktopwidget.h>
#include <qimage.h>
#include <qfontmetrics.h>
#include <qdialog.h>
#include <qlineedit.h>
#include <qfileinfo.h> 
#include <qtextcodec.h> 

#include "TROOT.h"
#include "TMath.h"
#include "TColor.h"
#include "TEnv.h"

#include "TQtApplication.h"
#include "TQtWidget.h"
#include "TGQt.h"
#include "TQtBrush.h"
#include "TQtClientFilter.h"
#include "TQtEventQueue.h"

#include "TSystem.h"
#ifdef R__QTWIN32
#  include "TWinNTSystem.h"
#  include "Win32Constants.h"
#  include <Winuser.h>
#else
# ifdef R__QTX11
#  include <X11/Xlib.h>
# endif
#endif

#include "TSysEvtHandler.h"
#include "TQtMarker.h"

#include "TError.h"

TGQt *gQt=0;
TVirtualX *TGQt::fgTQt = 0; // to remember the poiner fulishing ROOT PluginManager later.
static const int kDefault=2;
//______________________________________________________________________________
//  static methods:
//______________________________________________________________________________
//______________________________________________________________________________
//  static methods:
// kNone    =  no window
// kDefault =  means default desktopwindow
//  else the pointer to the QPaintDevice
//______________________________________________________________________________
Int_t         TGQt::iwid(QPaintDevice *wid) 
{ 
   // method to provide the ROOT "cast" from (QPaintDevice*) to ROOT windows "id"
   Int_t intWid = 0;
   QPaintDevice *topDevice = (QPaintDevice *)QApplication::desktop();
   if (wid == topDevice) intWid = kDefault;
   else if (wid)  intWid = Int_t(wid);
   return intWid;
}
//______________________________________________________________________________
QPaintDevice *TGQt::iwid(Int_t wid) 
{ 
   // method to restore (cast) the QPaintDevice object pointer from  ROOT windows "id"
   QPaintDevice *topDevice = 0; 
   if ( wid == Int_t(kNone) )    return 0;
   if ( wid == Int_t(kDefault) ) topDevice = (QPaintDevice *)QApplication::desktop();
   else topDevice = (QPaintDevice *)wid;
   return topDevice;
}
//______________________________________________________________________________
QWidget      *TGQt::winid(Window_t id)      
{
   // returns the top level QWidget fro the ROOT widget
   return (id != kNone)?((QWidget *)(TGQt::iwid(id)))->topLevelWidget():0; 
}

//______________________________________________________________________________
QWidget      *TGQt::wid(Window_t id)        
{
   // method to restore (dynamic cast) the QWidget object pointer (if any) from  ROOT windows "id"
   if (id == kNone || id == (unsigned int)(-1) ) return 0;
   QPaintDevice *dev = TGQt::iwid(id);
    assert(dev->devType() == QInternal::Widget);
   //if ( dev->devType() != QInternal::Widget) {
   //     printf(" %s %i type=%d name=%s, className=%s QInternal::Widget = %d\n", __FUNCTION__, __LINE__
   //        , dev->devType()
   //        , (const char *)dev->name(), (const char *)dev->className(), QInternal::Widget);
   //     assert (dev->devType() == QInternal::Widget);
   //}
   return (QWidget *)dev;
}
//______________________________________________________________________________
void TGQt::PrintEvent(Event_t &ev)
{
   // Dump trhe ROOT Event_t structure to debug the code

   //EGEventType fType;              // of event (see EGEventTypes)
   //Window_t    fWindow;            // window reported event is relative to
   //Time_t      fTime;              // time event event occured in ms
   //Int_t       fX, fY;             // pointer x, y coordinates in event window
   //Int_t       fXRoot, fYRoot;     // coordinates relative to root
   //UInt_t      fCode;              // key or button code
   //UInt_t      fState;             // key or button mask
   //UInt_t      fWidth, fHeight;    // width and height of exposed area
   //Int_t       fCount;             // if non-zero, at least this many more exposes
   //Bool_t      fSendEvent;         // true if event came from SendEvent
   //Handle_t    fHandle;            // general resource handle (used for atoms or windows)
   //Int_t       fFormat;            // Next fields only used by kClientMessageEvent
   //Long_t      fUser[5];           // 5 longs can be used by client message events
   //                                // NOTE: only [0], [1] and [2] may be used.
   //                                // [1] and [2] may contain >32 bit quantities
   //                                // (i.e. pointers on 64 bit machines)
   fprintf(stderr,"----- Window %p %s\n", TGQt::wid(ev.fWindow),(const char *)TGQt::wid(ev.fWindow)->name());
   fprintf(stderr,"event type =  %x, key or button code %d \n", ev.fType, ev.fCode);
   fprintf(stderr,"fX, fY, fXRoot, fYRoot = %d %d  :: %d %d\n", ev.fX, ev.fY,ev.fXRoot, ev.fYRoot);
}


//______________________________________________________________________________
static float CalibrateFont() 
{
    // Use the ROOT font with ID=1 to calibrate the current font on fly;
    bool  italic = TRUE;
    long  bold   = 5;
    QString fontName = "Times New Roman";

    QFont pattern;
   
    pattern.setWeight(bold*10);
    pattern.setItalic(italic);
    pattern.setFamily(fontName);
    pattern.setPixelSize(12);

   int w,h;
   QFontMetrics metrics(pattern);
   w = metrics.width("This is a PX distribution");
   h = metrics.height();

// I found 0.94 matches well what Rene thinks it should be
// for TTF and XFT and it should be 1.1 for X Fonts
//
//  X11 returns      h = 12
//  XFT returns      h = 14   
// WIN32 TTF returns h = 16   
   
//   printf(" Font metric w = %d , h = %d\n", w,h);
   float f;
    switch (h) {
       case 12: f = 1.13; break;
       case 14: f = 0.94; break;
       case 16: f = 0.94; break;
       default: f = 1.10; break;
    }
    return f;
}
 
//______________________________________________________________________________
static inline float FontMagicFactor(float size)
{
   // Adjust the font size to match that for Postscipt format
   static float calibration =0;
   if (calibration == 0) calibration = CalibrateFont();
   return TMath::Max(calibration*size,Float_t(1.0));
}

int TGQt::fgCoinFlag = 0; // no current coin viewer;
int TGQt::fgCoinLoaded = 0; // coint viewer DLL has not been loaded
//______________________________________________________________________________
int TGQt::CoinFlag()
{ 
  // return the Coin/QGL viewer flag safely
   qApp->lock();
   int ret = fgCoinFlag;
   qApp->unlock();
   return ret;
}
//______________________________________________________________________________
void TGQt::SetCoinFlag(int flag)
{
  // Set the Coin/QGL viewer flag safely
   qApp->lock();
   fgCoinFlag=flag;
   qApp->unlock();
}
//______________________________________________________________________________
void TGQt::SetCoinLoaded() {  fgCoinLoaded = 1; }

//______________________________________________________________________________
Int_t TGQt::IsCoinLoaded(){ return fgCoinLoaded;}

//______________________________________________________________________________
class TQtInputHandler : public TFileHandler 
{
  protected: 
    //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    Int_t OpenDisplay()
    {
      Int_t dsp = 0;
      if (GetFd() == -1 &&  (dsp =  gVirtualX->OpenDisplay(0)) ) {
        SetFd(dsp); 
        if (gSystem) {
          gSystem->AddFileHandler(this);
          Added(); // emit Added() signal
        }
      }
      return GetFd();
    }
  public:
    //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    TQtInputHandler(int fd=-1,int mask=0):TFileHandler(fd,mask) 
    {
      gXDisplay = this;
      OpenDisplay();
    }
    //_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    virtual Bool_t  Notify() 
    { 
      OpenDisplay();
      Event_t evnt;
      if ( qApp->hasPendingEvents ()) {
         gVirtualX->NextEvent(evnt); //       return fClient->HandleInput();

         Notified(); // it is not clear whether this should be done
         return kTRUE;  
      }
      return TFileHandler::Notify();  
    }
};
//______________________________________________________________________________
QPixmap *TGQt::MakeIcon(Int_t i)
{
//  Create the Qt QPixamp from the WIN32 system icon (for WIN32 only)
   QPixmap *tempIcon = NULL;
   if (i) { /* just to suspend the warning under UNIX */ }
#ifdef R__QTWIN32
   HICON largeIcon[1];
   HICON smallIcon[1];
   HICON icon = ((TWinNTSystem *)gSystem)->GetNormalIcon(i);
#if 0
   int numIcons = ::ExtractIconEx(
    "c:\winnt\explorer.exe",
    0,
    largeIcon,
    smallIcon,
    1);
   if (numIcons > 0)
   {
#endif
   tempIcon =new QPixmap (GetSystemMetrics(SM_CXSMICON),
                          GetSystemMetrics(SM_CYSMICON));
   HDC dc = tempIcon->handle();
   DrawIcon (dc, 0, 0, icon);
#else
   gSystem->ExpandPathName("$ROOTSYS/icons/");
//   tempIcon =new QPixmap (16,16),
#endif
   return tempIcon;
}

#define NoOperation (QPaintDevice *)(-1)




ClassImp(TGQt) 

//____________________________________________________
//
//   Some static methods
//______________________________________________________________________________
QString TGQt::RootFileFormat(const char *selector)
{  return RootFileFormat(QString(selector)); }
//______________________________________________________________________________
QString TGQt::RootFileFormat(const QString &selector)
{
   // Define whether the input string contains any pattern 
   // that matches the ROOT inmage formats
   // those Qt library can not provide
   QString saveType;
   QString defExtension[] = {"cpp","cxx","eps","svg","root","ps","C"};
   UInt_t nExt = sizeof(defExtension)/sizeof(const char *);

   for (UInt_t i = 0; i < nExt; i++) {
      if (selector.contains(defExtension[i],FALSE)) {
         saveType = defExtension[i];
         break;
      }
   }
   if (saveType.contains("C",FALSE)) saveType= "cxx";
   return saveType;
}

//______________________________________________________________________________
QString TGQt::QtFileFormat(const char *selector)
{ return QtFileFormat(QString(selector)); }

//______________________________________________________________________________
QString TGQt::QtFileFormat(const QString &selector)
{
   // returns Qt file format
   //
   // if no suitable format found and ther selector is empty 
   // the default PNG format is returned
   //
   // a special treatment of the "gif" format. 
   // If "gif" is not provided with the local Qt installation 
   // replace "gif" format with "png" one
   //
   QString saveType="PNG"; // it is the default format
   if (!selector.isEmpty())  {
      for (UInt_t j = 0; j < QImageIO::outputFormats().count(); j++ ) 
      {
         QString nextFormat =  QImageIO::outputFormats().at( j );
         // Trick to count both "jpeg" and "jpg" extenstion
         QString checkString = selector.contains("jpg",FALSE) ? "JPEG" : selector;
         if (checkString.contains(nextFormat,FALSE) ) {
            saveType = nextFormat; 
            break;
         } 
      }
      // a special treatment of the "gif" format. 
      // If "gif" is not provided with the local Qt installation 
      // replace "gif" format with "png" one
      if (saveType.isEmpty() && selector.contains("gif",FALSE)) saveType="PNG";
   }
   return saveType;
}

//______________________________________________________________________________
TQtApplication *TGQt::CreateQtApplicationImp()
{
   // The method to instantiate the QApplication if needed
   static TQtApplication *app = 0;
   if (!app) {
      //    app = new TQtApplication(gApplication->ApplicationName(),gApplication->Argc(),gApplication->Argv());
      static TString argvString ("$ROOTSYS/bin/root.exe");
      gSystem->ExpandPathName(argvString);
      char *argv[] = {(char *)argvString.Data()};

//     static char *argv[] = {"QtRoot"};
      int nArg = 1;
      app = new TQtApplication("Qt",nArg,argv);
   }
   return app;
}
//______________________________________________________________________________
void TGQt::PostQtEvent(QObject *receiver, QEvent *event)
{
   // Qt annnouced that QThread;;postEvent to become obsolete and 
   // we have to switch to the QAppication instead.
#if (QT_VERSION < 0x030200)
  QThread::postEvent(receiver,event);
#else
  QApplication::postEvent(receiver,event);
#endif
}

//______________________________________________________________________________
TGQt::TGQt() : TVirtualX(),fDisplayOpened(kFALSE),fQPainter(0),fQClientFilterBuffer(0)
,fCodec(0)
{
   //*-*-*-*-*-*-*-*-*-*-*-*Default Constructor *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                    ===================
   fgTQt = this;
   gQt   = this;
   fSelectedBuffer = 0;
   fSelectedWindow = fPrevWindow = NoOperation;
}
//______________________________________________________________________________
TGQt::TGQt(const char *name, const char *title) : TVirtualX(name,title),fDisplayOpened(kFALSE)
,fQPainter(0),fCursors(kNumCursors),fQClientFilter(0),fQClientFilterBuffer(0),fPointerGrabber(0)
,fCodec(0)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*Normal Constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                        ==================                              *-*
   fgTQt = this;
   gQt   = this;
#ifndef R__QTGUITHREAD
   CreateQtApplicationImp();
   Init();
#endif
}
//______________________________________________________________________________
TGQt::~TGQt()
{
   //*-*-*-*-*-*-*-*-*-*-*-*Default Destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                    ==================
   qApp->lock();
   gVirtualX = gGXBatch;
   gROOT->SetBatch();
   delete fQClientFilter;
   delete fQClientFilterBuffer;
   delete fQPainter; fQPainter = 0;
   qApp->unlock();
   // Stop GUI thread
   TQtApplication::Terminate();
   // fprintf(stderr, "TGQt::~TGQt()<------\n");
}
//______________________________________________________________________________
Bool_t TGQt::Init(void* /*display*/)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*Qt GUI initialization-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                        ========================                      *-*
   fprintf(stderr,"** $Id: TGQt.cxx,v 1.62 2004/07/21 21:55:42 fine Exp $ this=%p\n",this);

   if(fDisplayOpened)   return fDisplayOpened;
   fSelectedBuffer = fSelectedWindow = fPrevWindow = NoOperation;
   fTextAlignH      = 1;
   fTextAlignV      = 1;
   fTextMagnitude   = 1;
   fCharacterUpX    = 1;
   fCharacterUpY    = 1;
   fDrawMode        = Qt::CopyROP;
   fTextFontModified = 0;

   fTextAlign   = 0;
   fTextSize    = -1;
   fTextFont    = -1;
   fLineWidth   = -1;
   fFillColor   = -1;
   fLineColor   = -1;
   fLineStyle   = -1;
   fMarkerSize  = -1;
   fMarkerStyle = -1;
  
   fGLKernel = 0;
   // fGLKernel = new TWin32GLKernel();

   //
   // Retrieve the applicaiton instance
   //

   // --   fHInstance = GetModuleHandle(NULL);

   //
   // Create cursors
   //
   // Qt::BlankCursor - blank/invisible cursor
   // Qt::BitmapCursor
   fCursors.setAutoDelete(true);

   fCursors.insert(kBottomLeft, new QCursor(Qt::SizeBDiagCursor)); // diagonal resize (/) LoadCursor(NULL, IDC_SIZENESW);// (display, XC_bottom_left_corner);
   fCursors.insert(kBottomRight,new QCursor(Qt::SizeFDiagCursor)); // diagonal resize (\) LoadCursor(NULL, IDC_SIZENWSE);// (display, XC_bottom_right_corner);
   fCursors.insert(kTopLeft,    new QCursor(Qt::SizeFDiagCursor)); // diagonal resize (\)  (display, XC_top_left_corner);
   fCursors.insert(kTopRight,   new QCursor(Qt::SizeBDiagCursor)); // diagonal resize (/) LoadCursor(NULL, IDC_SIZENESW);// (display, XC_top_right_corner);
   //fCursors.insert(kBottomSide,   new QCursor(Qt::SplitHCursor));    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_bottom_side);
   //fCursors.insert(kLeftSide,     new QCursor(Qt::SplitVCursor));    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_left_side);
   //fCursors.insert(kTopSide,      new QCursor(Qt::SplitHCursor));    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_top_side);
   //fCursors.insert(kRightSide,    new QCursor(Qt::SplitVCursor));    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_right_side);
   fCursors.insert(kBottomSide, new QCursor(Qt::SizeVerCursor));    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_bottom_side);
   fCursors.insert(kLeftSide,   new QCursor(Qt::SizeHorCursor));    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_left_side);
   fCursors.insert(kTopSide,    new QCursor(Qt::SizeVerCursor));    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_top_side);
   fCursors.insert(kRightSide,  new QCursor(Qt::SizeHorCursor));    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_right_side);

   fCursors.insert(kMove,       new QCursor(Qt::SizeAllCursor));   //  all directions resize LoadCursor(NULL, IDC_SIZEALL); // (display, XC_fleur);
   fCursors.insert(kCross,      new QCursor(Qt::CrossCursor));     // - crosshair LoadCursor(NULL, IDC_CROSS);   // (display, XC_tcross);
   fCursors.insert(kArrowHor,   new QCursor(Qt::SizeHorCursor));   //   horizontal resize LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_sb_h_double_arrow);
   fCursors.insert(kArrowVer,   new QCursor(Qt::SizeVerCursor));   //  vertical resize LoadCursor(NULL, IDC_SIZENS)  (display, XC_sb_v_double_arrow);
   fCursors.insert(kHand,       new QCursor(Qt::PointingHandCursor)); //  a pointing hand LoadCursor(NULL, IDC_NO);      // (display, XC_hand2);
   fCursors.insert(kRotate,     new QCursor(Qt::ForbiddenCursor)); // - a slashed circle LoadCursor(NULL, IDC_ARROW);    // (display, XC_exchange);
   fCursors.insert(kPointer,    new QCursor(Qt::ArrowCursor));     // standard arrow cursor  / (display, XC_left_ptr);
   fCursors.insert(kArrowRight, new QCursor(Qt::UpArrowCursor));   // - upwards arrow LoadCursor(NULL, IDC_ARROW);   // XC_arrow
   fCursors.insert(kCaret,      new QCursor(Qt::IbeamCursor));     //  ibeam/text entry LoadCursor(NULL, IDC_IBEAM);   // XC_xterm
   fCursors.insert(kWatch,      new QCursor(Qt::WaitCursor));      // 

   // The default cursor

   fCursor = kCross;

   // Qt object used to paint the canvas
   fQPen     = new QPen;
   fQBrush   = new TQtBrush;
   fQtMarker = new TQtMarker;
   fQFont    = new QFont();
   // ((TGQt *)TGQt::GetVirtualX())->SetQClientFilter(
   fQClientFilter = new TQtClientFilter();

   //  Query the default font for Widget decoration.  
   fFontTextCode = "ISO8859-1";
   const char *default_font = 
      gEnv->GetValue("Gui.DefaultFont",  "-adobe-helvetica-medium-r-*-*-12-*-*-*-*-*-iso8859-1");
   QApplication::setFont(*(QFont *)LoadQueryFont(default_font));
   //  define the font code page
   QString fontName(default_font);
   fFontTextCode = fontName.section('-',13).upper();
   if  ( fFontTextCode.isEmpty() ) fFontTextCode = "ISO8859-1";
   

   //  printf(" TGQt::Init finsihed\n");
   // Install filter for the desktop
   // QApplication::desktop()->installEventFilter(QClientFilter());

   fDisplayOpened = kTRUE;
   return fDisplayOpened;
}
//______________________________________________________________________________
Int_t TGQt::CreatROOTThread()
{
//*-*-*-*-*dummy*-*-*-*-*-*-*-*-*
//*-*   
  return 0;
}
//______________________________________________________________________________
Int_t TGQt::InitWindow(ULong_t window)
{
   //*-*
   //*-*  if window == 0 InitWindow creates his own instance of  TQtWindowsObject object
   //*-*
   //*-*  Create a new windows
   //*-*
   // window is QWidget
   TQtWidget *wid = 0;
   QWidget *parent = (window == kDefault) ? 0 : dynamic_cast<QWidget *>(iwid(window));
   if (parent && gDebug==3) {
      // fprintf(stderr," New Canvas window with the parent %s\n",(const char *)parent->name());
   }
   wid = new TQtWidget(parent,"virtualx",Qt::WStyle_NoBorder,FALSE);
   wid->setCursor(*fCursors[kCross]);

   return iwid(wid);
}
//______________________________________________________________________________
Int_t TGQt::OpenPixmap(UInt_t w, UInt_t h)
{
   //*-*  Create a new pixmap object
   QPixmap *obj =  new QPixmap(w,h);
   return iwid(obj);
}
//______________________________________________________________________________
QColor &TGQt::ColorIndex(Color_t ic)
{
   // Define the QColor object by ROOT color index
#ifndef R__QTWIN32
   // There three different ways in ROOT to define RGB.
   // It took 4 months to figure out. 
   const int BIGGEST_RGB_VALUE=255;
   static QColor colorBuffer;
   // const int ColorOffset = 0;
   TColor *myColor = gROOT->GetColor(ic);
   if (myColor) {
      colorBuffer.setRgb(  int(myColor->GetRed()  *BIGGEST_RGB_VALUE +0.5)
         ,int(myColor->GetGreen()*BIGGEST_RGB_VALUE +0.5)
         ,int(myColor->GetBlue() *BIGGEST_RGB_VALUE +0.5)
         );
   } 
#ifdef QTDEBUG      
   else {
      fprintf(stderr," TGQt::%s:%d - Wrong color index %d\n", __FUNCTION__,__LINE__, ic);
   }
#endif      

   return colorBuffer;
#else
   QColor &c = fPallete[ic+ColorOffset];
   return c;
#endif 
}
//______________________________________________________________________________
UInt_t TGQt::ExecCommand(TGWin32Command* /*command*/)
{ 
   // deprecated
   fprintf(stderr,"** Error **:  TGQt::ExecCommand no implementation\n");
   return 0;
}

//______________________________________________________________________________
void TGQt::SetDoubleBufferOFF()
{ 
   // deprecated
   fprintf(stderr,"** Error **:  TGQt::SetDoubleBufferOFF no implementation\n");
}
//______________________________________________________________________________
void TGQt::SetDoubleBufferON()
{ 
   // deprecated
   fprintf(stderr,"** Error **:  TGQt::SetDoubleBufferON no implementation\n");
}
//______________________________________________________________________________
void TGQt::GetPlanes(Int_t &nplanes){
//*-*-*-*-*-*-*-*-*-*-*-*Get maximum number of planes*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============================
//*-*  nplanes     : number of bit planes
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   nplanes  = QColor::numBitPlanes ();
}

//______________________________________________________________________________
void  TGQt::ClearWindow()
{
   // Clear current window.
   // fprintf(stderr,"TGQt::ClearWindow() %p\n",fSelectedWindow);
   if (fSelectedWindow && fSelectedWindow != NoOperation) 
   {
      switch (fSelectedWindow->devType()) {
    case QInternal::Widget:
       ((TQtWidget *)fSelectedWindow)->erase();
       break;
    case QInternal::Pixmap:
       ((QPixmap *)fSelectedWindow)->fill();
       break;
    case QInternal::Picture:
    case QInternal::Printer:
    case QInternal::UndefinedDevice:
       fQPainter->eraseRect(GetQRect(*fSelectedWindow));
       break;
    default:
       assert(0);
       break;
      };
   }
}
//______________________________________________________________________________
void  TGQt::ClosePixmap()
{
   // Delete current pixmap.
   DeleteSelectedObj();
}
//______________________________________________________________________________
void  TGQt::CloseWindow()
{
   // Delete current window.
   DeleteSelectedObj();
}
//______________________________________________________________________________
void  TGQt::DeleteSelectedObj()
{
    // Delete current Qt object
  End();
  if (fSelectedWindow->devType() == QInternal::Widget) {
      TQtWidget *canvasWidget = dynamic_cast<TQtWidget *>(fSelectedWindow);
       QWidget *wrapper = 0;
       if (canvasWidget && (wrapper=canvasWidget->GetRootID())) {
            wrapper->hide();
            DestroyWindow(iwid( wrapper) );
       } else {
         ((QWidget *)fSelectedWindow)->hide();
         ((QWidget *)fSelectedWindow)->close(true);
       }
  } else {
     delete  fSelectedWindow;
  }
  fSelectedBuffer = fSelectedWindow = 0;
  fPrevWindow     = 0;
}
//______________________________________________________________________________
QRect TGQt::GetQRect(QPaintDevice &dev)
{
   // Define the rectangle of the current ROOT selection
  QRect res;
  switch (dev.devType()) {
  case QInternal::Widget:
    res = ((TQtWidget*)&dev)->rect();
    break;

  case QInternal::Pixmap: {
     TQtWidgetBuffer *pxmBuffer = dynamic_cast<TQtWidgetBuffer *>(&dev);
     if (pxmBuffer) res = pxmBuffer->rect();
     else           res = ((QPixmap *)&dev)->rect();
     break;
                          }
  case QInternal::Picture:
     res = ((QPicture *)&dev)->boundingRect();
     break;

  case QInternal::Printer:
  case QInternal::UndefinedDevice:
     break;
  default:
     assert(0);
     break;
  };
  return res;
}
//______________________________________________________________________________
void  TGQt::CopyPixmap(int wid, int xpos, int ypos)
{
   // Copy the pixmap wid at the position xpos, ypos in the current window.

   if (!wid || (wid == -1) ) return;
   assert(((QPaintDevice *)wid)->devType() == QInternal::Pixmap);
   QPixmap *src = (QPixmap *)(QPaintDevice *)wid;
   // fprintf(stderr," TGQt::CopyPixmap Selected = %p, Buffer = %p, wid = %p\n",
   //    fSelectedWindow,fSelectedBuffer,iwid(wid));
   if (fSelectedWindow )
   {
      QRect sr = src->rect();
      // fprintf(stderr,"x=%d,y=%d: %d %d %d %d\n",xpos,ypos,sr.x(),sr.y(),sr.width(),sr.height());
      QPaintDevice *dst = fSelectedBuffer ? fSelectedBuffer : fSelectedWindow;
      bool isPainted = dst->paintingActive ();
      if (isPainted) End();
      bitBlt ( dst,QPoint(xpos,ypos),src,sr,Qt::CopyROP); // bool ignoreMask )
      if (isPainted) Begin();
      Emitter()->EmitPadPainted(src);
      if (!fSelectedBuffer && (fSelectedWindow->devType() == QInternal::Widget ) )
      {
        TQtWidget *w = (TQtWidget *)fSelectedWindow;
        w->EmitCanvasPainted(); 
      }
   }
}
//______________________________________________________________________________
void TGQt::CreateOpenGLContext(int wid)
{
 // Create OpenGL context for win windows (for "selected" Window by default)
 // printf(" TGQt::CreateOpenGLContext for wid = %x fSelected= %x, threadID= %d \n",wid,fSelectedWindow,
 //    GetCurrentThreadId());
  if (!wid || (wid == -1) ) return;

#ifdef QtGL
    if (!wid)
    {
      SafeCallWin32
         ->W32_CreateOpenGL();
    }
    else
    {
      SafeCallW32(((TQtSwitch *)wid))
         ->W32_CreateOpenGL();
    }
#endif

}

//______________________________________________________________________________
void TGQt::DeleteOpenGLContext(int wid)
{
  // Delete OpenGL context for win windows (for "selected" Window by default)
  if (!wid || (wid == -1) ) return;

#ifdef QtGL
    if (!wid)
    {
      SafeCallWin32
         ->W32_DeleteOpenGL();
    }
    else
    {
      SafeCallW32(((TQtSwitch *)wid))
         ->W32_DeleteOpenGL();
    }
#endif
}

//______________________________________________________________________________
void  TGQt::DrawBox(int x1, int y1, int x2, int y2, EBoxMode mode)
{
   // Draw a box.
   // mode=0 hollow  (kHollow)
   // mode=1 solid   (kSolid)

   qApp->lock();
   if (fSelectedWindow)
   {
      fQPainter->save();
      //    fprintf(stderr, " Drawbox x1=%d, y1=%d, x2=%d, y2=%d, mode=%d\n",x1,y1,x2,y2,mode);
      if (mode == kHollow)
      {
         fQPainter->setBrush(Qt::NoBrush);
         fQPainter->drawRect(x1,y2,x2-x1+1,y1-y2+1);
      } else {
         if (fQBrush->style() != Qt::SolidPattern)
         {
            fQPainter->setBackgroundColor(fQBrush->GetColor());
            fQPainter->setBackgroundMode( Qt::OpaqueMode );
         }
         fQPainter->fillRect(x1,y2,x2-x1+1,y1-y2+1,*fQBrush);
      }
      fQPainter->restore();
   }
   qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic)
{
   // Draw a cell array.
   // x1,y1        : left down corner
   // x2,y2        : right up corner
   // nx,ny        : array size
   // ic           : array
   //
   // Draw a cell array. The drawing is done with the pixel presicion
   // if (X2-X1)/NX (or Y) is not a exact pixel number the position of
   // the top rigth corner may be wrong.

   qApp->lock();
   if (fSelectedWindow)
   {
      fQPainter->save();
      int i,j,icol,ix,w,h,current_icol,lh;

      current_icol = -1;
      w            = TMath::Max((x2-x1)/(nx),1);
      h            = TMath::Max((y1-y2)/(ny),1);
      lh           = y1-y2;
      ix           = x1;

      if (w+h == 2)
      {
         //*-*  The size of the box is equal a single pixel
         for ( i=x1; i<x1+nx; i++){
            for (j = 0; j<ny; j++){
               icol = ic[i+(nx*j)];
               if (current_icol != icol) {
                  current_icol = icol;
                  fQPainter->setPen(ColorIndex(current_icol));
               }
               fQPainter->drawPoint(i,y1-j);
            }
         }
      }
      else
      {
         //*-* The shape of the box is a rectangle
         QRect box(x1,y1,w,h);
         for ( i=0; i<nx; i++ ) {
            for ( j=0; j<ny; j++ ) {
               icol = ic[i+(nx*j)];
               if(icol != current_icol){
                  current_icol = icol;
                  fQPainter->setBrush(ColorIndex(current_icol));
               }
               fQPainter->drawRect(box);
               box.moveBy(0,-h);   // box.top -= h;
            }
            box.moveBy(w,lh);
         }
      }
      fQPainter->restore();
   }
   qApp->unlock();

}
//______________________________________________________________________________
void  TGQt::DrawFillArea(int n, TPoint *xy)
{
   // Fill area described by polygon.
   // n         : number of points
   // xy(2,n)   : list of points

   qApp->lock();
   if (fSelectedWindow && n>0)
   {
      fQPainter->save();
      if (fQBrush->style() == Qt::SolidPattern)
         fQPainter->setPen(Qt::NoPen);
      else {
         fQPainter->setBackgroundColor(fQBrush->GetColor());
         fQPainter->setBackgroundMode( Qt::OpaqueMode );
      } 
      QPointArray qtPoints(n);
      TPoint *rootPoint = xy;
      for (int i =0;i<n;i++,rootPoint++)
         qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
      fQPainter->drawPolygon(qtPoints);
      fQPainter->restore();
   }
   qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::DrawLine(int x1, int y1, int x2, int y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

  qApp->lock();
  if (fSelectedWindow) fQPainter->drawLine(x1,y1,x2,y2);
  qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::DrawPolyLine(int n, TPoint *xy)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points

  qApp->lock();
  if (fSelectedWindow)  {
    QPointArray qtPoints(n);
    TPoint *rootPoint = xy;
    for (int i =0;i<n;i++,rootPoint++)
       qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
    fQPainter->drawPolyline(qtPoints);
  }
  qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::DrawPolyMarker(int n, TPoint *xy)
{
   // Draw n markers with the current attributes at position x, y.
   // n    : number of markers to draw
   // xy   : x,y coordinates of markers
   qApp->lock();
   if (fSelectedWindow)
   {
      fQPainter->save();

      TQtMarker *CurMarker = fQtMarker;
      /* Set marker Color */
      QColor &mColor  = ColorIndex(fMarkerColor);

      if( CurMarker->GetNumber() <= 0 )
      {
         fQPainter->setPen(mColor);
         QPointArray qtPoints(n);
         TPoint *rootPoint = xy;
         for (int i=0;i<n;i++,rootPoint++)
            qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
         fQPainter->drawPoints(qtPoints);
      } else {
         int r = CurMarker->GetNumber()/2;
         fQPainter->setPen(mColor);
         switch (CurMarker -> GetType())
         {
         case 1:
         case 3:
         default:
            fQPainter->setBrush(mColor);
            break;
         case 0:
         case 2:
            fQPainter->setBrush(Qt::NoBrush);
            break;
         case 4:
            break;
         }

         for( int m = 0; m < n; m++ )
         {
            int i;
            switch( CurMarker->GetType() )
            {
            case 0:        /* hollow circle */
            case 1:        /* filled circle */
               fQPainter->drawEllipse(xy[m].fX-r, xy[m].fY-r, 2*r, 2*r);
               break;
            case 2:        /* hollow polygon */
            case 3:        /* filled polygon */
               {
                  QPointArray &mxy = fQtMarker->GetNodes();
                  QPoint delta(xy[m].fX,xy[m].fY);
                  for( i = 0; i < CurMarker->GetNumber(); i++ )
                  {
                     mxy[i] += delta;
                  }

                  fQPainter->drawPolygon(mxy);

                  for( i = 0; i < CurMarker->GetNumber(); i++ )
                  {
                     mxy[i] -= delta;
                  }
                  break;
               }
            case 4:        /* segmented line */
               {
                  QPointArray &mxy = fQtMarker->GetNodes();
                  QPoint delta(xy[m].fX,xy[m].fY);
                  for( i = 0; i < CurMarker->GetNumber(); i++ )
                  {
                     mxy[i] += delta;
                  }
                  fQPainter->drawLineSegments(mxy);
                  for( i = 0; i < CurMarker->GetNumber(); i++ )
                  {
                     mxy[i] -= delta;
                  }

                  break;
               }
            }
         }
      }
      fQPainter->restore();
   }
   qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::DrawText(int x, int y, float angle, float mgn, const char *text, TVirtualX::ETextMode /*mode*/)
{

   // Draw a text string using current font.
   // mode       : drawing mode
   // mode=0     : the background is not drawn (kClear)
   // mode=1     : the background is drawn (kOpaque)
   // x,y        : text position
   // angle      : text angle
   // mgn        : magnification factor
   // text       : text string


   //  We have to check angle to make sure we are setting the right font
#if 0
   if (fROOTFont.lfEscapement != (LONG) fTextAngle*10)  {
      fTextFontModified=1;
      fROOTFont.lfEscapement   = (LONG) fTextAngle*10;
   }
#endif
   // fprintf(stderr,"TGQt::DrawText: %s\n", text);
   if (text && text[0]) {
      qApp->lock();
      if (TMath::Abs(mgn-1) >0.05)  fQFont->setPixelSizeFloat(mgn*FontMagicFactor(fTextSize));
      UpdateFont();
      fQPainter->save();
      fQPainter->setPen(ColorIndex(fTextColor));
      fQPainter->setBrush(ColorIndex(fTextColor));

      QFontMetrics metrics(*fQFont);
      QRect bRect = metrics.boundingRect(text);
      switch( fTextAlignH ) {
           case 2: x -= bRect.width()/2; // h center;
              break;
           case 3: x -= bRect.width();         //  Right;
      };

      switch( fTextAlignV ) {
          case 2: y += bRect.height()/2 - metrics.descent(); // v center
             break;
          case 3: y += bRect.height() - metrics.descent(); // AlignTop;
      };
      fQPainter->translate(x,y);
      // Add rotation if any 
      if (TMath::Abs(angle) > 0.1 )  fQPainter->rotate(-angle);

      fQPainter->drawText (0, 0, GetTextDecoder()->toUnicode (text));
      
      fQPainter->restore();
      qApp->unlock();
   }
}
//______________________________________________________________________________
void  TGQt::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   // Return character up vector.

   qApp->lock();
   chupx = fCharacterUpX;
   chupy = fCharacterUpY;
   qApp->unlock();
}
//______________________________________________________________________________
Int_t  TGQt::GetDoubleBuffer(Int_t wid)
{
   // Query the double buffer value for the window wid.
   // return pointer to the off-screen buffer if any

   if (wid == -1 || wid == kDefault ) return 0;

   QPaintDevice *dev = iwid(wid);
   TQtWidget *widget = dynamic_cast<TQtWidget *>(dev);
   return  Int_t(widget && widget->IsDoubleBuffered() ? &widget->GetBuffer() : 0);
}

//______________________________________________________________________________
void  TGQt::GetGeometry(int wid, int &x, int &y, unsigned int &w, unsigned int &h)
{
   // Returns the global cooordinate of the window "wid"
   QRect devSize(0,0,0,0);
   QPaintDevice  *dev = iwid(wid);
   if( wid == -1 || wid == 0 || wid == kDefault)
   {
      QDesktopWidget *d = QApplication::desktop();
      devSize.setWidth (d->width() );
      devSize.setHeight(d->height());
   } else if (dev) {
      if ( dev->devType() == QInternal::Widget) {
         TQtWidget &thisWidget = *(TQtWidget *)dev;
         if (thisWidget.GetRootID() ) {
            // we are using the ROOT Gui factory 
            devSize = thisWidget.parentWidget()->geometry();
         } else{
            devSize = thisWidget.geometry();
         } 
         devSize.moveTopLeft(thisWidget.mapToGlobal(thisWidget.pos()));
      }
      else {
         devSize = GetQRect(*(QPaintDevice *)wid);
      }
   }
   x = devSize.left();
   y = devSize.top();
   w = devSize.width();
   h = devSize.height();
   // fprintf(stderr," TGQt::GetGeometry %d %d %d %d\n", x,y,w,h);
}
//______________________________________________________________________________
const char *TGQt::DisplayName(const char *){ return "localhost"; }

//______________________________________________________________________________
ULong_t  TGQt::GetPixel(Color_t cindex)
{ 
   // Return pixel value associated to specified ROOT color number.
   // see: GQTGUI.cxx:QtColor() also
   ULong_t rootPixel = 0;
   QColor color = ColorIndex(cindex);
   rootPixel =                    ( color.blue () & 255 ); 
   rootPixel = (rootPixel << 8) | ( color.green() & 255 ) ;
   rootPixel = (rootPixel << 8) | ( color.red  () & 255 );

   return rootPixel;
}

//______________________________________________________________________________
void  TGQt::GetRGB(int index, float &r, float &g, float &b)
{
   // Get rgb values for color "index".
   qApp->lock();
   const float BIGGEST_RGB_VALUE=255.;
   r = g = b = 0;
   if (fSelectedWindow != NoOperation) {
      int c[3];
      QColor &color = fPallete[index];
      color.rgb(&c[0],&c[1],&c[2]);

      r = c[0]/BIGGEST_RGB_VALUE;
      g = c[1]/BIGGEST_RGB_VALUE;
      b = c[2]/BIGGEST_RGB_VALUE;
   }
   qApp->unlock();
}
//______________________________________________________________________________
const QTextCodec *TGQt::GetTextDecoder()  
{
   if (!fCodec) {
      fCodec =  QTextCodec::codecForName(fFontTextCode); //CP1251
      if (!fCodec)
         fCodec=QTextCodec::codecForLocale();
      else 
         QTextCodec::setCodecForLocale(fCodec);
   }      
   return fCodec;
}
//______________________________________________________________________________
Float_t      TGQt::GetTextMagnitude(){return fTextMagnitude;}
//______________________________________________________________________________
void         TGQt::SetTextMagnitude(Float_t mgn){ fTextMagnitude = mgn;}
//______________________________________________________________________________
void  TGQt::GetTextExtent(unsigned int &w, unsigned int &h, char *mess)
{
   // Return the size of a character string.
   // iw          : text width
   // ih          : text height
   // mess        : message

   qApp->lock();
   if (fQFont) {
      QFontMetrics metrics(*fQFont);
      w = metrics.width(mess);
      h = metrics.height();
   }
   qApp->unlock();
}

//______________________________________________________________________________
Bool_t  TGQt::HasTTFonts() const {return kTRUE;}

//______________________________________________________________________________
void  TGQt::MoveWindow(Int_t wid, Int_t x, Int_t y)
{
   // Move the window wid.
   // wid  : Window identifier.
   // x    : x new window position
   // y    : y new window position

   if (wid != -1 && wid != 0 && wid != kDefault)
   {
      QPaintDevice *widget = iwid(wid);
      assert(widget->devType() == QInternal::Widget );
      ((TQtWidget *)widget)->move(x,y);
   }
}
//______________________________________________________________________________
void  TGQt::PutByte(Byte_t )
{   // deprecated
}
//______________________________________________________________________________
void  TGQt::QueryPointer(int &ix, int &iy){ 
   // deprecated
   if (ix*iy); 
}
//______________________________________________________________________________
Pixmap_t TGQt::ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id)
{ 
   // If id is NULL - loads the specified gif file at position [x0,y0] in the 
   // current window. Otherwise creates pixmap from gif file 

   QPixmap *pix = new QPixmap( QString (file) );
   if ( pix->isNull () ) { delete pix; pix = 0;         }
   else if (!id)         { CopyPixmap(iwid(pix),x0,y0); delete pix; pix = 0;}
   return iwid(pix);
}

//______________________________________________________________________________
Int_t  TGQt::RequestLocator(Int_t /*mode*/, Int_t /*ctyp*/, Int_t &/*x*/, Int_t &/*y*/)
{
   // deprecated
   return 0;
}
//______________________________________________________________________________
Int_t  TGQt::RequestString(int x, int y, char *text)
{
//*-*-*-*-*-*-*-*-*-*-*-*Request string*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============
//*-*  x,y         : position where text is displayed
//*-*  text        : text displayed (input), edited text (output)
//*-*
//*-*  Request string:
//*-*  text is displayed and can be edited with Emacs-like keybinding
//*-*  return termination code (0 for ESC, 1 for RETURN)
//*-*
//*-*  Return value:
//*-*
//*-*    0     -  input was canceled
//*-*    1     -  input was Ok
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  
  class requestString : public QDialog {
  public:
    QString   fText;
    QLineEdit fEdit;
    requestString(const char *text="") : QDialog(0,0,TRUE,Qt::WStyle_Customize | Qt::WStyle_NoBorder|Qt::WStyle_StaysOnTop), fText(text),fEdit(this)
    {
       setBackgroundMode(Qt::NoBackground);
       connect(&fEdit,SIGNAL( returnPressed () ), this, SLOT( accept() ));
    }
    ~requestString(){;}
  };
  int  res = QDialog::Rejected;
  if (fSelectedWindow->devType() == QInternal::Widget ) {
     TQtWidget *w = (TQtWidget *)fSelectedWindow;
     static requestString reqDialog;
     reqDialog.fEdit.setText(QString(text).stripWhiteSpace());
     int yFrame = reqDialog.frameGeometry().height() - reqDialog.geometry().height() + reqDialog.fontMetrics().height();
     reqDialog.move(w->mapToGlobal(QPoint(x,y-yFrame)));
     if (QClientFilter() && QClientFilter()->GetPointerGrabber() ) {
        // suspend the mouse grabbing for a while
        QClientFilter()->SetPointerGrabber(0); 
     }
     res = reqDialog.exec();
     if (res == QDialog::Accepted ) {
        QCString r = GetTextDecoder()->fromUnicode(reqDialog.fEdit.text());
        qstrcpy(text, (const char *)r);
     }
     reqDialog.hide();
     if (QClientFilter()) {
        // Restore the grabbing 
        QClientFilter()->SetPointerGrabber(fPointerGrabber);
     }
  }
  return res == QDialog::Accepted ? 1 : 0;
}

//______________________________________________________________________________
void  TGQt::RescaleWindow(int wid, UInt_t w, UInt_t h)
{
   // Rescale the window wid.
   // wid  : Window identifier
   // w    : Width
   // h    : Heigth

   qApp->lock();
   if (wid && wid != -1 && wid != kDefault )
   {
      QPaintDevice *widget = iwid(wid);
      if (widget->devType() == QInternal::Widget )
      {
         if (QSize(w,h) != ((TQtWidget *)widget)->size()) {
            if (((TQtWidget *)widget)->paintingActive() ) End();
            // fprintf(stderr," TGQt::RescaleWindow(int wid, UInt_t w=%d, UInt_t h=%d)\n",w,h);
            ((TQtWidget *)widget)->resize(w,h);
         }
      }
   }
   qApp->unlock();
}
//______________________________________________________________________________
Int_t  TGQt::ResizePixmap(int wid, UInt_t w, UInt_t h)
{
   // Resize a pixmap.
   // wid : pixmap to be resized
   // w,h : Width and height of the pixmap

   qApp->lock();
   if (wid && wid != -1 && wid != kDefault )
   {
      QPaintDevice *pixmap = iwid(wid);
      if (pixmap->devType() == QInternal::Pixmap )
      {
         if (QSize(w,h) != ((QPixmap *)pixmap)->size()) {
            bool paintStatus = pixmap->paintingActive ();
            if (paintStatus ) End();
            ((QPixmap *)pixmap)->resize(w,h);
            ((QPixmap *)pixmap)->fill();
            // fprintf(stderr," \n --- > Pixmap has been resized ,< --- \t  %p\n",pixmap);
            if (paintStatus) Begin();
         }
      }
   }
   qApp->unlock();
   return 1;
}
//______________________________________________________________________________
void  TGQt::ResizeWindow(int wid)
{
   // Resize the current window if necessary.

   if (wid == -1) return;
   QPaintDevice *dev = iwid(wid);
   TQtWidget *widget = dynamic_cast<TQtWidget *>(dev);
   if (widget) {
      bool painting = widget->paintingActive();
      if (painting) End();
      widget->adjustSize ();
      if (painting) Begin();
   }
}
//______________________________________________________________________________
void   TGQt::SelectPixmap(Int_t qpixid){ SelectWindow(qpixid);}

//______________________________________________________________________________
void  TGQt::SelectWindow(int wid)
{
   // Select window to which subsequent output is directed.

   // Don't select things twice
   QPaintDevice *dev = iwid(wid); 
   QPixmap *offScreenBuffer = (QPixmap *)GetDoubleBuffer(wid);
   if ((dev == fSelectedWindow) && !( (fSelectedBuffer==0) ^ (offScreenBuffer == 0) ) ) return;
   fPrevWindow     = fSelectedWindow;
   if (wid == -1) { fSelectedBuffer=0; fSelectedWindow = NoOperation; }
   else {
      fSelectedWindow = dev;
      fSelectedBuffer = offScreenBuffer;
   }
   // fprintf(stderr,"TGQt::SelectWindow fSelecteWindow old = %p; current= %p, buffer =%p\n"
   //            ,fPrevWindow,fSelectedWindow, fSelectedBuffer);
   if (fPrevWindow && (iwid(fPrevWindow) != -1) )            End();
   if (fSelectedWindow && (fSelectedWindow != NoOperation))  Begin();
}
//______________________________________________________________________________
void  TGQt::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   // Set character up vector.

   qApp->lock();
   if (chupx == fCharacterUpX  && chupy == fCharacterUpY) {
      qApp->unlock();
      return;
   }

   if      (chupx == 0  && chupy == 0)  fTextAngle = 0;
   else if (chupx == 0  && chupy == 1)  fTextAngle = 0;
   else if (chupx == -1 && chupy == 0)  fTextAngle = 90;
   else if (chupx == 0  && chupy == -1) fTextAngle = 180;
   else if (chupx == 1  && chupy ==  0) fTextAngle = 270;
   else {
      fTextAngle = ((TMath::ACos(chupx/TMath::Sqrt(chupx*chupx +chupy*chupy))*180.)/3.14159)-90;
      if (chupy < 0) fTextAngle = 180 - fTextAngle;
      if (TMath::Abs(fTextAngle) < 0.01) fTextAngle = 0;
   }

   fCharacterUpX = chupx;
   fCharacterUpY = chupy;
   qApp->unlock();
}
//______________________________________________________________________________
void  TGQt::SetClipOFF(Int_t /*wid*/)
{
   // Turn off the clipping for the window wid.
   // deprecated
   // fQPainter->setClipping(FALSE);
}
//______________________________________________________________________________
void  TGQt::SetClipRegion(int wid, int x, int y, UInt_t w, UInt_t h)
{
   // Set clipping region for the window wid.
   // wid        : Window indentifier
   // x,y        : origin of clipping rectangle
   // w,h        : size of clipping rectangle;

   QRect rect(x,y,w,h);
   qApp->lock();
   fClipMap.replace(iwid(wid),rect);
   if (fSelectedWindow == iwid(wid) && fSelectedWindow->paintingActive())
   {
      UpdateClipRectangle();
   }
   qApp->unlock();
}
//____________________________________________________________________________
void  TGQt::SetCursor(Int_t wid, ECursor cursor)
{
   // Set the cursor.
   fCursor = cursor;
   if (wid && wid != -1 && wid != kDefault)
   {
      QPaintDevice *widget = iwid(wid);
      if (widget->devType() == QInternal::Widget )
      {
         ((TQtWidget *)widget)->setCursor(*fCursors[fCursor]);
      }
   }
}
//______________________________________________________________________________
void  TGQt::SetDoubleBuffer(int wid, int mode)
{
   // Set the double buffer on/off on window wid.
   // wid  : Window identifier.
   //        999 means all the opened windows.
   // mode : 1 double buffer is on
   //        0 double buffer is off

   if (wid == -1 && wid == kDefault) return;
   QPaintDevice *dev = iwid(wid);
   TQtWidget *widget = dynamic_cast<TQtWidget *>(dev);
   if (widget) {
      widget->SetDoubleBuffer(mode);
   }
}
//______________________________________________________________________________
void  TGQt::SetDrawMode(TVirtualX::EDrawMode mode)
{
   // Set the drawing mode.
   // mode : drawing mode

   // Map EDrawMode    { kCopy = 1, kXor, kInvert };
   Qt::RasterOp newMode = Qt::CopyROP;
   switch (mode) {
    case kCopy:   newMode = Qt::CopyROP; break;
    case kXor:    newMode = Qt::XorROP;  break;
    case kInvert: newMode = Qt::NotROP;  break;
    default:      newMode = Qt::CopyROP; break;
   };
   if (newMode != fDrawMode)
   {
      fDrawMode = newMode;
      if (fQPainter->isActive()) { fQPainter->setRasterOp(fDrawMode); }
   }
}
//______________________________________________________________________________
void  TGQt::SetFillColor(Color_t cindex)
{
   // Set color index for fill areas.

   if (fFillColor != cindex ) 
   {
      fFillColor = cindex;
      if (fFillColor != -1) {
         fQBrush->SetColor(ColorIndex(cindex));
         UpdateBrush();
      }
   }
}
//______________________________________________________________________________
void  TGQt::SetFillStyle(Style_t fstyle)
{
   // Set fill area style.
   // fstyle   : compound fill area interior style
   //    fstyle = 1000*interiorstyle + styleindex

   if (fFillStyle != fstyle) 
   {
      fFillStyle = fstyle;
      if (fFillStyle != -1) {
         Int_t style = fstyle/1000;
         Int_t fasi  = fstyle%1000;

         fQBrush->SetStyle(style,fasi);
         UpdateBrush();
      }
   }
}
//______________________________________________________________________________
void TGQt::SetFillStyleIndex( Int_t style, Int_t fasi )
{
   // Set fill area style index.

   SetFillStyle(1000*style + fasi);
}
//______________________________________________________________________________
void  TGQt::SetLineColor(Color_t cindex)
{
//*-*-*-*-*-*-*-*-*-*-*Set color index for lines*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  cindex    : color index
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fLineColor != cindex) {
    fLineColor = cindex;
    if (fLineColor >= 0) {
      fQPen->setColor(ColorIndex(fLineColor));
      UpdatePen();
    }
  }
}
//______________________________________________________________________________
void  TGQt::SetLineType(int n, int* /*dash*/)
{
//*-*-*-*-*-*-*-*-*-*-*Set line style-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============
//*-*    Set line style:
//*-*    if n < 0 use pre-defined Windows style:
//*-*         0 - solid lines
//*-*        -1 - solid lines
//*-*        -2 - dash line
//*-*        -3 - dot  line
//*-*        -4 - dash-dot line
//*-*        -5 - dash-dot-dot line
//*-*     < -6 - solid line
//*-*
//*-*    if n > 0 use dashed lines described by DASH(N)
//*-*    e.g. n=4,DASH=(6,3,1,3) gives a dashed-dotted line with dash length 6
//*-*    and a gap of 7 between dashes
//*-*
  if (n < 0 ) {
    Qt::PenStyle styles[] = {
      Qt::NoPen          // - no line at all.
     ,Qt::SolidLine      // - a simple line.
     ,Qt::DashLine       // - dashes separated by a few pixels.
     ,Qt::DotLine        // - dots separated by a few pixels.
     ,Qt::DashDotLine    // - alternate dots and dashes.
     ,Qt::DashDotDotLine // - one dash, two dots, one dash, two dotsQt::NoPen
    };
    int l = -n;
    if (l > int(sizeof(styles)/sizeof(Qt::PenStyle)) ) l = 1; // Solid line "by default"
    fQPen->setStyle(styles[l]);
    UpdatePen();
  }
}
//______________________________________________________________________________
void  TGQt::SetLineStyle(Style_t linestyle)
{
//*-*-*-*-*-*-*-*-*-*-*Set line style-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============
//*-*    Use pre-defined Windows style:
//*-*    linestyle =
//*-*         0 - solid lines
//*-*        -1 - solid lines
//*-*        -2 - dash line
//*-*        -3 - dot  line
//*-*        -4 - dash-dot line
//*-*        -5 - dash-dot-dot line
//*-*      < -6 - solid line
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (fLineStyle != linestyle) { //set style index only if different
      fLineStyle = linestyle;
      SetLineType(-linestyle, NULL);
   }
}
//______________________________________________________________________________
void  TGQt::SetLineWidth(Width_t width)
{
   //*-*-*-*-*-*-*-*-*-*-*Set line width*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ==============
   //*-*  width   : line width in pixels
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   if (width==1) width =0;
   if (fLineWidth != width) {
      fLineWidth = width;
      if (fLineWidth >= 0 ) {
         fQPen->setWidth(fLineWidth);
         UpdatePen();
      }
   } 
}
//______________________________________________________________________________
void  TGQt::SetMarkerColor( Color_t cindex)
{
   //*-*-*-*-*-*-*-*-*-*-*Set color index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ===========================
   //*-*  cindex : color index defined my IXSETCOL
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fMarkerColor != cindex) fMarkerColor = cindex;
}

//______________________________________________________________________________
void  TGQt::SetMarkerSize(Float_t markersize)
{
   //*-*-*-*-*-*-*-*-*-*-*Set marker size index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  =================================
   //*-*  msize  : marker scale factor
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (markersize != fMarkerSize) {

      fMarkerSize = markersize;
      if (markersize >= 0) {
         SetMarkerStyle(-fMarkerStyle);
      }
   }
}

//______________________________________________________________________________
void  TGQt::SetMarkerStyle(Style_t markerstyle){
   //*-*-*-*-*-*-*-*-*-*-*Set marker style*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ================

   if (fMarkerStyle == markerstyle) return;
   TPoint shape[15];
   if (markerstyle >= 31) return;
   markerstyle  = TMath::Abs(markerstyle);
   fMarkerStyle = markerstyle;
   Int_t im = Int_t(4*fMarkerSize + 0.5);
   switch (markerstyle) {

case 2:
   //*-*--- + shaped marker
   shape[0].SetX(-im); shape[0].SetY( 0);
   shape[1].SetX(im);  shape[1].SetY( 0);
   shape[2].SetX(0) ;  shape[2].SetY( -im);
   shape[3].SetX(0) ;  shape[3].SetY( im);
   SetMarkerType(4,4,shape);
   break;

case 3:
   //*-*--- * shaped marker
   shape[0].SetX(-im);  shape[0].SetY(  0);
   shape[1].SetX( im);  shape[1].SetY(  0);
   shape[2].SetX(  0);  shape[2].SetY(-im);
   shape[3].SetX(  0);  shape[3].SetY( im);
   im = Int_t(0.707*Float_t(im) + 0.5);
   shape[4].SetX(-im);  shape[4].SetY(-im);
   shape[5].SetX( im);  shape[5].SetY( im);
   shape[6].SetX(-im);  shape[6].SetY( im);
   shape[7].SetX( im);  shape[7].SetY(-im);
   SetMarkerType(4,8,shape);
   break;

case 4:
case 24:
   //*-*--- O shaped marker
   SetMarkerType(0,im*2,shape);
   break;

case 5:
   //*-*--- X shaped marker
   im = Int_t(0.707*Float_t(im) + 0.5);
   shape[0].SetX(-im);  shape[0].SetY(-im);
   shape[1].SetX( im);  shape[1].SetY( im);
   shape[2].SetX(-im);  shape[2].SetY( im);
   shape[3].SetX( im);  shape[3].SetY(-im);
   SetMarkerType(4,4,shape);
   break;

case  6:
   //*-*--- + shaped marker (with 1 pixel)
   shape[0].SetX(-1);  shape[0].SetY( 0);
   shape[1].SetX( 1);  shape[1].SetY( 0);
   shape[2].SetX( 0);  shape[2].SetY(-1);
   shape[3].SetX( 0);  shape[3].SetY( 1);
   SetMarkerType(4,4,shape);
   break;

case 7:
   //*-*--- . shaped marker (with 9 pixel)
   shape[0].SetX(-1);  shape[0].SetY( 1);
   shape[1].SetX( 1);  shape[1].SetY( 1);
   shape[2].SetX(-1);  shape[2].SetY( 0);
   shape[3].SetX( 1);  shape[3].SetY( 0);
   shape[4].SetX(-1);  shape[4].SetY(-1);
   shape[5].SetX( 1);  shape[5].SetY(-1);
   SetMarkerType(4,6,shape);
   break;
case  8:
case 20:
   //*-*--- O shaped marker (filled)
   SetMarkerType(1,im*2,shape);
   break;
case 21:      //*-*- here start the old HIGZ symbols
   //*-*--- HIGZ full square
   shape[0].SetX(-im);  shape[0].SetY(-im);
   shape[1].SetX( im);  shape[1].SetY(-im);
   shape[2].SetX( im);  shape[2].SetY( im);
   shape[3].SetX(-im);  shape[3].SetY( im);
   //     shape[4].SetX(-im);  shape[4].SetY(-im);
   SetMarkerType(3,4,shape);
   break;
case 22:
   //*-*--- HIGZ full triangle up
   shape[0].SetX(-im);  shape[0].SetY( im);
   shape[1].SetX( im);  shape[1].SetY( im);
   shape[2].SetX(  0);  shape[2].SetY(-im);
   //     shape[3].SetX(-im);  shape[3].SetY( im);
   SetMarkerType(3,3,shape);
   break;
case 23:
   //*-*--- HIGZ full triangle down
   shape[0].SetX(  0);  shape[0].SetY( im);
   shape[1].SetX( im);  shape[1].SetY(-im);
   shape[2].SetX(-im);  shape[2].SetY(-im);
   //     shape[3].SetX(  0);  shape[3].SetY( im);
   SetMarkerType(3,3,shape);
   break;
case 25:
   //*-*--- HIGZ open square
   shape[0].SetX(-im);  shape[0].SetY(-im);
   shape[1].SetX( im);  shape[1].SetY(-im);
   shape[2].SetX( im);  shape[2].SetY( im);
   shape[3].SetX(-im);  shape[3].SetY( im);
   //     shape[4].SetX(-im);  shape[4].SetY(-im);
   SetMarkerType(2,4,shape);
   break;
case 26:
   //*-*--- HIGZ open triangle up
   shape[0].SetX(-im);  shape[0].SetY( im);
   shape[1].SetX( im);  shape[1].SetY( im);
   shape[2].SetX(  0);  shape[2].SetY(-im);
   //     shape[3].SetX(-im);  shape[3].SetY( im);
   SetMarkerType(2,3,shape);
   break;
case 27: {
   //*-*--- HIGZ open losange
   Int_t imx = Int_t(2.66*fMarkerSize + 0.5);
   shape[0].SetX(-imx); shape[0].SetY( 0);
   shape[1].SetX(  0);  shape[1].SetY(-im);
   shape[2].SetX(imx);  shape[2].SetY( 0);
   shape[3].SetX(  0);  shape[3].SetY( im);
   //     shape[4].SetX(-imx); shape[4].SetY( 0);
   SetMarkerType(2,4,shape);
   break;
         }
case 28: {
   //*-*--- HIGZ open cross
   Int_t imx = Int_t(1.33*fMarkerSize + 0.5);
   shape[0].SetX(-im);  shape[0].SetY(-imx);
   shape[1].SetX(-imx); shape[1].SetY(-imx);
   shape[2].SetX(-imx); shape[2].SetY( -im);
   shape[3].SetX(imx);  shape[3].SetY( -im);
   shape[4].SetX(imx);  shape[4].SetY(-imx);
   shape[5].SetX( im);  shape[5].SetY(-imx);
   shape[6].SetX( im);  shape[6].SetY( imx);
   shape[7].SetX(imx);  shape[7].SetY( imx);
   shape[8].SetX(imx);  shape[8].SetY( im);
   shape[9].SetX(-imx); shape[9].SetY( im);
   shape[10].SetX(-imx);shape[10].SetY(imx);
   shape[11].SetX(-im); shape[11].SetY(imx);
   //     shape[12].SetX(-im); shape[12].SetY(-imx);
   SetMarkerType(2,12,shape);
   break;
         }
case 29: {
   //*-*--- HIGZ full star pentagone
   Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
   Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
   Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
   Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
   shape[0].SetX(-im);  shape[0].SetY( im4);
   shape[1].SetX(-im2); shape[1].SetY(-im1);
   shape[2].SetX(-im3); shape[2].SetY( -im);
   shape[3].SetX(  0);  shape[3].SetY(-im2);
   shape[4].SetX(im3);  shape[4].SetY( -im);
   shape[5].SetX(im2);  shape[5].SetY(-im1);
   shape[6].SetX( im);  shape[6].SetY( im4);
   shape[7].SetX(im4);  shape[7].SetY( im4);
   shape[8].SetX(  0);  shape[8].SetY( im);
   shape[9].SetX(-im4); shape[9].SetY( im4);
   //     shape[10].SetX(-im); shape[10].SetY( im4);
   SetMarkerType(3,10,shape);
   break;
         }

case 30: {
   //*-*--- HIGZ open star pentagone
   Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
   Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
   Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
   Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
   shape[0].SetX(-im);  shape[0].SetY( im4);
   shape[1].SetX(-im2); shape[1].SetY(-im1);
   shape[2].SetX(-im3); shape[2].SetY( -im);
   shape[3].SetX(  0);  shape[3].SetY(-im2);
   shape[4].SetX(im3);  shape[4].SetY( -im);
   shape[5].SetX(im2);  shape[5].SetY(-im1);
   shape[6].SetX( im);  shape[6].SetY( im4);
   shape[7].SetX(im4);  shape[7].SetY( im4);
   shape[8].SetX(  0);  shape[8].SetY( im);
   shape[9].SetX(-im4); shape[9].SetY( im4);
   SetMarkerType(2,10,shape);
   break;
         }

case 31:
   //*-*--- HIGZ +&&x (kind of star)
   SetMarkerType(1,im*2,shape);
   break;
default:
   //*-*--- single dot
   SetMarkerType(0,0,shape);
   }
}

//______________________________________________________________________________
void  TGQt::SetMarkerType( int type, int n, TPoint *xy )
{
//*-*-*-*-*-*-*-*-*-*-*Set marker type*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============
//*-*  type      : marker type
//*-*  n         : length of marker description
//*-*  xy        : list of points describing marker shape
//*-*
//*-*     if N.EQ.0 marker is a single point
//*-*     if TYPE.EQ.0 marker is hollow circle of diameter N
//*-*     if TYPE.EQ.1 marker is filled circle of diameter N
//*-*     if TYPE.EQ.2 marker is a hollow polygon describe by line XY
//*-*     if TYPE.EQ.3 marker is a filled polygon describe by line XY
//*-*     if TYPE.EQ.4 marker is described by segmented line XY
//*-*     e.g. TYPE=4,N=4,XY=(-3,0,3,0,0,-3,0,3) sets a plus shape of 7x7 pixels
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
     fQtMarker->SetMarker(n,xy,type);
}

//______________________________________________________________________________
void  TGQt::SetRGB(int cindex, float r, float g, float b)
{
#define BIGGEST_RGB_VALUE 255  // 65535
   //  if (fSelectedWindow == NoOperation) return;
   if (cindex < 0 ) return;
   else {
      //    if (cindex >= fPallete.size()) fPallete.resize(cindex+1);
      //    fPallete[cindex].setRgb((r*BIGGEST_RGB_VALUE)
      fPallete[cindex] = QColor(
         int(r*BIGGEST_RGB_VALUE+0.5)
         ,int(g*BIGGEST_RGB_VALUE+0.5)
         ,int(b*BIGGEST_RGB_VALUE+0.5)
         );
   }
}
//______________________________________________________________________________
void  TGQt::SetTextAlign(Short_t talign)
{
   //*-*-*-*-*-*-*-*-*-*-*Set text alignment*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ==================
   //*-*  txalh   : horizontal text alignment
   //*-*  txalv   : vertical text alignment
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t txalh = talign/10;
   Int_t txalv = talign%10;

   fTextAlignH = txalh;
   fTextAlignV = txalv;

   fTextAlign = Qt::AlignAuto;
   switch( txalh ) {

  case 2:
     fTextAlign |= Qt::AlignHCenter;
     break;

  case 3:
     fTextAlign |= Qt::AlignRight;
     break;

  default:
     fTextAlign |= Qt::AlignLeft;
   }

   switch( txalv ) {

  case 1:
     fTextAlign |= Qt::AlignBottom;
     break;

  case 2:
     fTextAlign |= Qt::AlignVCenter;
     break;

  case 3:
     fTextAlign |= Qt::AlignTop;
     break;

  default:
     fTextAlign = Qt::AlignBottom;
   }
}

//______________________________________________________________________________
void  TGQt::SetTextColor(Color_t cindex)
{
   //*-*-*-*-*-*-*-*-*-*-*Set color index for text*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ========================
   //*-*  cindex    : color index defined my IXSETCOL
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fTextColor == cindex) return;
   fTextColor = cindex;
   if (cindex < 0) return;
}
//______________________________________________________________________________
Int_t  TGQt::SetTextFont(char* /*fontname*/, TVirtualX::ETextSetMode /*mode*/){
  return 1;
}
//______________________________________________________________________________
void  TGQt::SetTextFont(const char *fontname, int italic, int bold)
{

   //*-*    mode              : Option message
   //*-*    italic   : Italic attribut of the TTF font
   //*-*    bold     : Weight attribute of the TTF font
   //*-*    fontname : the name of True Type Font (TTF) to draw text.
   //*-*
   //*-*    Set text font to specified name. This function returns 0 if
   //*-*    the specified font is found, 1 if not.

   fQFont->setWeight((long) bold*10);
   fQFont->setItalic((Bool_t)italic);
   fQFont->setFamily(fontname);
   fTextFontModified = 1;
   // fprintf(stderr, "font: %s bold=%d italic=%d\n",fontname,bold,italic);
}

//______________________________________________________________________________
void  TGQt::SetTextFont(Font_t fontnumber)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set current text font number*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      ===========================
   //*-*  List of the currently supported fonts (screen and PostScript)
   //*-*  =============================================================
   //*-*   Font ID       X11                       Win32 TTF       lfItalic  lfWeight x 10
   //*-*        1 : times-medium-i-normal      "Times New Roman"      1           5
   //*-*        2 : times-bold-r-normal        "Times New Roman"      0           8
   //*-*        3 : times-bold-i-normal        "Times New Roman"      1           8
   //*-*        4 : helvetica-medium-r-normal  "Arial"                0           5
   //*-*        5 : helvetica-medium-o-normal  "Arial"                1           5
   //*-*        6 : helvetica-bold-r-normal    "Arial"                0           8
   //*-*        7 : helvetica-bold-o-normal    "Arial"                1           8
   //*-*        8 : courier-medium-r-normal    "Courier New"          0           5
   //*-*        9 : courier-medium-o-normal    "Courier New"          1           5
   //*-*       10 : courier-bold-r-normal      "Courier New"          0           8
   //*-*       11 : courier-bold-o-normal      "Courier New"          1           8
   //*-*       12 : symbol-medium-r-normal     "Symbol"               0           6
   //*-*       13 : times-medium-r-normal      "Times New Roman"      0           5
   //*-*       14 :                            "Wingdings"            0           5

   if ( fTextFont == fontnumber) return;
   fTextFont = fontnumber;
   if (fTextFont == -1) {
      fTextFontModified = 1;
      return;
   }
   int italic, bold;
   const char *fontName = "Times New Roman";

   switch(fontnumber/10) {

   case  1:
      italic = 1;
      bold   = 5;
      fontName = "Times New Roman";
      break;
   case  2:
      italic = 0;
      bold   = 8;
      fontName = "Times New Roman";
      break;
   case  3:
      italic = 1;
      bold   = 8;
      fontName = "Times New Roman";
      break;
   case  4:
      italic = 0;
      bold   = 5;
      fontName = "Arial";
      break;
   case  5:
      italic = 1;
      bold   = 5;
      fontName = "Arial";
      break;
   case  6:
      italic = 0;
      bold   = 8;
      fontName = "Arial";
      break;
   case  7:
      italic = 1;
      bold   = 8;
      fontName = "Arial";
      break;
   case  8:
      italic = 0;
      bold   = 5;
      fontName = "Courier New";
      break;
   case  9:
      italic = 1;
      bold   = 5;
      fontName = "Courier New";
      break;
   case 10:
      italic = 0;
      bold   = 8;
      fontName = "Courier New";
      break;
   case 11:
      italic = 1;
      bold   = 8;
      fontName = "Courier New";
      break;
   case 12:
      italic = 0;
      bold   = 6;
      fontName = "Symbol";
      break;
   case 13:
      italic = 0;
      bold   = 5;
      fontName = "Times New Roman";
      break;
   case 14:
      italic = 0;
      bold   = 5;
      fontName = "Wingdings";
      break;
   default:
      italic = 0;
      bold   = 5;
      fontName = "Times New Roman";
      break;

   }
   SetTextFont(fontName, italic, bold);
}
//______________________________________________________________________________
void  TGQt::SetTextSize(Float_t textsize)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set current text size*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      =====================
   if ( fTextSize != textsize ) {
      fTextSize = textsize;
      if (fTextSize > 0) {	
         fQFont->setPixelSize(int(FontMagicFactor(fTextSize)));
         fTextFontModified = 1;
      }
   }
}

//______________________________________________________________________________
void  TGQt::SetTitle(const char *title)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set title of the object*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      =======================
   if (fSelectedWindow->devType() == QInternal::Widget)
   {
      ((TQtWidget *)fSelectedWindow)->topLevelWidget()->setCaption(GetTextDecoder()->toUnicode(title));
   }
}

//______________________________________________________________________________
void  TGQt::UpdateWindow(int mode)
{
   // Update display.
   // mode : (1) update
   //        (0) sync

   if (fSelectedWindow && mode != 2 ) {
      ((TQtWidget *)fSelectedWindow)->paintFlag();
      ((TQtWidget *)fSelectedWindow)->repaint();
   }
}
//______________________________________________________________________________
void  TGQt::Warp(int /*ix*/, int /*iy*/) {
//     SafeCallWin32
//      ->W32_Warp(ix, iy);
}
//______________________________________________________________________________
Int_t  TGQt::WriteGIF(char *name)
{
   //
   // Writes the current active window into pixmap file. 
   // The format is defined by the file name extension
   // like "png","jpg","bmp"  . . .
   // If no extension is provided the "png" format is used by default
   //
   // Returns 1 in case of success,
   //         0 otherwise
   // Note: this method may not produce the expected result been called 
   // ----  from the ROOT prompt by simple reason:
   //       The active window will be console window 
   //       rather the last selected ROOT canvas.
   //
   WritePixmap(iwid(fSelectedWindow),UInt_t(-1),UInt_t(-1),name);
   return kTRUE;
}
//______________________________________________________________________________
void  TGQt::WritePixmap(int wid, UInt_t w, UInt_t h, char *pxname)
{
   // Write the pixmap wid in the bitmap file pxname in JPEG.
   // wid         : Pixmap address
   // w,h         : Width and height of the pixmap.
   //               if w = h = -1 the size of the pimxap is equal the size the wid size
   // pxname      : pixmap file name
   //               The format is defined by the file name extension
   //               like "png","jpg","bmp"  . . .
   //               If no or some unknown extension is provided then 
   //               the "png" format is used by default


   if (!wid || (wid == -1) ) return; 

   QPaintDevice &dev = *iwid(wid);
   QPixmap *pix=0;
   switch (dev.devType()) {
   case QInternal::Widget:
//    bitBlt ( pix,QPoint(xpos,ypos),src,sr,Qt::CopyROP);
      pix = &((TQtWidget*)&dev)->GetBuffer();
     break;

   case QInternal::Pixmap: {
      pix = (QPixmap *)&dev;
      break;
                          }
   case QInternal::Picture:
   case QInternal::Printer:
   case QInternal::UndefinedDevice:
   default: assert(0);
     break;
   };
   if (pix) {
      // Create intermediate pixmap to stretch the original one if any
      QPixmap outMap(0,0);
      if ( (h == w) && (w == UInt_t(-1) ) ) outMap.resize(pix->size());
      else outMap.resize(w,h);
      QPainter pnt(&outMap);
      pnt.drawPixmap(outMap.rect(),*pix);   
      //  define the file extension
      QString saveType = QtFileFormat(QFileInfo(pxname).extension(FALSE));
      if (saveType.isEmpty()) saveType="PNG";
      outMap.save(pxname,saveType);
   }
}

//______________________________________________________________________________
void TGQt::UpdateFont()
{
   // Update the current QFont within active QPainter
   if (fQFont && fQPainter->isActive()) {
      fQPainter->setFont(*fQFont);
      fTextFontModified = 0;
   }
}
//______________________________________________________________________________
void TGQt::UpdatePen()
{
   // Update the current QPen within active QPainter
   if (fQPen  && fQPainter->isActive()) {
      fQPainter->setPen(*fQPen);
      // fprintf(stderr," uu --- uu TGQt::UpdatePen() %p\n",fQPainter->device());
   }
}
//______________________________________________________________________________
void TGQt::UpdateBrush()
{ 
   // Update the current QBrush within active QPainter
   if (fQBrush && fQPainter->isActive()) 
   {
      fQPainter->setBrush(*fQBrush); 
      // fprintf(stderr,"  uu --- uu TGQt::UpdateBrush() %p, r:g:b=%d:%d:%d\n",fQPainter->device(),
      //   fQBrush->color().red(),fQBrush->color().green(),fQBrush->color().blue());
   }
}
//______________________________________________________________________________
void TGQt::UpdateClipRectangle()
{
   // Update the clip rectangle within active QPainter

   if (!fQPainter->isActive()) return;
   TQTCLIPMAP::iterator it= fClipMap.find(fSelectedWindow);
   QRect clipRect;
   if (it != fClipMap.end())  {
      clipRect = it.data();
      fQPainter->setClipRect(clipRect);
      fQPainter->setClipping(TRUE);
   }
}

//______________________________________________________________________________
void TGQt::Begin()
{
   // Start the painting of the current slection (Pixmap or Widget)

   if (!fQPainter) fQPainter = new QPainter();
   if (!fQPainter->isActive() )
   {
      QPaintDevice *src = fSelectedBuffer ? fSelectedBuffer : fSelectedWindow;
      // Adjust size 
      if ( fSelectedWindow->devType() ==  QInternal::Widget) 
         ((TQtWidget *)fSelectedWindow)->AdjustBufferSize();
      if (!fQPainter->begin(src) )
         fprintf(stderr,"---> TGQt::Begin() win=%p dev=%p\n",src,fQPainter->device());
      fQPainter->setBackgroundColor(Qt::white);
      UpdatePen();
      UpdateBrush();
      UpdateFont();
      TQTCLIPMAP::iterator it= fClipMap.find(fSelectedWindow);
      QRect clipRect;
      if (it != fClipMap.end())  {
         clipRect = it.data();
         fQPainter->setClipRect(clipRect);
         fQPainter->setClipping(TRUE);
      }
      fQPainter->setRasterOp(fDrawMode);
   }
}

//______________________________________________________________________________
void TGQt::End()
{
   // End  the painting of the current slection (Pixmap or Widget)

   if ( fQPainter->isActive() )
   {
      // fprintf(stderr,"<--- TGQt::End() %p\n",fQPainter->device());
      fQPainter->end();
   }
}

//______________________________________________________________________________
TVirtualX *TGQt::GetVirtualX(){ return fgTQt;}

//______________________________________________________________________________
Int_t TGQt::LoadQt(const char *shareLibFileName)
{
   // Make sure we load the GUI DLL from the gui thread
   return gSystem->Load(shareLibFileName);
}
//______________________________________________________________________________
Int_t TGQt::processQtEvents()
{
   // Force processing the Qt events only without entering the ROOT event loop
   qApp->processEvents();
   // QEventLoop::ExcludeUserInput QEventLoop::ExcludeSocketNotifiers
   return 0;
 }
