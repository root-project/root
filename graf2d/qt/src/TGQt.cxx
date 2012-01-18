// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                      *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGQt                                                                 //
//                                                                      //
// This class is the basic interface to the Qt graphics system. It is   //
// an implementation of the abstract TVirtualX class.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if defined(HAVE_CONFIG) || defined (R__HAVE_CONFIG)
# include "config.h"
#endif
#ifdef R__QTWIN32
#include <process.h>
#endif

#include <assert.h>

//  Qt include files

#include <QApplication>
#include <QWidget>
#include <QPolygon>
#include <QEvent>
#include <QImageWriter>
#include <QVector>
#include <QStack>
#include <QFrame>
#include <QPicture>
#include <QDebug>

#include <qpixmap.h>
#include <qcursor.h>
#include <qdesktopwidget.h>
#include <qimage.h>
#include <qfontmetrics.h>
#include <qdialog.h>
#include <qlineedit.h>
#include <qfileinfo.h>
#include <qtextcodec.h>
#include <qdir.h>

#include "TROOT.h"
#include "TMath.h"
#include "TColor.h"
#include "TEnv.h"
#include "TQtPen.h"

#include "TQtApplication.h"
#include "TQtWidget.h"
#include "TGQt.h"
#include "TQtBrush.h"
#include "TQtClientFilter.h"
#include "TQtEventQueue.h"
#include "TQtSymbolCodec.h"
#include "TQtLock.h"
#include "TQtPadFont.h"
#include "TStyle.h"
#include "TObjString.h"
#include "TObjArray.h"

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

#include "TImage.h"
#include "TError.h"

TGQt *gQt=0;
TVirtualX *TGQt::fgTQt = 0; // to remember the pointer foolishing ROOT PluginManager later.
TQtTextProxy  *TGQt::fgTextProxy = 0; // The pointer to the custom text proxy;
class TQtTextCloneProxy {
private:
   TQtTextProxy *fProxy;
   void *operator new(size_t);
   TQtTextCloneProxy(const TQtTextCloneProxy&);
   void operator=(const TQtTextCloneProxy&);
public:
   TQtTextCloneProxy()  : fProxy(0) 
   {
      if ( TGQt::TextProxy())
         fProxy = TGQt::TextProxy()->Clone();
   }
   inline ~TQtTextCloneProxy() { delete fProxy; }
   inline TQtTextProxy *operator->() { return fProxy;}
};

#define NoOperation (QPaintDevice *)(-1)

// static const int kDefault=2;
//__________________________________________________________________
static QWidget *IsWidget(QPaintDevice *d)
{ return  dynamic_cast<QWidget *>(d);   }

//__________________________________________________________________
static QPixmap *IsPixmap(QPaintDevice *d)
{ return dynamic_cast<QPixmap *>(d); }
//__________________________________________________________________
QString TGQt::SetFileName(const QString &fileName)
{
   // Set the file pattern
   QFileInfo  fi(fileName);
   QString saveFileMoviePattern =
            fi.path()+"/" + fi.completeBaseName()+ "_%04d" + "." + fi.suffix();
   return saveFileMoviePattern;
}
//__________________________________________________________________
QString TGQt::GetNewFileName(const QString &fileNamePrototype)
{
   // Find the filename for the given "fileNamePrototype"
   TString flN = fileNamePrototype.toStdString().c_str();
   gSystem->ExpandPathName(flN);
   QString fileName = (const char *)flN;

   Int_t counter = 0;
   QString formatPattern = SetFileName(fileName);
   while (gSystem->AccessPathName(fileName.toStdString().c_str())==0) {
      fileName = QString().sprintf(formatPattern.toStdString().c_str(),counter++);
   }
   return  fileName;
}
//______________________________________________________________________________
//
//  custom TQtPainter
//______________________________________________________________________________
class TQtPainter : public QPainter {
private:
    TGQt *fVirtualX;
protected:
   inline void UpdateBrush() { setBrush(*fVirtualX->fQBrush); }
   inline void UpdatePen()   { setPen(*fVirtualX->fQPen);     }
   inline void UpdateFont()  { setFont(*fVirtualX->fQFont);
                               fVirtualX->fTextFontModified = 0;
                             }
public:
   enum  { kNone        = 0,
           kUseFeedBack = 1,
           kUpdateFont  = 2,
           kUpdateBrush = 4,
           kUpdatePen   = 8
   };
   TQtPainter() : QPainter (), fVirtualX(0) {}
   TQtPainter(QPaintDevice * dev) : QPainter ( dev ), fVirtualX(0) {}
   TQtPainter( TGQt *dev, unsigned int useFeedBack=kUpdateBrush | kUpdatePen ) : fVirtualX(0) 
   {  begin(dev,useFeedBack);                                       }
   ~TQtPainter () { fVirtualX->fQPainter = 0; }
   bool begin ( TGQt *dev, unsigned int useFeedBack);
};
//______________________________________________________________________________
//
//   class TQtFeedBackWidget to back the TCanvas FeedBack mode
//______________________________________________________________________________
class TQtFeedBackWidget : public QFrame {
   QPixmap   *fPixBuffer;
   QPixmap   *fGrabBuffer;
   TQtWidget *fParentWidget;
protected:
   virtual void hideEvent (QHideEvent *ev) {
      // hide the feedback widget and remove the buffer
      delete fPixBuffer;  fPixBuffer  = 0;  
      delete fGrabBuffer; fGrabBuffer = 0;
      QFrame::hideEvent(ev);
      if (fParentWidget) {
         fParentWidget->SetIgnoreLeaveEnter(0);         
         SetParent(0);  // reparent // it is dangerous thing to do inside of the event processing
      }
   }
   virtual void paintEvent(QPaintEvent *ev) {
      if (fPixBuffer) {
         QRect rc = ev->rect();
         {
            QPainter p(this);
            p.setClipRect(rc);
            p.drawPixmap(0,0,*fPixBuffer);
         }
         ClearBuffer();
      } else if (fGrabBuffer) {
         QRect rc = ev->rect(); 
         QPainter p(this);
         p.setClipRect(rc);
         p.drawPixmap(rc,*fGrabBuffer);
      }
      QFrame::paintEvent(ev);
   }
public:
   TQtFeedBackWidget(QWidget *mother=0, Qt::WindowFlags f=0)  : QFrame(mother,f)
      ,fPixBuffer(0),fGrabBuffer(0),fParentWidget(0)
   {
      // Create the feedback widget
      setAttribute(Qt::WA_NoSystemBackground); 
      setEnabled(false);
      setBackgroundRole(QPalette::Window);
      setAutoFillBackground(false);
      QPalette  p = palette();
      p.setBrush(QPalette::Window, Qt::transparent);
      setPalette(p);
      setMouseTracking(true);
   }
   virtual ~TQtFeedBackWidget() 
   {
      fParentWidget = 0;
      delete fPixBuffer; fPixBuffer   = 0;
      delete fGrabBuffer; fGrabBuffer = 0;
   }
   void SetParent(TQtWidget *w) {
      setParent(fParentWidget = w);
   }
   void Hide() {
      if (fParentWidget) {
         fParentWidget->SetIgnoreLeaveEnter(0);         
         SetParent(0);  // reparent
      }
   }
   void Show() {
      // Protect (Qt >= 4.5.x) TCanvas  against of 
      // the confusing mouseMoveEvent
      if (fParentWidget) fParentWidget->SetIgnoreLeaveEnter(2);
      QFrame::show();
      if (fParentWidget) fParentWidget->SetIgnoreLeaveEnter();
   }
   QPaintDevice *PixBuffer() {
      // Create the feedback buffer if needed
      if (fParentWidget ) {
         // resize the feedback
         QSize canvasSize = fParentWidget->size();
         setGeometry(QRect(QPoint(0,0),canvasSize));
         if ( !fPixBuffer  || (fPixBuffer->size() != canvasSize) ) {
            delete fPixBuffer;
            fPixBuffer = new QPixmap(canvasSize);
            ClearBuffer();
         }
      }
      return  fPixBuffer;
   }
    QPaintDevice *GrabBuffer(QSize &s) { 
      // Create the feedback buffer to grab the parent TPad image
      if (fParentWidget ) {
         // resize the feedback
          if ( !fPixBuffer  || (fPixBuffer->size() != s) ) {
            delete fPixBuffer;
            fPixBuffer = new QPixmap(s);
            ClearBuffer();
         }
      }
      return  fPixBuffer;
   }
   void ClearBuffer() {
      // Fill the feedback buffer with the transparent background
      fPixBuffer->fill(Qt::transparent);
   }
   void SetGeometry(int xp,int yp, int w, int h, TQtWidget *src=0)
   {
       // Set the feedback widget position and geometry
       if (isHidden() && src ) {
          // grab the parent window and move the feedback 
          delete fGrabBuffer; fGrabBuffer = 0;
          QPixmap *canvas = src->GetOffScreenBuffer();
          if (canvas && w > 4 &&  h > 4 ) {
             fGrabBuffer = new QPixmap(canvas->copy(xp,yp,w,h));
          }
       }
       setGeometry(xp,yp,w,h);
   }
};

//______________________________________________________________________________
//
//   class TQtFeedBackWidget to back the TCanvas FeedBack mode
//______________________________________________________________________________

class TQtToggleFeedBack {
   TGQt *fGQt;
   TQtPainter  fFeedBackPainter;

public:
   TQtToggleFeedBack(TGQt *gqt) : fGQt(gqt)
   {
      // activate temporary TQtFeedBackWidget widget buffer
      if (fGQt->fFeedBackMode && fGQt->fFeedBackWidget->isHidden()) {
         fGQt->fFeedBackWidget->Show();
      }
   }
   ~TQtToggleFeedBack()
   {
      // Update the "feedback" widget
     if (fFeedBackPainter.isActive() ) fFeedBackPainter.end();
     if (fGQt->fFeedBackMode && fGQt->fFeedBackWidget) 
     {   fGQt->fFeedBackWidget->update();                   }
   }
   TQtPainter &painter() {
      // activate return  the "feedback" painter
      if (!fFeedBackPainter.isActive()) {
         fFeedBackPainter.begin(fGQt, TQtPainter::kUseFeedBack 
                                    | TQtPainter::kUpdatePen 
                                    | TQtPainter::kUpdateBrush);
         if (fGQt->fFeedBackMode) {
            fFeedBackPainter.setPen(QColor(128,128,128,128));// Qt::white);// darkGray);
        }
     }
     return fFeedBackPainter; 
   }
};

//______________________________________________________________________________
//
//  custom TQtPainter
//______________________________________________________________________________
inline bool TQtPainter::begin ( TGQt *dev, unsigned int useFeedBack)
{
  // Activate return  the "feedback" painter 
  bool res = false;
  if (dev && (dev->fSelectedWindow != NoOperation)) {
     fVirtualX = dev;
     QPaintDevice *src= 0;
     if ( (useFeedBack & kUseFeedBack) && dev->fFeedBackMode
                     && dev->fFeedBackWidget
                     && dev->fFeedBackWidget) 
     { src = dev->fFeedBackWidget->PixBuffer(); }
     else {
        src = dev->fSelectedWindow;
        if ( src->devType() ==  QInternal::Widget)
        {
           TQtWidget *theWidget =  (TQtWidget *)src;
          // Substitute the widget with its internal buffer
           src = theWidget->SetBuffer().Buffer();
        }
     }
     if (!(res= QPainter::begin(src)) ) {
        Error("TGQt::Begin()","Can not create Qt painter for win=0x%lx dev=0x%lx\n",(Long_t)src,(Long_t)dev);
        assert(0);
     } else {
        dev->fQPainter = (TQtPainter*)-1;
        UpdatePen();
        UpdateBrush();
        UpdateFont();
        TGQt::TQTCLIPMAP::iterator it= (dev->fClipMap).find(src);
        QRect clipRect;
        if (it != (dev->fClipMap).end())  {
           clipRect = it.value();
           setClipRect(clipRect);
           setClipping(TRUE);
        }
        if (src->devType() ==  QInternal::Image )
                 setCompositionMode(dev->fDrawMode);
     }
  }
  return res;
}


//----- Terminal Input file handler --------------------------------------------
//______________________________________________________________________________
class TQtEventInputHandler : public TTimer {
protected: // singleton
   TQtEventInputHandler() : TTimer(340) {  }
   static TQtEventInputHandler *gfQtEventInputHandler;
public:

   static TQtEventInputHandler *Instance() {
     if (!gfQtEventInputHandler)
       gfQtEventInputHandler =  new  TQtEventInputHandler();
       gfQtEventInputHandler->Start(240);
       return gfQtEventInputHandler;
   }
   Bool_t Notify()     {
      Timeout();       // emit Timeout() signal
      Bool_t ret = gQt->processQtEvents();
      Start(240);
      Reset();
      return  ret;
   }
   Bool_t ReadNotify() { return Notify(); }
};

TQtEventInputHandler *TQtEventInputHandler::gfQtEventInputHandler = 0;
//______________________________________________________________________________
//  static methods:
//______________________________________________________________________________
//______________________________________________________________________________
//  static methods:
// kNone    =  no window
// kDefault =  means default desktopwindow
//  else the pointer to the QPaintDevice

class TQWidgetCollection {
 private:
   QStack<int>             fFreeWindowsIdStack;
   QVector<QPaintDevice *> fWidgetCollection;
   Int_t                    fIDMax;       //  current max id
   Int_t                    fIDTotalMax;  // life-time max id
protected:
   //______________________________________________________________________________
   inline  Int_t SetMaxId(Int_t newId)
   {
      fIDMax =  newId;
      if (newId>fIDTotalMax) {
         fIDTotalMax  = newId;
         fWidgetCollection.resize(fIDTotalMax+1);
      }
      return fIDMax;
   }

 public:
   //______________________________________________________________________________
   TQWidgetCollection () : fIDMax(-1), fIDTotalMax(-1)
   {
       // mark the position before kNone and between kNone and kDefault
       // as "free position" if any
       int kDefault = 1;
       assert(!kNone);
       SetMaxId (kDefault);
       fWidgetCollection[kNone]    = (QPaintDevice*)0;
       fWidgetCollection[kDefault] = (QPaintDevice *)QApplication::desktop();
   }

   //______________________________________________________________________________
   inline Int_t GetFreeId(QPaintDevice *device) {

      Int_t Id = 0;
      if (!fFreeWindowsIdStack.isEmpty() ) {
         Id = fFreeWindowsIdStack.pop();
         if (Id > fIDMax ) SetMaxId ( Id );
      } else {
         Id = fWidgetCollection.count();
         assert(fIDMax <= Id  );
         SetMaxId ( Id );
      }
      fWidgetCollection[Id] = device;
      // fprintf(stderr," add %p as %d max Id = %d \n", device, Id,fIDMax);
      return Id;
   }
   //______________________________________________________________________________
   inline Int_t RemoveByPointer(QPaintDevice *device)
   {
      // method to provide the ROOT "cast" from (QPaintDevice*) to ROOT windows "id"
      Int_t intWid = kNone;             // TGQt::iwid(device);
      if ((ULong_t) device != (ULong_t) -1) {
          intWid = find( device);
          if ( intWid != -1 && 
             fWidgetCollection[intWid]) {
             fWidgetCollection[intWid] = (QPaintDevice *)(-1);
             fFreeWindowsIdStack.push(intWid);
             if (fIDMax == intWid) SetMaxId(--fIDMax);
          } else {
             intWid = kNone;
          }
      }
      return intWid;
   }

   //______________________________________________________________________________
   inline const QPaintDevice *DeleteById(Int_t Id)
   {
     QPaintDevice *device = fWidgetCollection[Id];
     if (device) {
        delete device;
        fWidgetCollection[Id] = (QPaintDevice *)(-1);
        fFreeWindowsIdStack.push(Id);
        if (fIDMax == Id) SetMaxId(--fIDMax);
     }
     //return device; this was a huge bug
     return 0;
   }
   //______________________________________________________________________________
   inline const QPaintDevice *ReplaceById(Int_t Id, QPaintDevice *newDev)
   {
      if (newDev) {
        // delete the old definition
        delete fWidgetCollection[Id];
        // add the new one instead
        fWidgetCollection[Id] = newDev;
      } else {
         DeleteById(Id);
      }
      return newDev;
   }

   //______________________________________________________________________________
   inline uint count() const { return fWidgetCollection.count();}
   //______________________________________________________________________________
   inline uint MaxId() const { return fIDMax;}
   //______________________________________________________________________________
   inline uint MaxTotalId() const { return fIDTotalMax;}
   //______________________________________________________________________________
   inline int find(const QPaintDevice *device, uint i=0) const
   { 
      return fWidgetCollection.indexOf((QPaintDevice*)device,i); 
   }
   //______________________________________________________________________________
   inline QPaintDevice *operator[](int i) const {return fWidgetCollection[i];}
};
TQWidgetCollection *fWidgetArray = 0;
//______________________________________________________________________________
QPaintDevice *TGQt::iwid(Window_t wd)
{
   // Convert ROOT Widget Id to the Qt QPaintDevice pointer
   QPaintDevice *topDevice = 0;
   if ( wd != kNone )   {
       topDevice = (wd == kDefault) ?
              (QPaintDevice *)QApplication::desktop()
             :
              (QPaintDevice*)wd;
   }
   return topDevice;
}

//______________________________________________________________________________
Int_t         TGQt::iwid(QPaintDevice *wd)
{
   // method to provide the ROOT "cast" from (QPaintDevice*) to ROOT windows "id"
   Int_t intWid = kNone;
       // look up the widget
   if ((ULong_t) wd == (ULong_t) -1) intWid = -1;
   else {
      intWid = fWidgetArray->find(wd);
      assert(intWid != -1);
      // if (intWid == -1) intWid = Int_t(wd);
   }
   return intWid;
}

//______________________________________________________________________________
QPaintDevice *TGQt::iwid(Int_t wd)
{
   // method to restore (cast) the QPaintDevice object pointer from  ROOT windows "id"
   QPaintDevice *topDevice = 0;
   if (0 <= wd && wd <= int(fWidgetArray->MaxId()) )
     topDevice = (*fWidgetArray)[wd];
     if (topDevice == (QPaintDevice *)(-1) ) topDevice = 0;
   else {
     assert(wd <= Int_t(fWidgetArray->MaxTotalId()));
     // this is allowed from the embedded TCanvas dtor only.
     //  at this point "wd" may have been destroyed
     //-- vf topDevice = (QPaintDevice *)wd;
   }
   return topDevice;
}

//______________________________________________________________________________
QWidget      *TGQt::winid(Window_t id)
{
   // returns the top level QWidget for the ROOT widget
   return (id != kNone)? TGQt::wid(id)->topLevelWidget():0;
}

//______________________________________________________________________________
Window_t    TGQt::wid(TQtClientWidget *widget)
{
   return rootwid(widget);
}
//______________________________________________________________________________
Window_t    TGQt::rootwid(QPaintDevice *dev)
{
   return Window_t(dev);
}
//______________________________________________________________________________
QWidget      *TGQt::wid(Window_t id)
{
   // method to restore (dynamic cast) the QWidget object pointer (if any) from  ROOT windows "id"
   QPaintDevice *dev = 0;
   if (id == (Window_t)kNone || id == (Window_t)(-1) ) return (QWidget *)dev;
   if ( id <= fWidgetArray->MaxId() )
      dev = (*fWidgetArray)[id];
   else
      dev = (QPaintDevice *)id;


#if 0
     if ( dev->devType() != QInternal::Widget) {
        fprintf(stderr," %s %i type=%d QInternal::Widget = %d id =%x id = %d\n", "TGQt::wid", __LINE__
           , dev->devType()
           , QInternal::Widget, id, id );
//           , (const char *)dev->name(), (const char *)dev->className(), QInternal::Widget);
   }
#endif
   assert(dev->devType() == QInternal::Widget);
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
   qDebug() << "----- Window "<<  TGQt::wid(ev.fWindow) << TGQt::wid(ev.fWindow) << " " << TGQt::wid(ev.fWindow)-> objectName();
   fprintf(stderr,"event type =  %x, key or button code %d \n", ev.fType, ev.fCode);
   fprintf(stderr,"fX, fY, fXRoot, fYRoot = %d %d  :: %d %d\n", ev.fX, ev.fY,ev.fXRoot, ev.fYRoot);
}

int TGQt::fgCoinFlag = 0; // no current coin viewer;
int TGQt::fgCoinLoaded = 0; // coint viewer DLL has not been loaded

//______________________________________________________________________________
int TGQt::CoinFlag()
{
  // return the Coin/QGL viewer flag safely
   TQtLock lock;
   int ret = fgCoinFlag;
   return ret;
}

//______________________________________________________________________________
void TGQt::SetCoinFlag(int flag)
{
  // Set the Coin/QGL viewer flag safely
   TQtLock lock;
   fgCoinFlag=flag;
}

//______________________________________________________________________________
void TGQt::SetCoinLoaded() {  fgCoinLoaded = 1; }

//______________________________________________________________________________
Int_t TGQt::IsCoinLoaded(){ return fgCoinLoaded;}

#if ROOT_VERSION_CODE < ROOT_VERSION(5,13,0)
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
# ifdef ROOTICONPATH
   gSystem->ExpandPathName(ROOTICONPATH);
# else
   gSystem->ExpandPathName("$ROOTSYS/icons/");
//   tempIcon =new QPixmap (16,16),
# endif
#endif
   return tempIcon;
}
#endif


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
   // that matches the ROOT image formats
   // those Qt library can not provide
   QString saveType;
   QString defExtension[] = {"cpp","cxx","eps","svg","root","pdf","ps","xml"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,13,0)
                             ,"gif"
#endif
                             ,"C"};
   UInt_t nExt = sizeof(defExtension)/sizeof(const char *);
    UInt_t i = 0;
   for (i=0; i < nExt; i++) {
      if (selector.contains(defExtension[i], Qt::CaseSensitive)) {
         saveType = defExtension[i];
         break;
      }
   }
   if (saveType.contains("C",Qt::CaseInsensitive)) saveType= "cxx";
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
   // if no suitable format found and the selector is empty
   // the default PNG format is returned
   //
   // a special treatment of the "gif" format.
   // If "gif" is not provided with the local Qt installation
   // replace "gif" format with "png" one
   //
   QString saveType="PNG"; // it is the default format
   if (!selector.isEmpty())  {
      QList<QByteArray> formats =  QImageWriter::supportedImageFormats();
      QList<QByteArray>::const_iterator j;
      for (j = formats.constBegin(); j != formats.constEnd(); ++j)
      {
         QString nextFormat =  *j;
         // Trick to count both "jpeg" and "jpg" extenstion
         QString checkString = selector.contains("jpg",Qt::CaseInsensitive) ? "JPEG" : selector;
         if (checkString.contains(nextFormat,Qt::CaseInsensitive) ) {
            saveType = nextFormat;
            break;
         }
      }
      // a special treatment of the "gif" format.
      // If "gif" is not provided with the local Qt installation
      // replace "gif" format with "png" one
      // -- if (saveType.isEmpty() && selector.contains("gif",FALSE)) saveType="PNG";
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
      static TString argvString (
#ifdef ROOTBINDIR
				 ROOTBINDIR "/root.exe"
#else
				 "$ROOTSYS/bin/root.exe"
#endif
				 );
      gSystem->ExpandPathName(argvString);
      static char *argv[] = {(char *)argvString.Data()};
      static int nArg = 1;
      app = new TQtApplication("Qt",nArg,argv);
   }
   return app;
}

//______________________________________________________________________________
void TGQt::PostQtEvent(QObject *receiver, QEvent *event)
{
   // Qt announced that QThread::postEvent to become obsolete and
   // we have to switch to the QAppication instead.
  QApplication::postEvent(receiver,event);
}

//______________________________________________________________________________
TGQt::TGQt() : TVirtualX(),fDisplayOpened(kFALSE),fQPainter(0),fhEvent ()
,fQBrush(),fQPen(),fQtMarker(),fQFont(),fQClientFilter(),fQClientFilterBuffer(0)
,fPointerGrabber(),fCodec(0),fSymbolFontFamily("Symbol"),fQtEventHasBeenProcessed(0)
,fFeedBackMode(kFALSE),fFeedBackWidget(0),fBlockRGB(kFALSE),fUseTTF(kTRUE)      
{
   //*-*-*-*-*-*-*-*-*-*-*-*Default Constructor *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                    ===================
   assert(!fgTQt);
   fgTQt = this;
   gQt   = this;
   fSelectedWindow = fPrevWindow = NoOperation;
}

//______________________________________________________________________________
TGQt::TGQt(const char *name, const char *title) : TVirtualX(name,title),fDisplayOpened(kFALSE)
,fQPainter(0),fhEvent(),fCursors(kNumCursors),fQClientFilter(0),fQClientFilterBuffer(0),fPointerGrabber(0)
,fCodec(0),fSymbolFontFamily("Symbol"),fQtEventHasBeenProcessed(0)
,fFeedBackMode(kFALSE),fFeedBackWidget(0),fBlockRGB(kFALSE),fUseTTF(kTRUE)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*Normal Constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                        ==================                              *-*
   assert(!fgTQt);
   fgTQt = this;
   gQt   = this;
   fSelectedWindow = fPrevWindow = NoOperation;
   CreateQtApplicationImp();
   Init();
}

//______________________________________________________________________________
TGQt::~TGQt()
{
   //*-*-*-*-*-*-*-*-*-*-*-*Default Destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                    ==================
   {  // critical section
      TQtLock lock;
      gVirtualX = gGXBatch;
      gROOT->SetBatch();
      // clear the color map
      QMap<Color_t,QColor*>::const_iterator it;
      for (it = fPallete.begin();it !=fPallete.end();++it) {
         QColor *c = *it; delete c;
      }
       qDeleteAll(fCursors.begin(), fCursors.end());

      delete fQClientFilter;       fQClientFilter = 0;
      delete fQClientFilterBuffer; fQClientFilterBuffer = 0;
      delete fgTextProxy;          fgTextProxy = 0;
 // ---     delete fQPainter; fQPainter = 0;
   }
   // Stop GUI thread
   TQtApplication::Terminate();
   // fprintf(stderr, "TGQt::~TGQt()<------\n");
}

//______________________________________________________________________________
Bool_t TGQt::Init(void* /*display*/)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*Qt GUI initialization-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                        ========================                      *-*
   fprintf(stderr,"** $Id$ this=%p\n",this);
#ifndef R__QTWIN32
   extern void qt_x11_set_global_double_buffer(bool);
//   qt_x11_set_global_double_buffer(false);
#endif

   if(fDisplayOpened)   return fDisplayOpened;
   fSelectedWindow = fPrevWindow = NoOperation;
   fTextAlignH      = 1;
   fTextAlignV      = 1;
   fTextMagnitude   = 1;
   fCharacterUpX    = 1;
   fCharacterUpY    = 1;
   fDrawMode        = QPainter::CompositionMode_Source; // Qt::CopyROP;
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

   //
   // Retrieve the application instance
   //

   // --   fHInstance = GetModuleHandle(NULL);

   //
   // Create cursors
   //
   // Qt::BlankCursor - blank/invisible cursor
   // Qt::BitmapCursor

   fCursors[kBottomLeft]  = new QCursor(Qt::SizeBDiagCursor); // diagonal resize (/) LoadCursor(NULL, IDC_SIZENESW);// (display, XC_bottom_left_corner);
   fCursors[kBottomRight] = new QCursor(Qt::SizeFDiagCursor); // diagonal resize (\) LoadCursor(NULL, IDC_SIZENWSE);// (display, XC_bottom_right_corner);
   fCursors[kTopLeft]     = new QCursor(Qt::SizeFDiagCursor); // diagonal resize (\)  (display, XC_top_left_corner);
   fCursors[kTopRight]    = new QCursor(Qt::SizeBDiagCursor); // diagonal resize (/) LoadCursor(NULL, IDC_SIZENESW);// (display, XC_top_right_corner);
   //fCursors[kBottomSide] = new QCursor(Qt::SplitHCursor);    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_bottom_side);
   //fCursors[kLeftSide]   = new QCursor(Qt::SplitVCursor);    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_left_side);
   //fCursors[kTopSide]    = new QCursor(Qt::SplitHCursor);    // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_top_side);
   //fCursors[kRightSide]  = new QCursor(Qt::SplitVCursor);    // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_right_side);
   fCursors[kBottomSide]  = new QCursor(Qt::SizeVerCursor);   // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_bottom_side);
   fCursors[kLeftSide]    = new QCursor(Qt::SizeHorCursor);   // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_left_side);
   fCursors[kTopSide]     = new QCursor(Qt::SizeVerCursor);   // - horziontal splitting LoadCursor(NULL, IDC_SIZENS);  // (display, XC_top_side);
   fCursors[kRightSide]   = new QCursor(Qt::SizeHorCursor);   // - vertical splitting LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_right_side);

   fCursors[kMove]        = new QCursor(Qt::SizeAllCursor);   //  all directions resize LoadCursor(NULL, IDC_SIZEALL); // (display, XC_fleur);
   fCursors[kCross]       = new QCursor(Qt::CrossCursor);     // - crosshair LoadCursor(NULL, IDC_CROSS);   // (display, XC_tcross);
   fCursors[kArrowHor]    = new QCursor(Qt::SizeHorCursor);   //   horizontal resize LoadCursor(NULL, IDC_SIZEWE);  // (display, XC_sb_h_double_arrow);
   fCursors[kArrowVer]    = new QCursor(Qt::SizeVerCursor);   //  vertical resize LoadCursor(NULL, IDC_SIZENS)  (display, XC_sb_v_double_arrow);
   fCursors[kHand]        = new QCursor(Qt::PointingHandCursor); //  a pointing hand LoadCursor(NULL, IDC_NO);      // (display, XC_hand2);
   fCursors[kRotate]      = new QCursor(Qt::ForbiddenCursor); // - a slashed circle LoadCursor(NULL, IDC_ARROW);    // (display, XC_exchange);
   fCursors[kPointer]     = new QCursor(Qt::ArrowCursor);     // standard arrow cursor  / (display, XC_left_ptr);
   fCursors[kArrowRight]  = new QCursor(Qt::UpArrowCursor);   // - upwards arrow LoadCursor(NULL, IDC_ARROW);   // XC_arrow
   fCursors[kCaret]       = new QCursor(Qt::IBeamCursor);     //  ibeam/text entry LoadCursor(NULL, IDC_IBEAM);   // XC_xterm
   fCursors[kWatch]       = new QCursor(Qt::WaitCursor);      //

   // The default cursor

   fCursor = kCross;

   // Qt object used to paint the canvas
   fQPen     = new TQtPen;
   fQBrush   = new TQtBrush;
   fQtMarker = new TQtMarker;
   fQFont    = new TQtPadFont();
   // ((TGQt *)TGQt::GetVirtualX())->SetQClientFilter(
   fQClientFilter = new TQtClientFilter();

   //  Query the default font for Widget decoration.
   fFontTextCode = "ISO8859-1";
   const char *default_font =
      gEnv->GetValue("Gui.DefaultFont",  "-adobe-helvetica-medium-r-*-*-12-*-*-*-*-*-iso8859-1");
   QFont *dfFont = (QFont *)LoadQueryFont(default_font);
   QApplication::setFont(*dfFont);
   delete dfFont;
   //  define the font code page
   QString fontName(default_font);
   fFontTextCode = fontName.section('-',13). toUpper();
   if  ( fFontTextCode.isEmpty() ) fFontTextCode = "ISO8859-5";
#ifndef R__QTWIN32
   // Check whether "Symbol" font is available
    QFontDatabase fdb;
    QString fontFamily;
    QStringList families = fdb.families();
    Bool_t symbolFontFound = kFALSE;
    Bool_t isXdfSupport = !gSystem->Getenv("QT_X11_NO_FONTCONFIG");
    for ( QStringList::Iterator f = families.begin(); f != families.end(); ++f ) {
        // qDebug() << "Next Symbol font family found:" << *f << fdb.writingSystems(*f);
       if ( (isXdfSupport && (*f).contains(fSymbolFontFamily)
             && fdb.writingSystems(*f).contains(QFontDatabase::Symbol)
         )  || (!isXdfSupport && ((*f) == fSymbolFontFamily)) )
        {
           symbolFontFound = kTRUE;
           qDebug() << "Symbol font family found: " <<  *f;
           if (*f == "Standard Symbols L") { fontFamily = *f; break; }
        }
    }

    if (isXdfSupport && !symbolFontFound) {
// Load the local ROOT font
       QString fontdir =
#ifdef TTFFONTDIR
    		TTFFONTDIR
#else
		  "$ROOOTSYS/fonts"
#endif
        ;
        QString symbolFontFile = fontdir + "/" + QString(fSymbolFontFamily).toLower() + ".ttf";
        symbolFontFound = QFontDatabase::addApplicationFont(symbolFontFile);
    }
    if (!symbolFontFound) {
        fprintf(stderr, "The font \"symbol.ttf\" was not installed yet\n");
         //  provide the replacement and the codec
        fontFamily = fSymbolFontFamily = "Arial";
        qDebug() << " Substitute it with \""<<fontFamily <<"\"";
        qDebug() << " Make sure your local \"~/.fonts.conf\" or \"/etc/fonts/fonts.conf\" file points to \""
                 << 
#ifdef TTFFONTDIR
		TTFFONTDIR
#else
		"$ROOOTSYS/fonts"
#endif
		<< "\" directory to get the proper support for ROOT TLatex class";
        // create a custom codec
        new QSymbolCodec();
    }
    if (symbolFontFound) TQtPadFont::SetSymbolFontFamily(fontFamily.toAscii().data());
#endif
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
       fUseTTF=gEnv->GetValue("Qt.Screen.TTF",kFALSE);
#else
       fUseTTF=gEnv->GetValue("Qt.Screen.TTF",kTRUE);
#endif
   //  printf(" TGQt::Init finsihed\n");
   // Install filter for the desktop
   // QApplication::desktop()->installEventFilter(QClientFilter());
   fWidgetArray =  new TQWidgetCollection();
   fDisplayOpened = kTRUE;
   TQtEventInputHandler::Instance();
   // Add $QTDIR include  path to the  list of includes for ACliC
   // make sure Qt SDK does exist.
   if (gSystem->Getenv("QTDIR")) {
      TString qtdir = "$(QTDIR)/include/";
      // Expand the QTDIR first to avoid the cross-platform issue
      gSystem->ExpandPathName(qtdir);
      TString testQtHeader = qtdir + "Qt/qglobal.h";
      if (!gSystem->AccessPathName((const char *)testQtHeader) ) {
         void *qtdirHandle = gSystem->OpenDirectory(qtdir);
         if (qtdirHandle) {
            TString incpath = " -I"; incpath+=qtdir;
            while(const char *nextQtInclude = gSystem->GetDirEntry(qtdirHandle)) {
               // Skip the hidden directories including the current "." and parent ".."
               if (nextQtInclude[0] != '.')  {
                     incpath += " -I"; incpath+=qtdir; incpath+=nextQtInclude;
               }
            }
            gSystem->FreeDirectory(qtdirHandle);
            gSystem->AddIncludePath((const char*)incpath);
         }
#ifdef R__WIN32
         QString libPath = gSystem->GetLinkedLibs();
         // detect the exact name of the Qt library
         TString qtlibdir= "$(QTDIR)";
         qtlibdir += QDir::separator().toAscii();
         qtlibdir += "lib";

         gSystem->ExpandPathName(qtlibdir);
         QDir qtdir((const char*)qtlibdir);
         if (qtdir.isReadable ()) {
            QStringList qtLibFile =  qtdir.entryList(QStringList("Q*4.lib"),QDir::Files);
            QStringListIterator libFiles(qtLibFile);
            if (libFiles.hasNext()) {
               libPath += " -LIBPATH:\"";libPath += qtlibdir;  libPath += "\" ";
#if 0
               while (libFiles.hasNext()) {
                  QString nf = libFiles.next();
                  if (nf.contains("d4")) continue; // skip the debug version of the libraries
                  libPath += nf.toLocal8Bit().constData();
                  libPath += " ";
               }
#else
                  libPath += "QtCore4.lib QtGui4.lib QtOpenGL4.lib Qt3Support4.lib";
#endif
               gSystem->SetLinkedLibs(libPath.toAscii().data());
            }
         } else {
            qWarning(" Can not open the QTDIR %s",(const char*)qtlibdir);
         }
#endif
      }
   }
   TString newPath =
# ifdef CINTINCDIR
      CINTINCDIR
# else
   "$(ROOTSYS)/cint/"
# endif
     ; newPath += "include";
#ifndef R__WIN32
   newPath += ":";
#else
   newPath += ";";
#endif
   newPath += gSystem->GetDynamicPath();
   gSystem->SetDynamicPath(newPath.Data());
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
Int_t  TGQt::RegisterWid(QPaintDevice *wd)
{
 // register QWidget for the embedded TCanvas
   Int_t id = fWidgetArray->find(wd);
   if (id == -1) id = fWidgetArray->GetFreeId(wd);
   return id;
}
//______________________________________________________________________________
Int_t  TGQt::UnRegisterWid(QPaintDevice *wd)
{
   // unregister QWidget to the TCanvas
   // return  = Root registration Id or zero if the wd was not registered
   return fWidgetArray->RemoveByPointer(wd);
}
//______________________________________________________________________________
Bool_t  TGQt::IsRegistered(QPaintDevice *wd)
{
   // Check whether the object has been registered
   return fWidgetArray->find(wd) == -1 ? kFALSE : kTRUE;
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
   TQtWidget *wd    = 0;
   QWidget   *parent = 0;
   if (window <= fWidgetArray->MaxId() )
      parent = dynamic_cast<TQtWidget *> (iwid(int     (window)));
   else {
      QPaintDevice *dev = dynamic_cast<QPaintDevice *>(iwid(Window_t(window)));
      parent = dynamic_cast<QWidget *>(dev);
   }

 //     QWidget *parent = (window == kDefault) ? 0 : dynamic_cast<QWidget *>(iwid(window));
 //   QWidget *parent = (window == kDefault) ? 0 : (QWidget *)iwid(window);
   wd = new TQtWidget(parent,"virtualx",Qt::FramelessWindowHint,FALSE);
   wd->setCursor(*fCursors[kCross]);
   Int_t id = fWidgetArray->GetFreeId(wd);
   // The default mode is the double buffer mode
   wd->SetDoubleBuffer(1);
   // fprintf(stderr," TGQt::InitWindow %d id=%d device=%p buffer=%p\n",window,id,wd,&wd->GetBuffer());
   return id;
}

//______________________________________________________________________________
Int_t TGQt::OpenPixmap(UInt_t w, UInt_t h)
{
   //*-*  Create a new pixmap object
   QPixmap *obj =  new QPixmap(w,h);
   // fprintf(stderr," TGQt::OpenPixmap %d %d %p\n",w,h,obj);
   return fWidgetArray->GetFreeId(obj);
   // return iwid(obj);
}

//______________________________________________________________________________
const QColor &TGQt::ColorIndex(Color_t ic) const
{
   // Define the QColor object by ROOT color index
   QColor *colorBuffer=0;
   static QColor unknownColor;
   // There are three different ways in ROOT to define RGB.
   // It took 4 months to figure out.
   // See #ifndef R_WIN32 with  TColor::SetRGB method
   if (!fPallete.contains(ic)) {
       Warning("ColorIndex","Unknown color. No RGB component for the index %d was defined\n",ic);
       return unknownColor;
   } else {
      // Make sure the alpha channel was set properly
      // due lack of the TVirtualX interface to account it elsewhere
      TColor *myColor = gROOT->GetColor(ic);
      Float_t a = myColor->GetAlpha();
      colorBuffer = fPallete[ic];
      if (TMath::Abs(colorBuffer->alphaF() - a) > 0.01) {
         colorBuffer->setAlphaF(a);
      }
   }
   return *colorBuffer;
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
   nplanes  = QPixmap::defaultDepth();
}

//______________________________________________________________________________
void  TGQt::ClearWindow()
{
   // Clear current window.
   if (fSelectedWindow && fSelectedWindow != NoOperation)
   {
      if (IsWidget(fSelectedWindow)) {
         ((TQtWidget *)fSelectedWindow)->Erase();
      } else if (IsPixmap(fSelectedWindow) ) {
         ((QPixmap *)fSelectedWindow)->fill(fQBrush->color()); // Qt::transparent);
      } else {
         TQtPainter p(this);
         p.eraseRect(GetQRect(*fSelectedWindow));
      }
   }
}

//______________________________________________________________________________
void  TGQt::ClosePixmap()
{
   // Delete the current pixmap.
   DeleteSelectedObj();
}

//______________________________________________________________________________
void  TGQt::CloseWindow()
{
   // Delete the current window.
   DeleteSelectedObj();
}

//______________________________________________________________________________
void  TGQt::DeleteSelectedObj()
{
    // Delete the current Qt object
  if (fSelectedWindow->devType() == QInternal::Widget) {
     TQtWidget *canvasWidget = dynamic_cast<TQtWidget *>(fSelectedWindow);
     if (canvasWidget) {
        canvasWidget->ResetCanvas();
     }
     QWidget *wrapper = 0;
     if (canvasWidget && (wrapper=canvasWidget->GetRootID())) {
        wrapper->hide();
        DestroyWindow(rootwid(wrapper) );
     } else {
        // check whether we are still registered
        if(UnRegisterWid(fSelectedWindow) != (Int_t) kNone) {
           ((QWidget *)fSelectedWindow)->hide();
           ((QWidget *)fSelectedWindow)->close();
        }
     }
  } else {
     UnRegisterWid(fSelectedWindow);
     delete  fSelectedWindow;
  }
  fClipMap.remove(fSelectedWindow);
  fSelectedWindow = 0;
  fPrevWindow     = 0;
}

//______________________________________________________________________________
QRect TGQt::GetQRect(QPaintDevice &dev)
{
   // Define the rectangle of the current ROOT selection
  QRect res(0,0,0,0);

  switch (dev.devType() ) {
  case QInternal::Widget:
    res = ((TQtWidget*)&dev)->rect();
    break;

  default:
     res.setSize(QSize(dev.width(),dev.height()));
     break;
  };
  return res;
}

//______________________________________________________________________________
void  TGQt::CopyPixmap(int wd, int xpos, int ypos)
{
   // Copy the pixmap wd at the position xpos, ypos in the current window.

   if (!wd || (wd == -1) ) return;
   QPaintDevice *dev = iwid(wd);
   assert(dev->devType() == QInternal::Pixmap);
   QPixmap *src = (QPixmap *)dev;
     //  QPixmap *src = (QPixmap *)(QPaintDevice *)wd;
     //  fprintf(stderr," TGQt::CopyPixmap Selected = %p, Buffer = %p, wd = %p\n",
     //  fSelectedWindow,fSelectedBuffer,iwid(wd));
   if (fSelectedWindow )
   {
      QPaintDevice *dst = fSelectedWindow;
      if (dst == (QPaintDevice *)-1) {
         Error("TGQt::CopyPixmap","Wrong TGuiFactory implementation was provided. Please, check your plugin settings");
         assert(dst != (QPaintDevice *)-1);
      }
      bool itIsWidget = fSelectedWindow->devType() == QInternal::Widget;
      TQtWidget *theWidget = 0;
      if (itIsWidget) { 
          theWidget =  (TQtWidget *)fSelectedWindow;
          dst = theWidget->GetOffScreenBuffer();
      }
      { 
        QPainter paint(dst);
        paint.drawPixmap(xpos,ypos,*src);
      }
      Emitter()->EmitPadPainted(src);
      if (theWidget)  theWidget->EmitCanvasPainted();
   }
}
//______________________________________________________________________________
void TGQt::CopyPixmap(const QPixmap &src, Int_t xpos, Int_t ypos)
{
   // Copy the pixmap p at the position xpos, ypos in the current window.
   if (fSelectedWindow )
   {
      QPaintDevice *dst = fSelectedWindow;
      QPainter paint(dst); paint.drawPixmap(xpos,ypos,src);
   }
}
//______________________________________________________________________________
void TGQt::CreateOpenGLContext(int wd)
{
 // Create OpenGL context for win windows (for "selected" Window by default)
 // printf(" TGQt::CreateOpenGLContext for wd = %x fSelected= %x, threadID= %d \n",wd,fSelectedWindow,
 //    GetCurrentThreadId());
  if (!wd || (wd == -1) ) return;

#ifdef QtGL
    if (!wd)
    {
      SafeCallWin32
         ->W32_CreateOpenGL();
    }
    else
    {
      SafeCallW32(((TQtSwitch *)wd))
         ->W32_CreateOpenGL();
    }
#endif

}

//______________________________________________________________________________
void TGQt::DeleteOpenGLContext(int wd)
{
  // Delete OpenGL context for win windows (for "selected" Window by default)
  if (!wd || (wd == -1) ) return;

#ifdef QtGL
    if (!wd)
    {
      SafeCallWin32
         ->W32_DeleteOpenGL();
    }
    else
    {
      SafeCallW32(((TQtSwitch *)wd))
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
#if QT_VERSION < 0x40000
   static const int Q3=1;
#else
   // Read: http://doc.trolltech.com/4.3/porting4.html
   // "In Qt 4, the result of drawing a QRect with
   //  a pen width of 1 pixel is 1 pixel wider and
   //  1 pixel taller than in Qt 3 "

   static const int Q3=0;
#endif
   if (!fSelectedWindow ) return;
   TQtLock lock;
   // Some workaround to fix issue from TBox::ExecuteEvent case pC. 
   // The reason of the problem has not been found yet.
   // By some reason TBox::ExecuteEvent messes y2 and y1
   if (y2 > y1) {
	   // swap them :-(()
	   int swap = y1; 
	   y1=y2; y2=swap;
   }
   if (x1 > x2) {
	   // swap them :-(()
	   int swap = x1; 
	   x1=x2; x2=swap;
   }
   if ( (fSelectedWindow->devType() ==  QInternal::Widget) && fFeedBackMode && fFeedBackWidget) {
      fFeedBackWidget->SetGeometry(x1,y2,x2-x1,y1-y2,(TQtWidget *)fSelectedWindow);
      if (fFeedBackWidget->isHidden() ) fFeedBackWidget->Show();
      return;
   }

   if ((mode == kHollow) || (fQBrush->style() == Qt::NoBrush) )
   { 
      TQtPainter p(this,TQtPainter::kUpdatePen);
      p.setBrush(Qt::NoBrush);
      p.drawRect(x1,y2,x2-x1+Q3,y1-y2+Q3);
   } else if (fQBrush->GetColor().alpha() ) {
      TQtPainter p(this);
      if (fQBrush->style() != Qt::SolidPattern) p.setPen(fQBrush->GetColor());
      p.fillRect(x1,y2,x2-x1,y1-y2,*fQBrush);
   }
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
   // Draw a cell array. The drawing is done with the pixel precision
   // if (X2-X1)/NX (or Y) is not a exact pixel number the position of
   // the top rigth corner may be wrong.

   TQtLock lock;
   if (fSelectedWindow)
   {
      int i,j,icol,w,h,current_icol,lh;

      current_icol = -1;
      w            = TMath::Max((x2-x1)/(nx),1);
      h            = TMath::Max((y1-y2)/(ny),1);
      lh           = y1-y2;

      if (w+h == 2)
      {
         //*-*  The size of the box is equal a single pixel
         TQtPainter p(this,TQtPainter::kUpdatePen);
         for ( i=x1; i<x1+nx; i++){
            for (j = 0; j<ny; j++){
               icol = ic[i+(nx*j)];
               if (current_icol != icol) {
                  current_icol = icol;
                  p.setPen(ColorIndex(current_icol));
               }
               p.drawPoint(i,y1-j);
            }
         }
      }
      else
      {
         //*-* The shape of the box is a rectangle
         QRect box(x1,y1,w,h);
         TQtPainter p(this,TQtPainter::kNone);
         for ( i=0; i<nx; i++ ) {
            for ( j=0; j<ny; j++ ) {
               icol = ic[i+(nx*j)];
               if(icol != current_icol){
                  current_icol = icol;
                  p.setBrush(ColorIndex(current_icol));
               }
               p.drawRect(box);
               box.translate(0,-h);   // box.top -= h;
            }
            box.translate(w,lh);
         }
      }
   }
}

//______________________________________________________________________________
void  TGQt::DrawFillArea(int n, TPoint *xy)
{
   // Fill area described by polygon.
   // n         : number of points
   // xy(2,n)   : list of points

   TQtLock lock;
   if (fSelectedWindow && n>0)
   {
      TQtPainter p(this);
      if (fQBrush->style() == Qt::SolidPattern) p.setPen(Qt::NoPen);
      QPolygon qtPoints(n);
      TPoint *rootPoint = xy;
      for (int i =0;i<n;i++,rootPoint++) qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
      p.drawPolygon(qtPoints);
   }
}

//______________________________________________________________________________
void  TGQt::DrawLine(int x1, int y1, int x2, int y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

  TQtLock lock;
  if (fSelectedWindow) {
     TQtToggleFeedBack  feedBack(this);
     feedBack.painter().drawLine(x1,y1,x2,y2);
#if 0
     if (x1> 1000) {
        // Qt 4.5.x bug
       qDebug() << "TGQt::DrawLine " << " x1=" << x1 
                                   << " y1=" << y1
                                   << " x2=" << x2
                                   << " y2=" << y2;
       //assert(0 && "Weird coordinate");
     }
#endif
  }
}

//______________________________________________________________________________
void  TGQt::DrawPolyLine(int n, TPoint *xy)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points

  TQtLock lock;
  if (fSelectedWindow)  {
     TQtToggleFeedBack  feedBack(this);
     QPolygon qtPoints(n);
     TPoint *rootPoint = xy;
     for (int i =0;i<n;i++,rootPoint++) qtPoints.setPoint(i,rootPoint->fX,rootPoint->fY);
     feedBack.painter().drawPolyline(qtPoints);
  }
}

//______________________________________________________________________________
void  TGQt::DrawPolyMarker(int n, TPoint *xy)
{
   // Draw n markers with the current attributes at position x, y.
   // n    : number of markers to draw
   // xy   : x,y coordinates of markers
   TQtLock lock;
   if (fSelectedWindow) {
      TQtPainter p(this,TQtPainter::kNone);
		fQtMarker->DrawPolyMarker(p,n,xy);
   }
}
//______________________________________________________________________________
TQtTextProxy  *TGQt::TextProxy()
{
   // Get the text proxy implementation pointer
   return fgTextProxy;
}
//______________________________________________________________________________
void  TGQt::SetTextProxy(TQtTextProxy  *proxy)
{
   // Set the text proxy implementation pointer
   fgTextProxy = proxy;
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


   // fprintf(stderr,"TGQt::DrawText: %s\n", text);
   if (text && text[0]) {
      TQtLock lock;
      fQFont->SetTextMagnify(mgn);
      TQtPainter p(this,TQtPainter::kUpdateFont);
      p.setPen(ColorIndex(fTextColor));
      p.setBrush(ColorIndex(fTextColor));
      unsigned int w=0;unsigned int h=0; unsigned int d = 0;
      bool textProxy = false;
      TQtTextCloneProxy proxy;
      if (fgTextProxy) {
         proxy->clear();
         QFontInfo fi(*fQFont);
         proxy->setBaseFontPointSize(fi.pointSize());
         proxy->setForegroundColor(ColorIndex(fTextColor));
         if ( ( textProxy = proxy->setContent(text) ) ) {
             w = proxy->width();
             h = proxy->height();
         }
      } 
      if (!textProxy) {
         QFontMetrics metrics(*fQFont);
         QRect bRect = metrics.boundingRect(text);
         w = bRect.width();
         h = bRect.height();
         d = metrics.descent();
      }
      p.translate(x,y);
      if (TMath::Abs(angle) > 0.1 ) p.rotate(-angle);
      int dx =0; int dy =0;
      
      switch( fTextAlignH ) {
           case 2: dx = -int(w/2);                  // h center;
              break;
           case 3: dx = -int(w);                    //  Right;
              break;
      };
      switch( fTextAlignV ) {
          case 2: dy = h/2 - d; // v center // metrics.strikeOutPos();
             break;
          case 3: dy = h   - d; // AlignTop;
      };
      if (textProxy)
         proxy->paint(&p,dx,-dy);
      else
         p.drawText (dx, dy, GetTextDecoder()->toUnicode (text));
   }
}

//______________________________________________________________________________
void  TGQt::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   // Return character up vector.

   TQtLock lock;
   chupx = fCharacterUpX;
   chupy = fCharacterUpY;
}

//______________________________________________________________________________
QPaintDevice *TGQt::GetDoubleBuffer(QPaintDevice *dev)
{
   // Query the pointer to the dev offscreen buffer if any

   QPaintDevice *buffer = 0;
   if (dev) {
       TQtWidget *widget = dynamic_cast<TQtWidget *>(dev);
       buffer = widget && widget->IsDoubleBuffered() ? widget->SetBuffer().Buffer() : 0;
   }
   return buffer;
}
//______________________________________________________________________________
Int_t  TGQt::GetDoubleBuffer(Int_t wd)
{
   // Query the double buffer value for the window wd.
   // return pointer to the off-screen buffer if any

   if (wd == -1 || wd == kDefault ) return 0;
   assert(0);
   QPaintDevice *dev = iwid(wd);
   TQtWidget *widget = dynamic_cast<TQtWidget *>(dev);
   return  Int_t(widget && widget->IsDoubleBuffered());
}

//______________________________________________________________________________
void  TGQt::GetGeometry(int wd, int &x, int &y, unsigned int &w, unsigned int &h)
{
   // Returns the global cooordinate of the window "wd"
   QRect devSize(0,0,0,0);
   if( wd == -1 || wd == 0 || wd == kDefault)
   {
      QDesktopWidget *d = QApplication::desktop();
      devSize.setWidth (d->width() );
      devSize.setHeight(d->height());
   } else {
      QPaintDevice  *dev = iwid(wd);
      if (dev) {
         if ( dev->devType() == QInternal::Widget) {
            TQtWidget &thisWidget = *(TQtWidget *)dev;
            if (thisWidget.GetRootID() ) {
               // we are using the ROOT Gui factory
               devSize = thisWidget.parentWidget()->geometry();
            } else{
               devSize = thisWidget.geometry();
            }
            devSize.moveTopLeft(thisWidget.mapToGlobal(QPoint(0,0)));
         } else {
            devSize = GetQRect(*dev);
         }
      }
   }
   x = devSize.left();
   y = devSize.top();
   w = devSize.width();
   h = devSize.height();
 //  fprintf(stderr," 2. TGQt::GetGeometry %d %d %d %d %d\n", wd, x,y,w,h);
}

//______________________________________________________________________________
const char *TGQt::DisplayName(const char *){ return "localhost"; }

//______________________________________________________________________________
ULong_t  TGQt::GetPixel(Color_t cindex)
{
   // Return pixel value associated to specified ROOT color number.
   // see: GQTGUI.cxx:QtColor() also
   ULong_t rootPixel = 0;
   const QColor &color = ColorIndex(UpdateColor(cindex));
#ifdef R__WIN32
   rootPixel =                    ( color.blue () & 255 );
   rootPixel = (rootPixel << 8) | ( color.green() & 255 ) ;
   rootPixel = (rootPixel << 8) | ( color.red  () & 255 );
#else
   rootPixel =                    ( color.red ()  & 255 );
   rootPixel = (rootPixel << 8) | ( color.green() & 255 ) ;
   rootPixel = (rootPixel << 8) | ( color.blue  ()& 255 );
#endif
   return rootPixel;
}

//______________________________________________________________________________
void  TGQt::GetRGB(int index, float &r, float &g, float &b)
{
   // Get rgb values for color "index".
   r = g = b = 0;
   TQtLock lock;
   if (fSelectedWindow != NoOperation) {
      qreal R,G,B;
      const QColor &color = *fPallete[index];
      color.getRgbF(&R,&G,&B);
      r = R; g = G; b = G;
   }
}

//______________________________________________________________________________
const QTextCodec *TGQt::GetTextDecoder()
{
   static  QTextCodec  *fGreekCodec = 0;
   QTextCodec  *codec = 0;
   if (!fCodec) {
      fCodec =  QTextCodec::codecForName(fFontTextCode.toAscii()); //CP1251
      if (!fCodec)
         fCodec=QTextCodec::codecForLocale();
      else
         QTextCodec::setCodecForLocale(fCodec);
   }
   codec = fCodec;
   if (fTextFont/10 == 12 ) {
        // We expect the Greek letters and should apply the right Codec
      if (!fGreekCodec) {
         if (QString(fSymbolFontFamily).contains("Symbol")) {
            fGreekCodec = (fFontTextCode == "ISO8859-1") ? fCodec:
                          QTextCodec::codecForName("ISO8859-1"); //iso8859-1
         } else {
            fGreekCodec  = QTextCodec::codecForName("symbol"); // ISO8859-7
         }
      }
      if (fGreekCodec) codec=fGreekCodec;
   }
   return codec;
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

   TQtLock lock;
   if (fQFont) {
      bool textProxy = false;
      if (fgTextProxy) {
         TQtTextCloneProxy proxy;
         proxy->clear();
         QFontInfo fi(*fQFont);
         proxy->setBaseFontPointSize(fi.pointSize());
         if ( (textProxy = proxy->setContent(mess)) ) {
            w = proxy->width();
            h = proxy->height();
         }
      }
      if (!textProxy) {
         QSize textSize = QFontMetrics(*fQFont).size(Qt::TextSingleLine,GetTextDecoder()->toUnicode(mess)) ;
         w = textSize.width();
         h = (unsigned int)(textSize.height());
      }
      // qDebug() << "  TGQt::GetTextExtent  w=" <<w <<" h=" << h << "font = " 
      //         << fTextFont << " size =" << fTextSize 
      //         << mess;
   }
}
//______________________________________________________________________________
Int_t TGQt::GetFontAscent() const
{
   // Returns ascent of the current font (in pixels).
   // The ascent of a font is the distance from the baseline 
   // to the highest position characters extend to
   Int_t ascent = 0;
   if (fQFont) {
      QFontMetrics fm(*fQFont);
      ascent = fm.ascent(); //+fm.descent();
   }
   return ascent;
}

//______________________________________________________________________________
Int_t TGQt::GetFontDescent() const 
{ 
   // Returns the descent of the current font (in pixels.
   // The descent is the distance from the base line 
   // to the lowest point characters extend to.
   Int_t descent = 0;
   if (fQFont) {
      QFontMetrics fm(*fQFont);
      descent = fm.descent();
   }
   return descent;
}

//______________________________________________________________________________
Bool_t  TGQt::HasTTFonts() const {return fUseTTF;}

//______________________________________________________________________________
void  TGQt::MoveWindow(Int_t wd, Int_t x, Int_t y)
{
   // Move the window wd.
   // wd  : Window identifier.
   // x    : x new window position
   // y    : y new window position

   if (wd != -1 && wd != 0 && wd != kDefault)
   {
      QPaintDevice *widget = iwid(wd);
      assert(widget->devType() == QInternal::Widget );
      ((TQtWidget *)widget)->move(x,y);
   }
}

//______________________________________________________________________________
void  TGQt::PutByte(Byte_t )
{ // deprecated
}

//______________________________________________________________________________
void  TGQt::QueryPointer(int &ix, int &iy)
{
   // Query pointer position.
   // ix       : X coordinate of pointer
   // iy       : Y coordinate of pointer
   QPoint pos = QCursor::pos();
   ix = pos.x(); iy = pos.y();
}

//______________________________________________________________________________
Pixmap_t TGQt::ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id)
{
   // If id is NULL - loads the specified gif file at position [x0,y0] in the
   // current window. Otherwise creates pixmap from gif file

   Int_t thisId = 0;
   QPixmap *pix = new QPixmap( QString (file) );
   if ( pix->isNull () ) { delete pix; pix = 0;         }
   else {
      thisId=fWidgetArray->GetFreeId(pix);
      if (!id ) { CopyPixmap(thisId,x0,y0); fWidgetArray->DeleteById(thisId); thisId = 0;}
   }
   return thisId;
}

//______________________________________________________________________________
Int_t  TGQt::RequestLocator(Int_t /*mode*/, Int_t /*ctyp*/, Int_t &/*x*/, Int_t &/*y*/)
{
   // deprecated
   return 0;
}
//______________________________________________________________________________
  class requestString : public QDialog {
  public:
    QString   fText;
    QLineEdit fEdit;
    requestString(const char *text="") : QDialog(0
          , Qt::FramelessWindowHint 
          | Qt::WindowStaysOnTopHint
          | Qt::Popup)
          , fText(text),fEdit(this)
    {
       setModal(true);
       connect(&fEdit,SIGNAL( returnPressed () ), this, SLOT( accept() ));
    }
    ~requestString(){;}
  };
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
  int  res = QDialog::Rejected;
  if (fSelectedWindow->devType() == QInternal::Widget ) {
     TQtWidget *w = (TQtWidget *)fSelectedWindow;
     static requestString reqDialog;
     reqDialog.fEdit.setText(QString(text).trimmed());
     int yFrame = reqDialog.frameGeometry().height() - reqDialog.geometry().height() + reqDialog.fontMetrics().height();
     reqDialog.move(w->mapToGlobal(QPoint(x,y-yFrame)));
     if (QClientFilter() && QClientFilter()->PointerGrabber() ) {
        // suspend the mouse grabbing for a while
//        QClientFilter()->SetPointerGrabber(0);
        QClientFilter()->PointerGrabber()->DisactivateGrabbing();
     }
     res = reqDialog.exec();
     if (res == QDialog::Accepted ) {
        // save the currect test font to select the proper codec
        Font_t textFontSave =  fTextFont;
        fTextFont = 62;
        QByteArray obr = GetTextDecoder()->fromUnicode(reqDialog.fEdit.text());
        const char *r = obr.constData();
        qstrcpy(text, (const char *)r);
        // restore the font
        fTextFont = textFontSave;
     }
     reqDialog.hide();
     if (QClientFilter() && QClientFilter()->PointerGrabber()) {
        // Restore the grabbing
//        QClientFilter()->SetPointerGrabber(fPointerGrabber);
        QClientFilter()->PointerGrabber()->ActivateGrabbing();
     }
  }
  return res == QDialog::Accepted ? 1 : 0;
}

//______________________________________________________________________________
void  TGQt::RescaleWindow(int wd, UInt_t w, UInt_t h)
{
   // Rescale the window wd.
   // wd  : Window identifier
   // w    : Width
   // h    : Heigth

   TQtLock lock;
   if (wd && wd != -1 && wd != kDefault )
   {
      QPaintDevice *widget = iwid(wd);
      if (widget->devType() == QInternal::Widget )
      {
         if (QSize(w,h) != ((TQtWidget *)widget)->size()) {
            // fprintf(stderr," TGQt::RescaleWindow(int wd, UInt_t w=%d, UInt_t h=%d)\n",w,h);
            ((TQtWidget *)widget)->resize(w,h);
         }
      }
   }
}

//______________________________________________________________________________
Int_t  TGQt::ResizePixmap(int wd, UInt_t w, UInt_t h)
{
   // Resize a pixmap.
   // wd : pixmap to be resized
   // w,h : Width and height of the pixmap

   TQtLock lock;
   if (wd && wd != -1 && wd != kDefault )
   {
      QPaintDevice *pixmap = iwid(wd);
      if (pixmap->devType() == QInternal::Pixmap )
      {
         if (QSize(w,h) != ((QPixmap *)pixmap)->size()) {
             QPixmap *newpix =  new QPixmap(w,h);
             newpix->fill();
             fWidgetArray->ReplaceById(wd,newpix);
             if (fSelectedWindow == pixmap) fSelectedWindow = newpix;
            // fprintf(stderr," \n --- > Pixmap has been resized ,< --- \t  %p\n",pixmap);
         }
      }
   }
   return 1;
}

//______________________________________________________________________________
void  TGQt::ResizeWindow(int /* wd */)
{
   // Resize the current window if necessary.
   // No implementation is required under Qt.

   return;
}

//______________________________________________________________________________
void   TGQt::SelectPixmap(Int_t qpixid){ SelectWindow(qpixid);}

//______________________________________________________________________________
void  TGQt::SelectWindow(int wd)
{
   // Select window to which subsequent output is directed.
   // fprintf(stderr," TGQt::SelectWindow %d \n", wd);
   // Don't select things twice
   QPaintDevice *dev = 0;
   if (wd == -1 || wd == (int) kNone) {
       fSelectedWindow = NoOperation;
      //return;
   } else {
      dev = iwid(wd);
      fSelectedWindow = dev ? dev : NoOperation;
   }
   if (fPrevWindow != fSelectedWindow) 
      fPrevWindow     = fSelectedWindow;
}

//______________________________________________________________________________
void  TGQt::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   // Set character up vector.

   TQtLock lock;
   if (chupx == fCharacterUpX  && chupy == fCharacterUpY) {

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
}

//______________________________________________________________________________
void  TGQt::SetClipOFF(Int_t /*wd*/)
{
   // Turn off the clipping for the window wd.
   // deprecated
   // fQPainter->setClipping(FALSE);
}

//______________________________________________________________________________
void  TGQt::SetClipRegion(int wd, int x, int y, UInt_t w, UInt_t h)
{
   // Set clipping region for the window wd.
   // wd        : Window indentifier
   // x,y        : origin of clipping rectangle
   // w,h        : size of clipping rectangle;

   QRect rect(x,y,w,h);
   TQtLock lock;
   fClipMap.remove(iwid(wd));
   fClipMap.insert(iwid(wd),rect);
}

//____________________________________________________________________________
void  TGQt::SetCursor(Int_t wd, ECursor cursor)
{
   // Set the cursor.
   fCursor = cursor;
   if (wd && wd != -1 && wd != kDefault)
   {
      QPaintDevice *widget = iwid(wd);
      if ( TQtWidget *w = (TQtWidget *)IsWidget(widget) )
         w->setCursor(*fCursors[fCursor]);
   }
}

//______________________________________________________________________________
void  TGQt::SetDoubleBuffer(int wd, int mode)
{
   // Set the double buffer on/off on window wd.
   // wd  : Window identifier.
   //        999 means all the opened windows.
   // mode : 1 double buffer is on
   //        0 double buffer is off
   if (wd == -1 || wd == kDefault) return;
   QPaintDevice *dev = iwid(wd);
   TQtWidget *widget = 0;
   if (dev && (widget = (TQtWidget *)IsWidget(dev) ))  {
      widget->SetDoubleBuffer(mode);
      // fprintf(stderr," TGQt::SetDoubleBuffer \n");
   }
}

//______________________________________________________________________________
void  TGQt::SetDrawMode(TVirtualX::EDrawMode mode)
{
   // Set the drawing mode.
   // mode : drawing mode

   // Map EDrawMode    { kCopy = 1, kXor, kInvert };
   Bool_t feedBack =  (mode==kInvert);
   if (feedBack != fFeedBackMode) {
      fFeedBackMode = feedBack;
      if (fFeedBackMode) {
         // create
         if (!fFeedBackWidget) {
            fFeedBackWidget = new TQtFeedBackWidget;
            fFeedBackWidget->setFrameStyle(QFrame::Box);
         }
	 // This makes no sense on X11 yet due the  
	 // TQtWidget::setAttribute(Qt::WA_PaintOnScreen) flag
	 // TQtWidget keeps painting itself over the feedback windows. Wierd !!!
         // reparent if needed
         fFeedBackWidget->SetParent(0);
         fFeedBackWidget->SetParent((TQtWidget *)fSelectedWindow);
      } else if (fFeedBackWidget) {
         fFeedBackWidget->Hide();
      }
   }
#if 0
   QPainter::CompositionMode newMode = QPainter::CompositionMode_Source;
   switch (mode) {
    case kCopy:   newMode = QPainter::CompositionMode_Source; break;
    case kXor:    newMode = QPainter::CompositionMode_Xor;  break;
    case kInvert: newMode = QPainter::CompositionMode_Destination;  break;
    default:      newMode = QPainter::CompositionMode_Source; break;
   };
   if (newMode != fDrawMode)
   {
      fDrawMode = newMode;
//      if (fQPainter->isActive() && (fQPainter->device()->devType() !=  QInternal::Widget ))
      if (fQPainter->isActive() && (fQPainter->device()->devType() ==  QInternal::Image ))
      {
         fQPainter->setCompositionMode(fDrawMode);
     }
     // sfprintf(stderr,"TGQt::SetDrawMode \n");
   }
#endif
}

//______________________________________________________________________________
void  TGQt::SetFillColor(Color_t cindex)
{
   // Set color index for fill areas.

   if (fFillColor != cindex )
      fQBrush->SetColor(fFillColor = UpdateColor(cindex));
}

//______________________________________________________________________________
void  TGQt::SetFillStyle(Style_t fstyle)
{
   // Set fill area style.
   // fstyle   : compound fill area interior style
   //    fstyle = 1000*interiorstyle + styleindex

  //  The current fill area color is used to paint some pixels in a small
  //  rectangle the other pixels are not paint.
  //    Olivier Couet
  //            Thursday, July 14, 2005

   if (fFillStyle != fstyle)
      fQBrush->SetStyle(fFillStyle = fstyle);
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
    fLineColor = UpdateColor(cindex);
    if (fLineColor >= 0) fQPen->SetLineColor(fLineColor);
  }
}

//______________________________________________________________________________
void  TGQt::SetLineType(int n, int*dash)
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
   fQPen->SetLineType(n,dash);
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
   // Copy/Paste from TGX11::SetLineStyle (it is called "subclassing")
   // Set line style.

   if (fLineStyle != linestyle) { //set style index only if different
      fLineStyle = linestyle;
      fQPen->SetLineStyle(linestyle);
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
      if (fLineWidth >= 0 ) fQPen->SetLineWidth(fLineWidth);
   }
}

//______________________________________________________________________________
void  TGQt::SetMarkerColor( Color_t cindex)
{
   //*-*-*-*-*-*-*-*-*-*-*Set color index for markers*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ===========================
   //*-*  cindex : color index defined my IXSETCOL
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

	if (fQtMarker->GetColor() != cindex)		
		fQtMarker->SetColor(fMarkerColor = UpdateColor(cindex));
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
   markerstyle  = TMath::Abs(markerstyle);
   if (markerstyle%1000 >= 31) return;
   fMarkerStyle = markerstyle%1000;
   Style_t penWidth = markerstyle-fMarkerStyle;
   Int_t im = Int_t(4*fMarkerSize + 0.5);
   switch (fMarkerStyle) {

case 2:
   //*-*--- + shaped marker
   shape[0].SetX(-im); shape[0].SetY( 0);
   shape[1].SetX(im);  shape[1].SetY( 0);
   shape[2].SetX(0) ;  shape[2].SetY( -im);
   shape[3].SetX(0) ;  shape[3].SetY( im);
   SetMarkerType(4+penWidth,4,shape);
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
   SetMarkerType(4+penWidth,8,shape);
   break;

case 4:
case 24:
   //*-*--- O shaped marker
   SetMarkerType(0+penWidth,im*2,shape);
   break;

case 5:
   //*-*--- X shaped marker
   im = Int_t(0.707*Float_t(im) + 0.5);
   shape[0].SetX(-im);  shape[0].SetY(-im);
   shape[1].SetX( im);  shape[1].SetY( im);
   shape[2].SetX(-im);  shape[2].SetY( im);
   shape[3].SetX( im);  shape[3].SetY(-im);
   SetMarkerType(4+penWidth,4,shape);
   break;

case  6:
   //*-*--- + shaped marker (with 1 pixel)
   shape[0].SetX(-1);  shape[0].SetY( 0);
   shape[1].SetX( 1);  shape[1].SetY( 0);
   shape[2].SetX( 0);  shape[2].SetY(-1);
   shape[3].SetX( 0);  shape[3].SetY( 1);
   SetMarkerType(4+penWidth,4,shape);
   break;

case 7:
   //*-*--- . shaped marker (with 9 pixel)
   shape[0].SetX(-1);  shape[0].SetY( 1);
   shape[1].SetX( 1);  shape[1].SetY( 1);
   shape[2].SetX(-1);  shape[2].SetY( 0);
   shape[3].SetX( 1);  shape[3].SetY( 0);
   shape[4].SetX(-1);  shape[4].SetY(-1);
   shape[5].SetX( 1);  shape[5].SetY(-1);
   SetMarkerType(4+penWidth,6,shape);
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
   SetMarkerType(2+penWidth,4,shape);
   break;
case 26:
   //*-*--- HIGZ open triangle up
   shape[0].SetX(-im);  shape[0].SetY( im);
   shape[1].SetX( im);  shape[1].SetY( im);
   shape[2].SetX(  0);  shape[2].SetY(-im);
   //     shape[3].SetX(-im);  shape[3].SetY( im);
   SetMarkerType(2+penWidth,3,shape);
   break;
case 27: {
   //*-*--- HIGZ open losange
   Int_t imx = Int_t(2.66*fMarkerSize + 0.5);
   shape[0].SetX(-imx); shape[0].SetY( 0);
   shape[1].SetX(  0);  shape[1].SetY(-im);
   shape[2].SetX(imx);  shape[2].SetY( 0);
   shape[3].SetX(  0);  shape[3].SetY( im);
   //     shape[4].SetX(-imx); shape[4].SetY( 0);
   SetMarkerType(2+penWidth,4,shape);
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
   SetMarkerType(2+penWidth,12,shape);
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
   SetMarkerType(3+penWidth,10,shape);
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
   SetMarkerType(2+penWidth,10,shape);
   break;
         }

case 31:
   //*-*--- HIGZ +&&x (kind of star)
   SetMarkerType(1,im*2,shape);
   break;
default:
   //*-*--- single dot
   SetMarkerType(0+penWidth,0,shape);
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
int  TGQt::UpdateColor(int cindex)
{
   // [protected] update the color parameters if needed.
#define BIGGEST_RGB_VALUE 255  // 65535
   //  if (fSelectedWindow == NoOperation) return;
   if (cindex >= 0 ) {
      //    if (cindex >= fPallete.size()) fPallete.resize(cindex+1);
      //    fPallete[cindex].setRgb((r*BIGGEST_RGB_VALUE)
      if (!fPallete.contains(cindex)) {
         // qDebug() << "TGQt::UpdateRGB: Add the new index:" << cindex;
         fBlockRGB = kTRUE; // to eliminate a recursive setting via TGQt::SetRGB()
         TColor *rootColor = gROOT->GetColor(cindex);
         fBlockRGB = kFALSE;
         if (rootColor) {
             float r,g,b,a;
             rootColor->GetRGB(r,g,b);
             a= rootColor->GetAlpha();

             fPallete[cindex] =  new QColor(
                int(r*BIGGEST_RGB_VALUE+0.5)
               ,int(g*BIGGEST_RGB_VALUE+0.5)
               ,int(b*BIGGEST_RGB_VALUE+0.5)
               ,int(a*BIGGEST_RGB_VALUE+0.5)
            );
         }
      }
   }
   return cindex;
}
//______________________________________________________________________________
void  TGQt::SetRGB(int cindex, float r, float g, float b)
{
#define BIGGEST_RGB_VALUE 255  // 65535
   //  if (fSelectedWindow == NoOperation) return;
   if ( !fBlockRGB && cindex >= 0 ) {
      //    if (cindex >= fPallete.size()) fPallete.resize(cindex+1);
      //    fPallete[cindex].setRgb((r*BIGGEST_RGB_VALUE)
       // qDebug() << "TGQt::SetRGB: Add the new index: " << cindex;
       QMap<Color_t,QColor*>::iterator i = fPallete.find(cindex);
       if (i != fPallete.end()) {
          delete i.value();
          fPallete.erase(i);
       }
       fPallete[cindex] =  new QColor(
          int(r*BIGGEST_RGB_VALUE+0.5)
         ,int(g*BIGGEST_RGB_VALUE+0.5)
         ,int(b*BIGGEST_RGB_VALUE+0.5)
         );
   }
}
//______________________________________________________________________________
void  TGQt::SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b, Float_t a)
{
   // Set the color with the alpha component (supported wuth Qt 4 only)
   SetRGB(cindex, r, g,b);
   SetAlpha(cindex,a);
}
//______________________________________________________________________________
void  TGQt::SetAlpha(Int_t cindex, Float_t a)
{
   // Add  the alpha component (supported with Qt 4 only)
   if (cindex < 0 || a < 0 ) return;
   QColor *color = fPallete[cindex];
   if (color) color->setAlphaF(a);

}
//______________________________________________________________________________
void  TGQt::GetRGBA(Int_t cindex, Float_t &r, Float_t &g, Float_t &b, Float_t &a)
{
   // Return RGBA components for the color cindex
   GetRGB(cindex,r,g,b);
   a = GetAlpha(cindex);
}
//______________________________________________________________________________
Float_t TGQt::GetAlpha(Int_t cindex)
{
   // Return Alpha component for the color cindex
   if (cindex < 0 ) return 1.0;
   const QColor *color = fPallete[cindex];
   return (Float_t)color->alphaF();
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

   fTextAlign = Qt::AlignLeft;
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
   fTextColor = UpdateColor(cindex);
   if (cindex < 0) return;
}

//______________________________________________________________________________
Int_t  TGQt::SetTextFont(char* /*fontname*/, TVirtualX::ETextSetMode /*mode*/)
{
   // Set text font to specified name.
   // mode       : loading flag
   // mode=kCheck = 0     : search if the font exist (kCheck)
   // mode= kLoad = 1     : search the font and load it if it exists (kLoad)
   // font       : font name
   //
   // Set text font to specified name. This function returns 0 if
   // the specified font is found, 1 if not.

   // Qt takes care to make sure the proper font is loaded and scaled.
   return 0;
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
   fQFont->SetTextFont(fontnumber);
   fTextFontModified = 1;
}

//______________________________________________________________________________
void  TGQt::SetTextSize(Float_t textsize)
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*Set current text size*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                      =====================
   if ( fTextSize != textsize ) {
      fTextSize = textsize;
      if (fTextSize > 0) {
         fQFont->SetTextSize(textsize);
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
      ((TQtWidget *)fSelectedWindow)->topLevelWidget()-> setWindowTitle(GetTextDecoder()->toUnicode(title));
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
#ifndef R__WIN32
      // X11 needs "repaint" operation to be forced by some reason
//      qDebug() <<  " TGQt::UpdateWindow " 
//               <<  " Please check whether the \"" 
//               <<  "QCoreApplication::processEvents(QEventLoop::ExcludeUserInput | QEventLoop::ExcludeSocketNotifiers, 200);"
//               << "\" still needed !!!";
#endif
   }
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
void  TGQt::WritePixmap(int wd, UInt_t w, UInt_t h, char *pxname)
{
   // Write the pixmap wd in the bitmap file pxname in JPEG.
   // wd         : Pixmap address
   // w,h         : Width and height of the pixmap.
   //               if w = h = -1 the size of the pimxap is equal the size the wd size
   // pxname      : pixmap file name
   //               The format is defined by the file name extension
   //               like "png","jpg","bmp"  . . .
   //               If no or some unknown extension is provided then
   //               the "png" format is used by default
   // --
   // Take in account the special ROOT filename syntax 26.12.2006 vf
   //               "gif+NN" - an animated GIF file is produced, where NN is delay in 10ms units

   if (!wd || (wd == -1) ) return;

   QPaintDevice &dev = *iwid(wd);
   QPixmap grabWidget;
   QPixmap *pix=0;
   switch (dev.devType()) {
   case QInternal::Widget: {
        TQtWidget *thisWidget = (TQtWidget*)&dev;
        if (thisWidget->IsDoubleBuffered() ) {
           pix = ((const TQtWidget*)&dev)->GetOffScreenBuffer();
        } else {
           // Grab the widget rectangle area directly from the screen
           // The image may contain the "alien" pieces !
           grabWidget = QPixmap::grabWindow(thisWidget->winId());
           pix = &grabWidget;
        }
     }
     break;

   case QInternal::Pixmap: {
      pix = (QPixmap *)&dev;
      break;
                          }
   case QInternal::Picture:
   case QInternal::Printer:
   default: assert(0);
     break;
   };
   if (pix) {
      // Create intermediate pixmap to stretch the original one if any
      QPixmap *finalPixmap = 0;
      if ( ( (h == w) && (w == UInt_t(-1) ) ) || ( QSize(w,h) == pix->size()) ) {
         finalPixmap = new QPixmap(*pix);
      }  else  {
         finalPixmap = new QPixmap(pix->scaled(w,h));
      }
      // Detect the special case "gif+"
      QString fname = pxname;
      int plus = fname.indexOf("+");
      if (plus>=0) fname = fname.left(plus);

      //  define the file extension
      QString saveType = QtFileFormat(QFileInfo(fname).suffix());
      //Info("WritePixmap"," type %s name = %s plus =  %d\n", (const char *)saveType,
      //   (const char*) fname, plus);
      if (saveType.isEmpty()) saveType="PNG";

      else if (QFileInfo(fname).suffix() == "gif") {
         // TrollTech doesn't allow the  GIF writting due
         // the patent problem.
         Int_t saver = gErrorIgnoreLevel;
         gErrorIgnoreLevel = kFatal;
         TImage *img = TImage::Create();
         if (img) {
            img->SetImage(Pixmap_t(rootwid(finalPixmap)),0);
            img->WriteImage(pxname,
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,13,0)
                  plus>=0 ? TImage::kAnimGif: TImage::kGif);
#else
                  TImage::kGif);
#endif
            delete img;
         }
         gErrorIgnoreLevel = saver;
      } else {
         if (plus>=0) fname = GetNewFileName(fname);
         finalPixmap->save(fname,saveType.toAscii().data());
      }
      delete finalPixmap;
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
Int_t TGQt::processQtEvents(Int_t maxtime)
{
   // Force processing the Qt events only without entering the ROOT event loop
   QCoreApplication::processEvents(QEventLoop::AllEvents,maxtime);
   // QEventLoop::ExcludeUserInput QEventLoop::ExcludeSocketNotifiers
   return 0;
 }
