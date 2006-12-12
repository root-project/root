// @(#)root/qt:$Name:  $:$Id: TGQt.h,v 1.18 2006/10/04 16:08:47 antcheva Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGQt
#define ROOT_TGQt


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGQt                                                                 //
//                                                                      //
// Interface to low level Qt GUI. This class gives an access            //
// to the basic Qt graphics, pixmap, text and font handling routines.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQtRConfig.h"

#ifndef __CINT__
#include <vector>
#include <map>

#include <qobject.h>
#include <qmap.h>
#include <qcolor.h>
#include <qcursor.h>
#include <qpainter.h>
#include <qrect.h>
#include <qmap.h>
#include <qnamespace.h>

#if (QT_VERSION > 0x039999)
#  include <QPixmap>
#  include <QEvent>
#  include <q3memarray.h>
#  include <q3ptrqueue.h>
#  include <q3ptrlist.h>
#  include <q3ptrvector.h>
#else
#  include <qmemarray.h>
#  include <qptrqueue.h>
#  include <qptrlist.h>
#  include <qptrvector.h>
#endif /* QT_VERSION */

#include <qfontdatabase.h>

#include "TQtClientGuard.h"

#else
  class QObject;
  class QEvent;
#endif  /* CINT */

class  QPen;
class  QMarker;
class  QFont;
class  QPaintDevice;
class  QTextCodec;

#include "TVirtualX.h"
#include "TQtEmitter.h"

class TQtMarker;

class TQtPen;
class TQtSwitch;
class TQtBrush;
class TQtCommand;
class TFileHandler;
class TQtApplication;
class TQtClientFilter;
class TQtEventQueue;


//#define TRACE_TGQt() fprintf(stdout, "TGQt::%s() %d\n", __FUNCTION__, __LINE__)

class TGQt  : public TVirtualX  {

#ifndef __CINT__
   friend class TQtObject;
   friend class TQtWindowsObject;
   friend class TQtPixmapObject;
   friend class TPadOpenGLView;
   friend class TQtWidget;
   friend class TQtClientWidget;
   friend class TQtImage;
   friend class TQtClientGuard;
   friend class TQtClientFilter;
   
protected:
   enum DEFWINDOWID { kDefault=1 };
   QPaintDevice *fSelectedWindow;      // Pointer to the current "paintdevice: PixMap, Widget etc"
   QPaintDevice *fSelectedBuffer;      // Pointer to the current "paintdevice buffer"
   QPaintDevice *fPrevWindow;          // Pointer to the previous "Window"
   Int_t         fDisplayOpened;
   QPainter     *fQPainter;
   TQtEmitter    fEmitter;             // object to emit Qt signals on behalf of TVirtualX
   static TVirtualX     *fgTQt;        // The hiden poiner to fullish  ROOT TPluginManager

   void        *fhEvent;                   // The event object to synch threads

#if QT_VERSION < 0x40000
   QPtrVector<QCursor>   fCursors;
#else /* QT_VERSION */
   Q3PtrVector<QCursor>   fCursors;
#endif /* QT_VERSION */
//   Qt::CursorShape  fCursors[kNumCursors];  //List of cursors
   ECursor         fCursor;                 // Current cursor number;

   Style_t      fMarkerStyle;

   Int_t        fTextAlignH;         //Text Alignment Horizontal
   Int_t        fTextAlignV;         //Text Alignment Vertical
   Float_t      fCharacterUpX;       //Character Up vector along X
   Float_t      fCharacterUpY;       //Character Up vector along Y
   Int_t        fTextFontModified;   // Mark whether the text font has been modified
   Float_t      fTextMagnitude;      //Text Magnitude

//   Common HANDLES of the graphics attributes for all HIGZ windows

   TQtBrush  *fQBrush;
   QPen      *fQPen;
   TQtMarker *fQtMarker;
   QFont     *fQFont;
#if (QT_VERSION <0x40000)
   Qt::RasterOp fDrawMode;
#else
   QPainter::CompositionMode  fDrawMode;
#endif

   typedef QMap<QPaintDevice *,QRect> TQTCLIPMAP;
   TQTCLIPMAP fClipMap;

//
//  Colors staff
//

//   QMemArray<QColor> fPallete;
    QMap<Color_t,QColor> fPallete;
    TQtClientFilter *fQClientFilter;
    TQtEventQueue   *fQClientFilterBuffer;
    TQtClientGuard       fQClientGuard;  // guard TQtClientWibdget against of dead pointers
    TQtPixmapGuard       fQPixmapGuard;  // guard TQtClientWibdget against of dead pointers
    typedef std::map<ULong_t, QColor * > COLORMAP;
    COLORMAP fColorMap;  // to back the TG widgets
    TQtClientWidget       *fPointerGrabber;
    QTextCodec            *fCodec;            // The Current text decoder
    QString                fFontTextCode;     // The default code text code page (from the Gui.DefaultFont)
    const char            *fSymbolFontFamily; // the name of the font to substiute the non-standard "Symbol"


//
//   Text management
//

   //Qt::AlignmentFlags fTextAlign;

   void  SetTextFont(const char *fontname, Int_t italic, Int_t bold);
   Int_t CreatROOTThread();
   void  DeleteSelectedObj();

//  Qt methods
   static QRect GetQRect(QPaintDevice &dev);
   void Begin();
   void End();
   void UpdateFont();
   void UpdatePen();
   void UpdateBrush();
   void UpdateClipRectangle();
   
   QPaintDevice *GetDoubleBuffer(QPaintDevice *dev);

#endif
   static Int_t   RegisterWid(QPaintDevice *wid);   // register QWidget for the embedded TCanvas
   static Int_t   UnRegisterWid(QPaintDevice *wid); // unregister QWidget of the TCanvas
   static Bool_t  IsRegistered(QPaintDevice *wid);  // Check whether the object has been registered
private:
   TGQt& operator=(const TGQt&);
public:

    TGQt();
    TGQt(const TGQt &vx): TVirtualX(vx) { MayNotUse("TGQt(const TGQt &)"); }   // without dict does not compile? (rdm)
    TGQt(const Text_t *name, const Text_t *title);
    virtual ~TGQt();
// Include the base TVirtualX class interface
#include "TVirtualX.interface.h"
#ifndef __CINT__
// extracted methods
    virtual const QColor&   ColorIndex(Color_t indx) const;
    virtual QPaintDevice *GetSelectedWindow(){ return fSelectedWindow; }
    virtual void      SetFillStyleIndex( Int_t style, Int_t fasi);
    virtual void      SetMarkerType( Int_t type, Int_t n, TPoint *xy );
    virtual void      SetTitle(const char *title);
    virtual void      CopyPixmap(const QPixmap &p, Int_t px1, Int_t py1);
    virtual void      SetTextDecoder(const char * /*textDeocerName*/){;}  // for the future
    virtual const QTextCodec *GetTextDecoder();
#endif
// obsolete methods
        virtual void      PutByte(Byte_t b);
// ---------------------------------------------

   virtual Bool_t       IsHandleValid(Window_t id);


   // static methods:
   static TQtApplication *CreateQtApplicationImp();
   static Int_t          iwid(QPaintDevice *wid);
   static QPaintDevice  *iwid(Int_t wid);
   static QPaintDevice  *iwid(Window_t wid);
#ifndef __CINT__
#if ROOT_VERSION_CODE < ROOT_VERSION(5,13,0)
   static QPixmap       *MakeIcon(Int_t indx);
#endif
   static TVirtualX     *GetVirtualX();
   static QWidget       *winid(Window_t id);
   static QWidget       *wid(Window_t id);
   static Window_t       wid(TQtClientWidget *widget);
   static Window_t       rootwid(QPaintDevice *dev);
   static void           PrintEvent(Event_t &);
   
   void SetQClientFilter(TQtClientFilter *filter) {fQClientFilter = filter;}
   TQtClientFilter  *QClientFilter() const {return fQClientFilter;}
   QColor &QtColor(ULong_t pixel);
   void SendDestroyEvent(TQtClientWidget *) const;

   TQtEmitter *Emitter(){ return &fEmitter;}
#endif
// Future interface :
   virtual void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b, Float_t a);
   virtual void      SetAlpha(Int_t cindex, Float_t a);
   virtual void      GetRGBA(Int_t cindex, Float_t &r, Float_t &g, Float_t &b, Float_t &a);
   virtual Float_t   GetAlpha(Int_t cindex);
   virtual Int_t LoadQt(const char *shareLibFileName);
   static void PostQtEvent(QObject *receiver, QEvent *event);
   virtual Int_t processQtEvents(Int_t maxtime=300); //milliseconds
   // temporary this should be moved to the QTGL interface
   private:
      static int fgCoinFlag; // no coin viewer;
      static int fgCoinLoaded; // no coin viewer;
   public:
      static int CoinFlag();
      static void SetCoinFlag(int flag);
      static void SetCoinLoaded();
      static Int_t IsCoinLoaded();
#ifndef __CINT__
      static QString RootFileFormat(const char *selector);
      static QString RootFileFormat(const QString &selector);
      static QString QtFileFormat(const char *selector);
      static QString QtFileFormat(const QString &selector);
#endif

#ifndef Q_MOC_RUN
//MOC_SKIP_BEGIN
   ClassDef(TGQt,0)  //Interface to Qt GUI
//MOC_SKIP_END
#endif

};

R__EXTERN  TGQt *gQt;


#endif
