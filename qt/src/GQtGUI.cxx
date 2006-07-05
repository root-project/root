// @(#)root/qt:$Name:  $:$Id: GQtGUI.cxx,v 1.26 2006/06/28 09:30:58 antcheva Exp $
// Author: Valeri Fine   23/01/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <assert.h>
#include "TGQt.h"
#include "TQUserEvent.h"
#include "TQtClientFilter.h"
#include "TQtWidget.h"
#include "TQtClientWidget.h"
#include "TQtEventQueue.h"

#include "TROOT.h"
#include "TEnv.h"

#include <qapplication.h>
#include <qcursor.h>
#include <qpixmap.h>
#include <qbitmap.h>
#include <qregion.h>
#include <qclipboard.h>

#if QT_VERSION < 0x40000
#  include <qpaintdevicemetrics.h>
#  include <qobjectlist.h>
#  include <qcstring.h>
#else /* QT_VERSION */
#  include <q3paintdevicemetrics.h>
#  include <q3cstring.h>
#  include <QBoxLayout>
#  include <Q3Frame>
#  include <QKeyEvent>
#  include <QEvent>
#  include <Q3ValueList>
#  include <QVBoxLayout>
#  include <Q3PointArray>
#  include <QDesktopWidget>
#endif /* QT_VERSION */

#include <qfontmetrics.h>
#include <qpoint.h>
#include <qpainter.h>

#include <qlayout.h>
#include <qdatetime.h>
#include <qimage.h>
#include <qtextcodec.h>

#include "TMath.h"
#include "TQtBrush.h"

#include "TSystem.h"
#include "KeySymbols.h"

#ifdef R__QTWIN32
#  include "TWinNTSystem.h"
#else
# ifdef  R__QTX11
#  include <X11/Xlib.h>
# endif
#endif


#include "TError.h"

#include "TGFrame.h"
#include "TQtClientWidget.h"

static Window_t fgDefaultRootWindows = Window_t(-1);
// static const unsigned int kDefault=2;

static TQtClientWidget* cwid(Window_t id) { return ((TQtClientWidget*)TGQt::wid(id)); }
//static int GetRValue(ULong_t color){return (int)color;}
//static int GetGValue(ULong_t color){return (int)color;}
//static int GetBValue(ULong_t color){return (int)color;}
//______________________________________________________________________________
//
//     class QtGContext
//
class QtGContext : public QWidget {
   friend class TGQt;
   friend class TQtPainter;
protected:
   Mask_t       fMask;       // mask the active values
#if QT_VERSION < 0x40000
   Qt::RasterOp fROp;        // raster operation
#else /* QT_VERSION */
   QPainter::CompositionMode  fROp;   // composition mode
#endif /* QT_VERSION */
   QPen         fPen;        // line styles
   QBrush       fBrush;      // fill styles
   QPixmap     *fTilePixmap; // tile pixmap for tiling operations
   QPixmap     *fStipple;    // tile pixmap for tiling operations
   QPoint       fTileRect;   // The origin for the tile operation
   QPoint       fClipOrigin; // The origin for clipping
   QBitmap     *fClipMask;   // bitmap clipping; other calls for rects
   QRegion      fClipRegion; // Clip region
   QFont       *fFont;       // A copy of the font pointer
public:
   enum EContext { kROp  =1, kPen,     kBrush,     kTilePixmap
                 , kStipple, kTileRect,kClipOrigin,kClipMask,kClipRegion
                 , kFont
                 , kAllFields
                 };
   QtGContext() : QWidget(0,"rootGCContext") ,fMask(0),fTilePixmap(0),fStipple(0),fClipMask(0),fFont(0) {}
   QtGContext(const GCValues_t &gval) : QWidget(0,"rootGCContext") ,fMask(0),fTilePixmap(0),fStipple(0),fClipMask(0),fFont(0){Copy(gval);}
   QtGContext(const QtGContext & /*src*/)  : QWidget() {fprintf(stderr,"QtGContext(const QtGContext &src)\n");}
   void              Copy(const QtGContext &dst,Mask_t mask = 0xff);
   const QtGContext &Copy(const GCValues_t &gval);
   void              DumpMask() const;
   bool              HasValid(EContext bit) const { return TESTBIT (fMask , bit); }
   Mask_t            Mask() const { return fMask; }
   void              SetBackground(ULong_t background);
   void              SetMask(Mask_t mask){ fMask = mask;}
   void              SetForeground(ULong_t foreground);
   GContext_t        gc() const { return (GContext_t)this; }
   operator          GContext_t() const { return gc(); }
   const QtGContext &operator=(const GCValues_t &gval){ return Copy(gval);}
   QColor &QtColor(ULong_t pixel) {return gQt->QtColor(pixel);}

};

//______________________________________________________________________________
static QtGContext   &qtcontext(GContext_t context) { return *(QtGContext *)context;}
//______________________________________________________________________________
QColor &TGQt::QtColor(ULong_t pixel)
{
   // Look up the color table and return the reference to the QColor object.
   // Add a new entry if the object witht he "oixel" color value was not found
   COLORMAP::iterator colorIterator;
   //   QColor *color = fColorMap[pixel];
   if ((colorIterator = fColorMap.find(pixel)) != fColorMap.end()) {
      QColor *c =  (*colorIterator).second;
      return *c;
   } else {
      // this is a new color (red x green x blue)
      ColorStruct_t newColor;
      newColor.fRed  =  (pixel & 255);
      pixel = pixel >> 8;
      newColor.fGreen = (pixel & 255);
      pixel = pixel >> 8;
      newColor.fBlue   = (pixel & 255);
      Colormap_t cmap=0; // fake map
      gVirtualX->AllocColor(cmap, newColor);
      return QtColor(newColor.fPixel);
   }
}
//______________________________________________________________________________
#ifdef CopyQTContext
#define __saveCopyQTContext__  CopyQTContext
#undef  CopyQTContext
#endif

#define CopyQTContext(member)                                                 \
   if (dst.HasValid(_NAME2_(k,member))&&TESTBIT(mask, _NAME2_(k,member))) {   \
      SETBIT( fMask, _NAME2_(k,member));                                      \
      _NAME2_(f,member) = dst._NAME2_(f,member);                              \
   }
//______________________________________________________________________________
#if QT_VERSION < 0x40000
void DumpROp(Qt::RasterOp op) {
#else /* QT_VERSION */
void DumpROp(QPainter::CompositionMode op) {
#endif /* QT_VERSION */
   // Dump QT Raster Operation Code
   QString s;
   switch (op) {
#if QT_VERSION < 0x40000
        case Qt::ClearROP:  s = "Qt::ClearROP -> dst = 0 ";               break;
        case Qt::AndROP:    s = "Qt::AndROP dst = src AND dst ";          break;
        case Qt::AndNotROP: s = "Qt::AndNotROP dst = src AND (NOT dst) "; break;
        case Qt::CopyROP:   s = "Qt::CopyROP dst = src ";                 break;
        case Qt::NotAndROP: s = "Qt::NotAndROP dst = (NOT src) AND dst";  break;
        case Qt::NorROP:    s = "Qt::NorROP dst = NOT (src OR dst) ";     break;
        case Qt::NopROP:    s = "Qt::NopROP dst = dst ";                  break;
        case Qt::XorROP:    s = "Qt::XorROP dst = src XOR dst ";          break;
        case Qt::OrROP:     s = "Qt::OrROP dst = src OR dst ";            break;
        case Qt::NotXorROP: s = "Qt::NotXorROP dst = (NOT src) XOR dst  // Qt::NotOrROP);  // !!! This is not a GDK_EQUIV; !!!";  break;
        case Qt::NotROP:    s = "Qt::NotROP dst = NOT dst ";              break;
        case Qt::OrNotROP:  s = "Qt::OrNotROP dst = src OR (NOT dst) ";   break;
        case Qt::NotCopyROP:s = "Qt::NotCopyROP dst = NOT src ";          break;
        case Qt::NotOrROP:  s = "Qt::NotOrROP dst = (NOT src) OR dst ";   break;
        case Qt::NandROP:   s = "Qt::NandROP dst = NOT (src AND dst)";    break;
        case Qt::SetROP:    s = "Qt::SetROP dst = 1";                     break;
        default: s = "UNKNOWN";                                           break;
#else /* QT_VERSION */
        case QPainter::CompositionMode_Clear:     s = "Qt::ClearROP   dst = 0 ";                break; // ClearROP
  //    case QPainter::CompositionMode_AndROP:    s = "Qt::AndROP     dst = src AND dst ";      break;
  //    case QPainter::CompositionMode_AndNotROP: s = "Qt::AndNotROP  dst = src AND (NOT dst) ";break;
        case QPainter::CompositionMode_Source:    s = "Qt::CopyROP    dst = src ";              break; // CopyROP
  //    case QPainter::CompositionMode_NotAndROP: s = "Qt::NotAndROP  dst = (NOT src) AND dst"; break;
  //    case QPainter::CompositionMode_NorROP:    s = "Qt::NorROP     dst = NOT (src OR dst) "; break;
        case QPainter::CompositionMode_Destination:s= "Qt::NopROP     dst = dst ";              break; // NopROP
        case QPainter::CompositionMode_Xor:       s = "Qt::XorROP     dst = src XOR dst ";      break; // Qt::XorROP
  //    case QPainter::CompositionMode_OrROP:     s = "Qt::OrROP      dst = src OR dst ";       break;
  //    case QPainter::CompositionMode_NotXorROP: s = "Qt::NotXorROP  dst = (NOT src) XOR dst  // Qt::NotOrROP);  // !!! This is not a GDK_EQUIV; !!!";  break;
  //    case QPainter::CompositionMode_NotROP:    s = "Qt::NotROP     dst = NOT dst ";          break;
  //    case QPainter::CompositionMode_OrNotROP:  s = "Qt::OrNotROP   dst = src OR (NOT dst) "; break;
  //    case QPainter::CompositionMode_NotCopyROP:s = "Qt::NotCopyROP dst = NOT src ";          break;
  //    case QPainter::CompositionMode_NotOrROP:  s = "Qt::NotOrROP   dst = (NOT src) OR dst "; break;
  //    case QPainter::CompositionMode_NandROP:   s = "Qt::NandROP    dst = NOT (src AND dst)"; break;
  //    case QPainter::CompositionMode_SetROP:    s = "Qt::SetROP     dst = 1";                 break;
        default: s = "UNKNOWN";                                                                 break;
#endif /* QT_VERSION */
   }
#if QT_VERSION < 0x40000
   fprintf(stderr," Dump QT Raster Operation Code: %x \"%s\"\n",op, (const char *)s);
#else /* QT_VERSION */
   fprintf(stderr," Dump QT Composition mode Code: %x \"%s\"\n",op, (const char *)s);
#endif /* QT_VERSION */
}

//______________________________________________________________________________
#ifdef DumpGCMask
#define __saveDumpGCMask__
#undef DumpGCMask
#endif

#define DumpGCMask(member)                                                    \
 if (HasValid(_NAME2_(k,member))) {                                           \
   fprintf(stderr," mask bit : ");                                            \
   fprintf(stderr, _QUOTE_(_NAME2_(k,member)));                               \
   fprintf(stderr," is defined\n"); }
inline void QtGContext::DumpMask() const
{
   fprintf(stderr,"  Dump QtGContext mask %x \n", fMask);
   DumpROp(fROp);
   DumpGCMask(ROp);       DumpGCMask(Pen);     DumpGCMask(Brush);     DumpGCMask(TilePixmap);
   DumpGCMask(Stipple);   DumpGCMask(TileRect);DumpGCMask(ClipOrigin);DumpGCMask(ClipMask);
   DumpGCMask(ClipRegion);DumpGCMask(Font);
}
#ifdef __saveDumpGCMask__
#undef DumpGCMask
#define DumpGCMask __saveDumpGCMask__
#undef __saveDumpGCMask__
#endif

//______________________________________________________________________________
inline void QtGContext::Copy(const QtGContext &dst, Mask_t mask)
{
   // fprintf(stderr,"&QtGContext::Copy(const QtGContext &dst, Mask_t mask=%x)\n",mask);
   CopyQTContext(ROp);
   CopyQTContext(Pen);
   CopyQTContext(Brush);
   CopyQTContext(TilePixmap);
   CopyQTContext(Stipple);
   CopyQTContext(TileRect);
   CopyQTContext(ClipOrigin);
   CopyQTContext(ClipMask);
   CopyQTContext(ClipRegion);
   CopyQTContext(Font);

}
#ifdef __saveCopyQTContext__
#undef  CopyQTContext
#define CopyQTContext __saveCopyQTContext__
#undef __saveCopyQTContext__
#endif
//______________________________________________________________________________
const QtGContext  &QtGContext::Copy(const GCValues_t &gval)
{
   if (!&gval) return *this;
   // Fill this object from the "GCValues_t" structure
   // map GCValues_t to QtGContext
   Mask_t mask = gval.fMask;
   // fprintf(stderr,"&QtGContext::Copy(const GCValues_t &gval) this=%p mask=%x function=%x\n",this, mask,kGCFunction);
   if ((mask & kGCFunction)) {
      // fprintf(stderr," QtGContext::Copy this=%p, kGCFunction,%x, %d\n",this,  mask, gval.fFunction);
      SETBIT(fMask, kROp);
      switch (gval.fFunction)
      {
#if QT_VERSION < 0x40000
      case kGXclear:
         fROp = Qt::ClearROP;  // dst = 0
         break;
      case kGXand:
         fROp = Qt::AndROP;    // dst = src AND dst
         break;
      case kGXandReverse:
         fROp = Qt::AndNotROP; // dst = src AND (NOT dst) //
         break;
      case kGXcopy:
         fROp = Qt::CopyROP;   // dst = src
         break;
      case kGXandInverted:
         fROp = Qt::NotAndROP; // dst = (NOT src) AND dst
         break;
      case kGXnor:
         fROp = Qt::NorROP;   //  dst = NOT (src OR dst)
         break;
      case kGXnoop:
         fROp = Qt::NopROP;    // dst = dst
         break;
      case kGXxor:
         fROp = Qt::XorROP;     // dst = src XOR dst
         break;
      case kGXor:
         fROp = Qt::OrROP;     // dst = src OR dst
         break;
      case kGXequiv:
         fROp = Qt::NotXorROP; // dst = (NOT src) XOR dst  // Qt::NotOrROP);  // !!! This is not a GDK_EQUIV; !!!
         break;
      case kGXinvert:
         fROp = Qt::NotROP;    // dst = NOT dst
         break;
      case kGXorReverse:
         fROp = Qt::OrNotROP;  // dst = src OR (NOT dst)
         break;
      case kGXcopyInverted:
         fROp = Qt::NotCopyROP; // dst = NOT src
         break;
      case kGXorInverted:
         fROp = Qt::NotOrROP;   // dst = (NOT src) OR dst
         break;
      case kGXnand:
         fROp = Qt::NandROP;   // dst = NOT (src AND dst)
         break;
      case kGXset:
         fROp = Qt::SetROP;     // dst = 1
         break;
#endif /* not QT_VERSION */
      default:
#if QT_VERSION < 0x40000
	      fROp = Qt::CopyROP;
	break;
#else /* QT_VERSION */
        fROp = QPainter::CompositionMode_Source; //Qt::CopyROP;
        break;
#endif /* QT_VERSION */
      }
      // DumpROp(fROp);
//      fprintf(stderr," kGCFunction: fROp = %x\n",fROp );
   } else {
        // Fons said this must be like this. 4/07/2003 Valeri Fine
        SETBIT(fMask, kROp);
#if QT_VERSION < 0x40000
	fROp = Qt::CopyROP;
#else /* QT_VERSION */
	fROp = QPainter::CompositionMode_Source; // Qt::CopyROP;
#endif /* QT_VERSION */
   };

   if (mask & kGCSubwindowMode) {
#if 0
      SETBIT(fMask,kXXX
      if (gval.fSubwindowMode == kIncludeInferiors)
         fSubwindow_mode = GDK_INCLUDE_INFERIORS;
      else
         fSubwindow_mode = GDK_CLIP_BY_CHILDREN
#endif
   }
   if ((mask & kGCForeground)) {
      // xmask |= GDK_GC_FOREGROUND;
      // QColor paletteBackgroundColor - the background color of the widget
       SetForeground(gval.fForeground);
	 // fprintf(stderr," kGCForeground %s \root.exen", (const char*)QtColor(gval.fForeground).name());
   }
   if ((mask & kGCBackground)) {
       SetBackground(gval.fBackground);
        // fprintf(stderr," kGCBackgroun %s \n", (const char*)QtColor(gval.fBackground).name());
   }
   if ((mask & kGCLineWidth)) {
      SETBIT(fMask,kPen);
      fPen.setWidth(gval.fLineWidth);
   }
   if ((mask & kGCLineStyle)) {
      SETBIT(fMask,kPen);
      Qt::PenStyle style = Qt::NoPen;
      switch (gval.fLineStyle)
      {
        case kLineSolid:      style = Qt::SolidLine;   break;
        case kLineOnOffDash:  style = Qt::DashLine;    break;
        case kLineDoubleDash: style = Qt::DashDotLine; break;
      };
      fPen.setStyle(style);
   }
   if ((mask & kGCCapStyle)) {
      SETBIT(fMask,kPen);
      Qt::PenCapStyle style = Qt::FlatCap;
      switch (gval.fCapStyle)
      {
         case kCapNotLast:    style = Qt::FlatCap;   break;
         case kCapButt:       style = Qt::FlatCap;   break;
         case kCapRound:      style = Qt::RoundCap;  break;
         case kCapProjecting: style = Qt::SquareCap; break;  // no idea what this does mean
      };
      fPen.setCapStyle(style);
   }
   if ((mask & kGCJoinStyle)) {
      SETBIT(fMask,kPen);
      Qt::PenJoinStyle style = Qt::MiterJoin;
      switch (gval.fJoinStyle)
      {
         case kJoinMiter: style = Qt::MiterJoin; break;
         case kJoinBevel: style = Qt::BevelJoin; break;
         case kJoinRound: style = Qt::RoundJoin; break;
      };
      fPen.setJoinStyle(style);
   }
   if ((mask & kGCFillStyle)) {
      SETBIT(fMask,kBrush);
      Qt::BrushStyle style = Qt::NoBrush;
      switch (gval.fFillStyle)
      {
         case kFillSolid:          style = Qt::SolidPattern;  break;
         case kFillTiled:          style = Qt::Dense1Pattern; break;
         case kFillStippled:       style = Qt::Dense6Pattern; break;
         case kFillOpaqueStippled: style = Qt::Dense7Pattern; break;
         default:                  style = Qt::NoBrush;       break;
      };
      fBrush.setStyle(style);
   }
   if ((mask & kGCTile)) {
#ifdef QTDEBUG
      fprintf(stderr," kGCTile,%x, %p\n",mask,(QPixmap *) gval.fTile);
#endif
      if ( gval.fTile  != 0xFFFFFFFF ) {
         SETBIT(fMask,kTilePixmap);
         fTilePixmap = (QPixmap *) gval.fTile;
      }
   }
   if ((mask & kGCStipple)) {
      // fprintf(stderr," kGCStipple,%x, %p\n",mask,(QPixmap *) gval.fStipple);
      SETBIT(fMask,kStipple);
      fStipple = (QPixmap *) gval.fStipple;
      // setPaletteBackgroundPixmap (*fStipple);
      fBrush.setPixmap(*fStipple);
      SETBIT(fMask, kROp);
#if QT_VERSION < 0x40000
      fROp = Qt::XorROP;
#else /* QT_VERSION */
      fROp = QPainter::CompositionMode_Xor; // Qt::XorROP;
#endif /* QT_VERSION */
   }
   if ((mask & kGCTileStipXOrigin)) {
      SETBIT(fMask,kTileRect);
      fTileRect.setX(gval.fTsXOrigin);
   }
   if ((mask & kGCTileStipYOrigin)) {
      SETBIT(fMask,kTileRect);
      fTileRect.setY(gval.fTsYOrigin);
   }
   if ((mask & kGCFont)) {
      SETBIT(fMask,kFont);
      setFont(*(QFont *) gval.fFont);
      fFont = (QFont *) gval.fFont;
      // fprintf(stderr,"kGCFont font=0x%p\n", fFont);
   }
   if ((mask & kGCGraphicsExposures)) {
#if 0
      xmask |= GDK_GC_EXPOSURES;
      fGraphics_exposures = gval.fGraphicsExposures;
#endif
   }
   if ((mask & kGCClipXOrigin)) {
      // fprintf(stderr," kGCClipXOrigin,%x, %p\n",mask,(QPixmap *) gval.fClipXOrigin);
      SETBIT(fMask,kClipOrigin);
      fClipOrigin.setX(gval.fClipXOrigin);
   }
   if ((mask & kGCClipYOrigin)) {
      SETBIT(fMask,kClipOrigin);
      fClipOrigin.setY(gval.fClipYOrigin);
   }
   if ((mask & kGCClipMask)) {
      SETBIT(fMask,kClipMask);
      fClipMask = (QBitmap *) gval.fClipMask;
   }
   return *this;
}
//______________________________________________________________________________
void   QtGContext::SetBackground(ULong_t background)
{
    // reset the context background color
    SETBIT(fMask,kBrush);
    setPaletteBackgroundColor(QtColor(background));
    setEraseColor(QtColor(background));
}
//______________________________________________________________________________
void   QtGContext::SetForeground(ULong_t foreground)
{
   // xmask |= GDK_GC_FOREGROUND;
   // QColor paletteBackgroundColor - the background color of the widget
   SETBIT(fMask,kBrush);
   SETBIT(fMask,kPen);
   setPaletteForegroundColor (QtColor(foreground));
   fBrush.setColor(QtColor(foreground));
   fPen.setColor(QtColor(foreground)); 
}
//______________________________________________________________________________
//
//     class TQtPainter
//
class TQtPainter : public QPainter {
public:
#if QT_VERSION < 0x40000
   TQtPainter(const QPaintDevice * pd,const QtGContext &rootContext, Mask_t mask=0xff,bool unclipped = FALSE):
      QPainter(pd,unclipped){
#else /* QT_VERSION */
   TQtPainter(QPaintDevice * pd,const QtGContext &rootContext, Mask_t mask=0xff,bool unclipped = FALSE):
      QPainter(pd){
         setClipping(!unclipped);
#endif /* QT_VERSION */
         if (mask){}
         if (rootContext.HasValid(QtGContext::kROp)) {
#if QT_VERSION < 0x40000
            setRasterOp (rootContext.fROp);
#else /* QT_VERSION */
           if (device()->devType() !=  QInternal::Widget ) 
              setCompositionMode(rootContext.fROp);
#endif /* QT_VERSION */
         }
         if (rootContext.HasValid(QtGContext::kPen)) {
            setPen(rootContext.fPen);
         }
         if (rootContext.HasValid(QtGContext::kBrush)) {
            setBrush(rootContext.fBrush);
         }
         if (rootContext.HasValid(QtGContext::kTilePixmap)) {
            setBrush(rootContext.fBrush);
#ifdef QTDEBUG
            fprintf(stderr," NO special painter Qt implementation for TilePixmap option yet\n");
#endif
         }
         if (rootContext.HasValid(QtGContext::kStipple)) {
            setBrush(rootContext.fBrush);
         }
         if (rootContext.HasValid(QtGContext::kTileRect)) {
            setBrush(rootContext.fBrush);
#ifdef QTDEBUG
            fprintf(stderr," NO special painter  Qt implementation for TileRect option yet\n");
#endif
         }
         if (rootContext.HasValid(QtGContext::kClipOrigin)) {
            // fprintf(stderr," NO special painter  Qt implementation for ClipOrigin option yet\n");
            // setClipRect ( fClipOrigin , int w, int h, CoordinateMode m = CoordDevice )
         }
         if (rootContext.HasValid(QtGContext::kClipMask)) {
            // fprintf(stderr," NO special painter  Qt implementation for ClipMask option yet\n");
         }
         if (rootContext.HasValid(QtGContext::kClipRegion)) {
            setClipRegion (rootContext.fClipRegion);
         }
      }
};
//______________________________________________________________________________
//
//     class TQtGrabPointerFilter
//
class TQtGrabPointerFilter : public QObject {
protected:
   bool eventFilter( QObject *o, QEvent *e );
};
//______________________________________________________________________________
bool TQtGrabPointerFilter::eventFilter( QObject *, QEvent *e)
{
   // if ( e->type() == QEvent::KeyPress )
   {
      // special processing for key press
      QKeyEvent *k = (QKeyEvent *)e;
      qDebug( "Ate key press %d", k->key() );
      return TRUE; // eat event
   }
   // standard event processing
   return FALSE;
}
//______________________________________________________________________________
class TXlfd {
   // Naive parsinf and comparision of XLDF font descriptors
   public:
         QString fFontFoundry;
         QString fFontFamily;
         Int_t   fIsFontBold;
         Int_t   fIsFontItalic;
         Int_t   fPointSize;
         Int_t   fPixelSize;
   //______________________________________________________________________________
   TXlfd (const char* fontName)    { Init(QString(fontName)); }

   //______________________________________________________________________________
   TXlfd (const char* fontFamily, Int_t isFontBold, Int_t isFontItalic=-1)
   { Init(QString(fontFamily), isFontBold, isFontItalic);       }

   //______________________________________________________________________________
   TXlfd (const QString &fontFamily, Int_t isFontBold, Int_t isFontItalic)
   { Init(fontFamily, isFontBold, isFontItalic);                }

      //______________________________________________________________________________
   TXlfd (const QString &fontName) { Init(fontName);          }

   //______________________________________________________________________________
   inline void Init(const QString &fontName) {
      // Undefine all values;
      fIsFontBold  = fIsFontItalic = fPointSize = fPixelSize = -1;
      fFontFoundry = "*";
      fFontFamily  = fontName.section('-',2,2);

      QString fontWeight  = fontName.section('-',3,3);
      if (fontWeight != "*") 
         fIsFontBold = fontWeight.startsWith("bold") ? 1 : 0;

      QString fontSlant = fontName.section('-',4,4);
      if (fontSlant != "*" ) 
         fIsFontItalic = ((fontSlant[0] == 'i') || (fontSlant[0] == 'o')) ? 1 : 0;
      
      bool ok;
      QString fontPointSize = fontName.section('-',8,8);
      if (fontPointSize != "*") 
        fPointSize = fontPointSize.toInt(&ok);
      if (!ok) fPointSize = -1;

      QString fontPixelSize = fontName.section('-',7,7);
      if (fontPixelSize != "*") 
        fPixelSize = fontPixelSize .toInt(&ok);
      if (!ok) fPixelSize = -1;
   }
    //______________________________________________________________________________
    inline void Init(const QString &fontFamily, Int_t isFontBold
                   ,Int_t isFontItalic=-1, Int_t pointSize=-1, Int_t pixelSize=-1)
    {
       fFontFoundry = "*";
       fFontFamily  = fontFamily;
       fIsFontBold  = isFontBold;  fIsFontItalic = isFontItalic;
       fPointSize   = pointSize;   fPixelSize    = pixelSize;
       // ROOT doesn't want to see the point size.
       // To make it happy let calculate it
       fPixelSize = SetPointSize(pointSize);
       if (fPixelSize == -1) fPixelSize = pixelSize;
   }
   //______________________________________________________________________________
   inline Int_t SetPointSize(Int_t pointSize)
   {
     // Set the point size and return the pixel size of the font
       Int_t pixelSize = -1;
       fPointSize = pointSize;
       if (fPointSize > 0) {
          QFont sizeFont( fFontFamily, fPointSize, QFont::Normal, FALSE );
          pixelSize = sizeFont.pixelSize();
       }
       return pixelSize;
   }
   //______________________________________________________________________________
   inline bool operator==(const TXlfd &xlfd) const {
      return    ( (fFontFamily  == "*") || (xlfd.fFontFamily  == "*") || ( fFontFamily   == xlfd.fFontFamily)   )
             && ( (fFontFoundry == "*") || (xlfd.fFontFoundry == "*") || ( fFontFoundry  == xlfd.fFontFoundry)  )
             && ( (fIsFontBold  == -1 ) || (xlfd.fIsFontBold  == -1 ) || ( fIsFontBold   == xlfd.fIsFontBold)  )
             && ( (fIsFontItalic== -1 ) || (xlfd.fIsFontItalic== -1 ) || ( fIsFontItalic == xlfd.fIsFontItalic))
             && ( (fPointSize   == -1 ) || (xlfd.fPointSize   == -1 ) || ( fPointSize    == xlfd.fPointSize)   )
             && ( (fPixelSize   == -1 ) || (xlfd.fPixelSize   == -1 ) || ( fPixelSize    == xlfd.fPixelSize)   );
   }
   //______________________________________________________________________________
   inline bool operator!=(const TXlfd  &xlfd) const { return !operator==(xlfd); }
   //______________________________________________________________________________
   inline QString ToString() const 
   {
      QString xLDF = "-";
      xLDF += fFontFoundry + "-";  // text name of font creator
      xLDF += fFontFamily  + "-";  // name of the font. 
                                   // Related fonts generally have the same base names; 
                                   // i.e. helvetica, helvetica narrow , etc.
      QString weight_name = "*";   // usually one of [light|medium|demibold|bold] but other types may exist
      if (fIsFontBold > -1) 
         weight_name = fIsFontBold ? "bold" : "medium";
      xLDF += weight_name  + "-";
      
      QString slant_name = "*";   // one of [r|i|o]. i and o are used similarly, AFAIK
      if (fIsFontItalic  > -1) 
         slant_name = fIsFontItalic ? "i" : "r";
      xLDF += slant_name  + "-";
      
      // SETWIDTH_NAME    - [normal|condensed|narrow|double wide] 
      // ADD_STYLE_NAME   - not a classification field, used only for additional differentiation 
      xLDF += "*-*-";  // we do not crae (yet) about SETWIDTH and ADD_STYLE
      
      QString pixelsize = "*";   // 0 = scalable font; integer typicially height of bounding box  
      if (fPixelSize   > -1) 
         pixelsize  = QString::number(fPixelSize);
      xLDF += pixelsize  + "-";

      QString pointsize = "*";   // 0 = scalable font; integer typicially height of bounding box  
      if (fPointSize   > -1) 
         pointsize  = QString::number(fPointSize);
      xLDF += pointsize  + "-";

      // RESOLUTION_X - horizontal dots per inch 
      // RESOLUTION_Y - vertical dots per inch 
      // SPACING      - [p|m|c] p = proportional, m = monospaced, c = charcell. Charcell is 
      //                 a special case of monospaced where no glyphs have pixels outside 
      //                 the character cell; i.e. there is no kerning (no negative metrics). 
      // AVERAGE_WIDTH  - unweighted arithmetic mean of absolute value of width of each glyph 
      //                  in tenths of pixels
      
      xLDF += "*-*-*-*-";  // we do not crae (yet) about  RESOLUTION_X RESOLUTION_Y  SPACING  AVERAGE_WIDTH

      // CHARSET_REGISTRY and CHARSET_ENCODING 
      //                         the chararterset used to encode the font; ISO8859-1 for Latin 1 fonts 
      xLDF += "ISO8859-1";
      return xLDF;
   }
};

//______________________________________________________________________________
void  TGQt::SetOpacity(Int_t) { }
//______________________________________________________________________________
Window_t TGQt::GetWindowID(Int_t id) {
   // Create a "client" wrapper for the "canvas" widget to make Fons happy
   QPaintDevice *widDev = iwid(id);
   TQtWidget *canvasWidget = dynamic_cast<TQtWidget *>(iwid(id));
   if (widDev && !canvasWidget) {
      // The workaround for V.Onuchine ASImage - extremely error prone and dangerous
      // MUST be fixed later
     return rootwid(widDev);
   }
   assert(canvasWidget);
   TQtClientWidget  *client = 0;
   // Only one wrapper per "Canvas Qt Widget" is allowed
   if (! (client = (TQtClientWidget  *)canvasWidget->GetRootID() )  ) {
      //   QWidget *canvasWidget = (QWidget *)wid(id);
      QWidget *parent  = canvasWidget->parentWidget();
      client  = (TQtClientWidget  *)wid(CreateWindow(rootwid(parent)
         ,0,0,canvasWidget->width(),canvasWidget->height()
         ,0,0,0,0,0,0));
      // reparent the canvas
      canvasWidget->reparent(client,QPoint(0,0));
      QBoxLayout * l = new QVBoxLayout( client );
      l->addWidget( canvasWidget );
      canvasWidget->SetRootID(client);
      client->SetCanvasWidget(canvasWidget);
      canvasWidget->setMouseTracking(kFALSE);
   }
   return rootwid(client);
}
//______________________________________________________________________________
Window_t  TGQt::GetDefaultRootWindow() const
{
   return kDefault;
}
//______________________________________________________________________________
void TGQt::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   // Get window attributes and return filled in attributes structure.
   if (id == kNone) return;
   const QWidget &thisWindow = *wid(id);
 //  const QWidget &thisWindow = *(QWidget *)(TGQt::iwid(id));
   assert(&thisWindow);
   memset(&attr,0,sizeof(WindowAttributes_t));
   attr.fX        = thisWindow.x();
   attr.fY        = thisWindow.y();
   attr.fWidth    = thisWindow.width ();
   attr.fHeight   = thisWindow.height ();
   attr.fBorderWidth =  (thisWindow.frameGeometry().width() - thisWindow.width())/2;
   attr.fClass    = kInputOutput;
   attr.fRoot     = Window_t(thisWindow.topLevelWidget () );
#ifdef R__QTX11
   attr.fVisual   = thisWindow.x11Visual(); // = gdk_window_get_visual((GdkWindow *) id);
#else
   attr.fVisual   = 0; // = gdk_window_get_visual((GdkWindow *) id);
#endif
   // QPaintDeviceMetrics pdm(&thisWindow);
   attr.fDepth    = QPixmap::defaultDepth();
   attr.fColormap = (Colormap_t)&thisWindow.palette ();
   if (thisWindow.isShown ()) {
      attr.fMapState = thisWindow.isVisible() ? kIsViewable : kIsUnviewable;
   } else {
      attr.fMapState = kIsUnmapped;
   }
   attr.fBackingStore     = kNotUseful;
   attr.fSaveUnder        = kFALSE;
   attr.fMapInstalled     = kTRUE;
   attr.fOverrideRedirect = kFALSE;   // boolean value for override-redirect
   attr.fScreen   = QApplication::desktop()->screen() ;


   //fprintf(stderr, "GetWindowAttributes: %s: w=%d h=%d\n"
   //      ,(const char *)thisWindow.name()
   //      ,attr.fWidth ,attr.fHeight);
   attr.fYourEventMask = 0;
   // I have no idea what these bits mean

   attr.fBitGravity = 0;           // one of bit gravity values
   attr.fWinGravity = 0;           // one of the window gravity values
   attr.fAllEventMasks = 0;        // set of events all people have interest in
   attr.fDoNotPropagateMask = 0;   // set of events that should not propagate
}
//______________________________________________________________________________
Bool_t TGQt::ParseColor(Colormap_t /*cmap*/, const char *cname, ColorStruct_t &color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   // Set ColorStruct_t structure to default. Let system think we could
   // parse color.
   color.fPixel = 0;
   color.fRed   = 0;
   color.fGreen = 0;
   color.fBlue  = 0;
   color.fMask  = kDoRed | kDoGreen | kDoBlue;

   QColor thisColor(cname);
   if (thisColor.isValid() ) {
      color.fPixel = thisColor.pixel();
      color.fRed   = thisColor.red();
      color.fGreen = thisColor.green();
      color.fBlue  = thisColor.blue();
   }

   return thisColor.isValid();
}

//______________________________________________________________________________
Bool_t TGQt::AllocColor(Colormap_t /*cmap*/, ColorStruct_t &color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   // Set pixel value. Let system think we could allocate color.

   // Fons thinks they must be 65535  (see TColor::RGB2Pixel and TColor::RGB2Pixel)
   int cFactor=1;
   if (color.fRed>256 || color.fGreen>256 || color.fBlue>256 ){
      cFactor = 257;
   }
   QColor *thisColor = new QColor(color.fRed/cFactor,color.fGreen/cFactor,color.fBlue/cFactor);
   color.fPixel = thisColor->pixel();
//   color.fPixel = (ULong_t)new QColor(color.fRed/257,color.fGreen,color.fBlue);
   // Add the color to the cash
   fColorMap[color.fPixel] = thisColor;
   return kTRUE;
}
//______________________________________________________________________________
void TGQt::QueryColor(Colormap_t /*cmap*/, ColorStruct_t &color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel  the pointer to the QColor object should be set
   // on  return the fRed, fGreen and fBlue components will be set.
   // Thsi method should not be called (vf 16/01/2003) at all.
   // Set color components to default.

   // fprintf(stderr,"QueryColor(Colormap_t cmap, ColorStruct_t &color)\n");
   QColor &c  = QtColor(color.fPixel);
   // Fons thinks they must be 65535  (see TColor::RGB2Pixel and TColor::RGB2Pixel)
   color.fRed   = c.red()  <<8;
   color.fGreen = c.green()<<8;
   color.fBlue  = c.blue() <<8;
}
//______________________________________________________________________________
void TGQt::NextEvent(Event_t &event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   // Map the accumulated Qt events to the ROOT one tp process:
   if (qApp->hasPendingEvents ()) qApp->processEvents ();

   memset(&event,0,sizeof(Event_t));
   event.fType   = kOtherEvent;
#ifndef R__QTGUITHREAD
   if (!fQClientFilterBuffer)
      fQClientFilterBuffer = fQClientFilter->Queue();
   // qApp->processEvents ();
#endif
//--

   //   if (qApp->hasPendingEvents())  qApp->processOneEvent ();
   //   qApp->processEvents (5000);
   if (fQClientFilterBuffer) {
      Event_t *ev = fQClientFilterBuffer->dequeue ();
      if (ev) {
         // There is a danger of artifacts at this point.
         // For example the mouse pointer had left some screen area but
         // event keeps reporting it is still there
         event = *ev; delete ev;
         if (gDebug > 3) fprintf(stderr," TGQt::NextEvent event type=%d win=%p, WM handle=%lx\n", event.fType,(void *)event.fWindow,wid(event.fWindow)->handle());
      }
   }
}
//______________________________________________________________________________
void TGQt::GetPasteBuffer(Window_t /*id*/, Atom_t /*atom*/, TString &text, Int_t &nchar,
                           Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.
   // Get paste buffer. By default always empty.

   text = "";
   nchar = 0;
   QClipboard *cb = QApplication::clipboard();
   QClipboard::Mode mode =
      cb->supportsSelection() ? QClipboard::Selection :QClipboard::Clipboard;
   text = (const char *)cb->text(mode);
   if (text) nchar = strlen(text);
   if (del) cb->clear(mode);
}

// ---- Methods used for GUI -----
//______________________________________________________________________________
void         TGQt::MapWindow(Window_t id)
{
   // Map window on screen.
   if (id == kNone || wid(fgDefaultRootWindows) == wid(id) || id == kDefault ) return;

   // QWidget *nextWg = 0;
   QWidget *wg = wid(id);
   if ( wg ) {
      if  ( wg->isTopLevel () ){ 
         wg->showNormal();
      } else wg->show();
      // wg->update();
   }
}
//______________________________________________________________________________
void         TGQt::MapSubwindows(Window_t id)
{
   // Map sub (unhide) windows.
   // The XMapSubwindows function maps all subwindows for a specified window
   // in top-to-bottom stacking order.
   // In other words this method does reverese the Z-order of the child widgets

   if (id == kNone || id == kDefault) return;
//   return;
#if QT_VERSION < 0x40000
   const QObjectList *childList = wid(id)->children();
#else /* QT_VERSION */
   const QObjectList &childList = wid(id)->children();
#endif /* QT_VERSION */
   int nSubWindows = 0;
   int nChild = 0;
#if QT_VERSION < 0x40000
   if (childList) {
      nChild = childList->count();
      QObjectListIterator next(*childList);
      next.toLast();
#else /* QT_VERSION */
   if (!childList.isEmpty () ) {
      nChild = childList.count();
      QListIterator<QObject *> next(childList);
#endif /* QT_VERSION */
      QObject *widget = 0;
      int childCounter = 0; // to debug;
      // while ( (widget = *next) )
      Bool_t updateUnable;
      if ( (updateUnable = wid(id)->isUpdatesEnabled()) && nChild >0 )
            wid(id)->setUpdatesEnabled(FALSE);
#if QT_VERSION < 0x40000
      for (widget=next.toLast(); (widget = next.current()); --next)
#else /* QT_VERSION */
      next.toBack();
      while (next.hasPrevious())
#endif /* QT_VERSION */
      {
#if QT_VERSION >= 0x40000
         widget = next.previous();
#endif /* QT_VERSION */
         childCounter++;
         if (widget->isWidgetType ())
         {
            ((QWidget *)widget)->show();
            nSubWindows++;
         } else {
            // It is "QVBoxLayout" instantiated by TGQt::GetWindowID method. it is Ok.
            // fprintf(stderr," *****  TGQt::MapSubwindow the object %d is NOT a widget !!! %p %p %s \n"
            // , childCounter, id, widget, (const char *)widget->name(),(const char *)widget->className());
         }
      }
      if (updateUnable  && nChild >0 )
           wid(id)->setUpdatesEnabled(TRUE);
   }
}
//______________________________________________________________________________
void         TGQt::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.
   //
   //   Here we have to mimic the XMapRaised X11 function
   //   The XMapRaised function essentially is similar to XMapWindow in that it
   //   maps the window and all of its subwindows that have had map requests.
   //   However, it also raises the specified window to the top of the stack.

   if (id == kNone || id == kDefault) return;
#ifndef OLDQT25042003
   // fprintf(stderr, "   TGQt::MapRaised id = %p \n", id);
   QWidget *wg = wid(id);
   Bool_t updateUnable;
   if ( (updateUnable = wg->isUpdatesEnabled()) )
            wg->setUpdatesEnabled(FALSE);
   RaiseWindow(id);
   MapWindow(id);
   do {
////      wg->show();
      wg->setHidden (false);
      wg = wg->parentWidget();
   }  while ( wg && (!wg->isVisible()) );
   if (updateUnable)
       wid(id)->setUpdatesEnabled(TRUE);

   if (wid(id)->isTopLevel()) {
      // fprintf(stderr, "   TGQt::MapRaised top level id = %p \n", id);
      // wid(id)->update();
   }
#else
#if 0
     QWidget *wg = winid(id);
     wg->setHidden(false);
     wg->raise();
     wg->show();
#ifdef R__QTWIN32
   // raising the window under MS Windows needs some extra effort.
   HWND h = wg->winId();
   SetWindowPos(h,HWND_TOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
   SetWindowPos(h,HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE );
#endif
   wg->showNormal();
   MapSubwindows(id);

#else
   // Put the top window on the top top of desktop
   // fprintf(stderr, "\n  -1- TGQt::MapRaised id = %p vis=%d\n", id, wid(id)->isVisible());
   MapWindow(id);
   // fprintf(stderr, "  -2- TGQt::MapRaised id = %p vis=%d \n", id, wid(id)->isVisible());
   QWidget *wg = wid(id);
   if ( wg->isTopLevel() )  {
      // wg->setHidden(false);
#ifdef R__QTWIN32
      // raising the window under MS Windows needs some extra effort.
      HWND h = wg->winId();
      SetWindowPos(h,HWND_TOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
      SetWindowPos(h,HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE );
#endif
      wg->showNormal();
   } else {
      do {
         wg->raise();
         // fprintf(stderr, "  -%d- TGQt::MapRaised visible = %d x=%d, y=%d w=%d; counter = %d \n", ii++, wg->isVisible(),wg->x(),wg->y(),wg->width(),fileCounter);
      } while ( (! wg->isVisible() )  && (wg = wg->parentWidget()) );
   }
#endif
#endif
   // wg->showNormal();
}
//______________________________________________________________________________
void         TGQt::UnmapWindow(Window_t id)
{
   // Unmap window from screen.

   if (id == kNone) return;
   // fprintf(stderr, "\n 1.  TGQt::%s  %d %p visible = %d \n", __FUNCTION__, id, id, wid(id)->isVisible());
   if (!wid(id)->isHidden()) wid(id)->hide();
   // fprintf(stderr, "  2. TGQt::%s  %d %p visible = %d \n",  __FUNCTION__, id, id, wid(id)->isVisible());
}
//______________________________________________________________________________
void         TGQt::DestroyWindow(Window_t id)
{
   // Destroy the window

   if (id == kNone || id == kDefault ) return;
   fQClientGuard.Delete(wid(id));
   // wid(id)->close(true);
}
//______________________________________________________________________________
void  TGQt::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.
   if (id == kNone || id == kDefault ) return;
   wid(id)-> lower();
}
//______________________________________________________________________________
void  TGQt::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.
   //  fprintf(stderr," TGQt::MoveWindow %d %d \n",x,y);
   if (id == kNone || id == kDefault ) return;
   wid(id)->move(x,y);
}
//______________________________________________________________________________
void  TGQt::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize a window.
   // fprintf(stderr," TGQt::MoveResizeWindow %d %d %d %d\n",x,y,w,h);
   if (id == kNone || id == kDefault ) return;
//   if (0 <= x < 3000 && 0 <= y < 3000 && w < 3000 && h < 3000)
   {
   wid(id)->setGeometry(x,y,w,h);
   //wid(id)->setFixedSize(w,h);
}
}
//______________________________________________________________________________
void  TGQt::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.
   //  fprintf(stderr," TGQt::ResizeWindow w,h=%d %d\n",w,h);
   if (id == kNone || id == kDefault ) return;
   //   if ( w < 3000 && h < 3000)
   {
      wid(id)->resize(w,h);
      //   wid(id)->setFixedSize(w,h);
   }
}
//______________________________________________________________________________
void  TGQt::IconifyWindow(Window_t id)
{
   // Shows the widget minimized, as an icon
   if ( id == kNone || id == kDefault ) return;
   wid(id)->showMinimized();
}

//______________________________________________________________________________
void  TGQt::RaiseWindow(Window_t id)
{
  // Put window on top of window stack.
  //X11 says: The XRaiseWindow function raises the specified window to the top of the
  //  stack so that no sibling window obscures it.
  // OLD: Put the top window on the top top of desktop
   if ( id == kNone || id == kDefault ) return;
#ifndef OLDQT23042004
   QWidget *wg = wid(id);
   wg->raise();
#else
   MapWindow(id);
   QWidget *wg = wid(id);
   // fprintf(stderr, " 1.  TGQt::RaiseWindow id = %p IsTop=%d; parent = %p visible = %d \n", id,wg->isTopLevel(),iwid(wg->parentWidget()),wg->isVisible());
   if ( wg->isTopLevel() )  {
      QWidget *wg = ((TQtClientWidget *)wid(id))->topLevelWidget();
#ifdef R__QTWIN32
      // raising the window under MS Windows needs some extra effort.
      HWND h = wg->winId();
      SetWindowPos(h,HWND_TOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
      SetWindowPos(h,HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE );
#endif
      wg->showNormal();
   } else {
      do {
         wg->raise();
      } while ( (!wg->isVisible()) && (wg = wg->parentWidget()) );
      wid(id)->show();
      // fprintf(stderr, " 2.  TGQt::RaiseWindow id = %p  visible = %d \n", id,wid(id)->isVisible());
   }
#endif
}
//______________________________________________________________________________
void         TGQt::SetIconPixmap(Window_t id, Pixmap_t pix)
{
   // Set pixmap the WM can use when the window is iconized.
   if (id == kNone || id == kDefault || (pix==0) ) return;
   wid(id)->setIcon(*fQPixmapGuard.Pixmap(pix));
}
//______________________________________________________________________________
void         TGQt::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.
   if (id == kNone || id == kDefault ) return;
   wid(id)->setEraseColor(QtColor(color));
   // wid(id)->setPaletteBackgroundColor(QtColor(color));
}
//______________________________________________________________________________
void         TGQt::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.
   if (pxm  != kNone && id != kNone && id != kDefault )
      wid(id)->setPaletteBackgroundPixmap(*fQPixmapGuard.Pixmap(pxm));
}
//______________________________________________________________________________
Window_t TGQt::CreateWindow(Window_t parent, Int_t x, Int_t y,
                                    UInt_t w, UInt_t h, UInt_t border,
                                    Int_t , UInt_t ,
                                    void *, SetWindowAttributes_t *attr,
                                    UInt_t wtype)
{
   // Create QWidget to back TGWindow ROOT GUI object

   QWidget *pWidget = parent ? wid(parent):0;
//   if ( !pWidget) pWidget = QApplication::desktop();
   if (pWidget == QApplication::desktop())  pWidget = 0;
   TQtClientWidget *win = 0;
      // we don't want to introduce the high level class depedency at this point yet.
      // Alas ROOT design does require us to do the dirt thing
   if (        wtype & kTransientFrame) {
      win =  fQClientGuard.Create(pWidget,"TransientFrame");
#if QT_VERSION < 0x40000
      win->setFrameShape(QFrame::Box);      //  xattr.window_type = GDK_WINDOW_DIALOG;
   }  else if (wtype & kMainFrame)  {
      win =  fQClientGuard.Create(pWidget,"MainFrame"); //,Qt::WDestructiveClose);
      win->setFrameShape(QFrame::WinPanel); // xattr.window_type   = GDK_WINDOW_TOPLEVEL;
   }  else if (wtype & kTempFrame) {
      win =  fQClientGuard.Create(pWidget,"tooltip", Qt::WStyle_StaysOnTop | Qt::WStyle_Customize | Qt::WStyle_NoBorder | Qt::WStyle_Tool | Qt::WX11BypassWM );
      win->setFrameStyle(QFrame::PopupPanel | QFrame::Plain);
   } else {
      win =  fQClientGuard.Create(pWidget,"Other", Qt::WStyle_StaysOnTop | Qt::WStyle_Customize | Qt::WX11BypassWM );
      if (!pWidget) {
           win->setFrameStyle( QFrame::PopupPanel | QFrame::Plain );
//         printf(" 2 TGQt::CreateWindow %p parent = %p \n", win,pWidget);
      }
#else
      win->setFrameShape(Q3Frame::Box);      //  xattr.window_type = GDK_WINDOW_DIALOG;
   }  else if (wtype & kMainFrame)  {
      win =  fQClientGuard.Create(pWidget,"MainFrame"); //,Qt::WDestructiveClose);
      win->setFrameShape(Q3Frame::WinPanel); // xattr.window_type   = GDK_WINDOW_TOPLEVEL;
   }  else if (wtype & kTempFrame) {
      win =  fQClientGuard.Create(pWidget,"tooltip", Qt::WStyle_StaysOnTop | Qt::WStyle_Customize | Qt::WStyle_NoBorder | Qt::WStyle_Tool | Qt::WX11BypassWM );
      win->setFrameStyle(QFrame::StyledPanel | Q3Frame::Plain);
   } else {
      win =  fQClientGuard.Create(pWidget,"Other",   Qt::WStyle_StaysOnTop | Qt::WStyle_Customize | Qt::WX11BypassWM  );
      if (!pWidget) {
         win->setFrameStyle(QFrame::PopupPanel | QFrame::Plain);
       //   printf(" TGQt::CreateWindow %p parent = %p \n", win,pWidget);
      }
#endif
  }

   // printf(" TQt::CreateWindow %p parent = %p \n", win,pWidget);

   if (QClientFilter()) {
      win->installEventFilter(QClientFilter());
   }
   if (border > 0)
      win->setMargin(border);
   if (attr) {
      if ((attr->fMask & kWABackPixmap))
         if (attr->fBackgroundPixmap != kNone && attr->fBackgroundPixmap != kParentRelative )
         {
            //        win->setPaletteBackgroundPixmap(*(QPixmap *)attr->fBackgroundPixmap);
         }
         if ((attr->fMask & kWABackPixel)) {
            win->setPaletteBackgroundColor(QtColor(attr->fBackgroundPixel));
         }
         if ( attr->fMask & kWAEventMask) {
            // Long_t     fEventMask;            // set of events that should be saved
            win->SetAttributeEventMask(attr->fEventMask);
         }
   }
   MoveResizeWindow(rootwid(win),x,y,w,h);
   return rootwid(win);
}
//______________________________________________________________________________
Int_t        TGQt::OpenDisplay(const char *dpyName)
{
  // The dummy method to fit the X11-like interface
  if (dpyName){}
#ifdef R__QTX11
  return ConnectionNumber( GetDisplay() );
#else
  return 1;
#endif
}
//______________________________________________________________________________
void    TGQt::CloseDisplay()
{
   // The close all remainign QWidgets
   qApp->closeAllWindows();
}
//______________________________________________________________________________
Display_t  TGQt::GetDisplay() const
{
   // Returns handle to display (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   // Calling implies the direct X11 manipulation
   // Using this method makes the rest of the ROOT X11 depended

#ifdef R__QTX11
   return (Display_t)QPaintDevice::x11AppDisplay ();
#else
   // The dummy method to fit the X11-like interface
   return 0;
#endif
}
//______________________________________________________________________________
Visual_t   TGQt::GetVisual() const
{
   // Returns handle to visual (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   // Calling implies the direct X11 manipulation
   // Using this method makes the rest of the ROOT X11 depended

#ifdef R__QTX11
   return (Visual_t)QPaintDevice::x11AppVisual ();
#else
   // The dummy method to fit the X11-like interface
   return 0;
#endif
}
//______________________________________________________________________________
Int_t      TGQt::GetScreen() const
{
   // Returns screen number (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   // Calling implies the direct X11 manipulation
   // Using this method makes the rest of the ROOT X11 depended

#ifdef R__QTX11
   return QPaintDevice::x11AppScreen ();
#else
   // The dummy method to fit the X11-like interface
   return 0;
#endif
}
//______________________________________________________________________________
Int_t      TGQt::GetDepth() const
{
   // Returns depth of screen (number of bit planes).
#ifdef R__QTX11
   return QPaintDevice::x11AppDepth ();
#else
   return QPixmap::defaultDepth();
#endif
}
//______________________________________________________________________________
Colormap_t TGQt::GetColormap() const { return 0; }

//______________________________________________________________________________
Atom_t     TGQt::InternAtom(const char *atom_name, Bool_t /*only_if_exist*/)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   const char *rootAtoms[] = {  "WM_DELETE_WINDOW"
                              , "_MOTIF_WM_HINTS"
                              , "_ROOT_MESSAGE"
                              , "_ROOT_CLIPBOARD"
                              , "CLIPBOARD"
                              , ""  };
  int nSize = sizeof(rootAtoms)/sizeof(char *);
  nSize --;
  int i;
  for (i=0; (i<nSize)  && ( strcmp(atom_name,rootAtoms[i])) ;i++){;}
  // printf("  TGQt::InternAtom %d <%s>:<%s> \n",i, rootAtoms[i],atom_name );
  return i;
}
//______________________________________________________________________________
Window_t     TGQt::GetParent(Window_t id) const
{
   // Return the parent of the window.
   if ( id == kNone || id == kDefault ) return id;
   QWidget *dadWidget = wid(id)->parentWidget();
   assert(dynamic_cast<TQtClientWidget*>(dadWidget));
   return rootwid(dadWidget);
}
//______________________________________________________________________________
FontStruct_t TGQt::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise a opaque pointer to the FontStruct_t.
   // Parse X11-like font definition to QFont parameters:
   // -adobe-helvetica-medium-r-*-*-12-*-*-*-*-*-iso8859-1
   // -adobe-helvetica-medium-r-*-*-*-12-*-*-*-*-iso8859-1
#ifdef R__UNIX
   QFont *newFont = new QFont();
   newFont->setRawName(QString(font_name));
#else
   QString fontName(font_name);
   QString fontFamily = fontName.section('-',1,2);
   int weight = QFont::Normal;

   QString fontWeight = fontName.section('-',3,3);
   if (fontWeight.startsWith("bold")) weight = QFont::Bold;

   bool italic = (fontName.section('-',4,4)[0] == 'i');


   bool ok;
   int fontSize=12;
   int fontPointSize   = fontName.section('-',8,8).toInt(&ok);
   if (ok) fontSize = fontPointSize;
   QFont *newFont = new QFont(fontFamily,fontSize,weight,italic);
   if (!ok) {
      int fontPixelSize   = fontName.section('-',7,7).toInt(&ok);
      if (ok)
         newFont->setPixelSize(int(TMath::Max(fontPixelSize,1)));
   }
#endif
   //fprintf(stderr, " 0x%p = LoadQueryFont(const char *%s) = family=%s, w=%s, size=%d (pt), pixel size=%d\n",
   //        newFont, font_name,(const char *)fontFamily,(const char *)fontWeight,fontSize,newFont->pixelSize());
   return FontStruct_t(newFont);
}
//______________________________________________________________________________
FontH_t      TGQt::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.
   // This is adummy operation.
   //  There is no reason to use any handle
   // fprintf(stderr," TGQt::GetFontHandle(FontStruct_t fs) %s\n",(const char *) ((QFont *)fs)->toString());
   return (FontH_t)fs;
}
//______________________________________________________________________________
void         TGQt::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure.
   delete (QFont *)fs;
}
//______________________________________________________________________________
GContext_t   TGQt::CreateGC(Drawable_t /*id*/, GCValues_t *gval)
{
  // Create a graphics context using the values set in gval (but only for
  // those entries that are in the mask).
   QtGContext *context = 0;
   if (gval) 
      context =  new QtGContext(*gval);
   else 
      context =  new QtGContext();      
//   MapGCValues(*gval, context)
  return GContext_t(context);
}
//______________________________________________________________________________
void         TGQt::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.
   qtcontext(gc) = *gval;
}
//______________________________________________________________________________
void         TGQt::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
  // Copies graphics context from org to dest. Only the values specified
  // in mask are copied. Both org and dest must exist.
   qtcontext(dest).Copy(qtcontext(org),mask);
}
//______________________________________________________________________________
void         TGQt::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.
   delete &qtcontext(gc);
}
//______________________________________________________________________________
Cursor_t     TGQt::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).
  return Cursor_t(fCursors[cursor]);
}
//______________________________________________________________________________
void         TGQt::SetCursor(Window_t id, Cursor_t curid)
{
   // Set the specified cursor.
  if (id && id != Window_t(-1)) cwid(id)->SetCursor(curid);

}
//______________________________________________________________________________
Pixmap_t     TGQt::CreatePixmap(Drawable_t /*id*/, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.
   QPixmap *p = fQPixmapGuard.Create(w, h);
   return Pixmap_t(rootwid(p));
}
//______________________________________________________________________________
Pixmap_t     TGQt::CreatePixmap(Drawable_t /*id*/, const char *bitmap, UInt_t width,
                               UInt_t height, ULong_t forecolor, ULong_t backcolor,
                               Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.
   QPixmap *p = 0;
   if (depth >1) {
      QBitmap bp(width, height,(const uchar*)bitmap,kTRUE);
      QBrush  fillBrush(QtColor(backcolor), bp);
      p =  fQPixmapGuard.Create(width,height,depth);
      QPainter pixFill(p);
      pixFill.setBackgroundColor(QtColor(backcolor));
      pixFill.setPen(QtColor(forecolor));
      pixFill.fillRect(0,0,width, height,fillBrush);
   } else {
      p = fQPixmapGuard.Create(width, height,(const uchar*)bitmap);
   }
   return Pixmap_t(rootwid(p));
}
//______________________________________________________________________________
Pixmap_t     TGQt::CreateBitmap(Drawable_t id, const char *bitmap,
                               UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.
   return CreatePixmap(id,bitmap,width,height, 1,0,1);
}
//______________________________________________________________________________
void         TGQt::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.
   if (pmap  != kNone )
      fQPixmapGuard.Delete((QPixmap *)iwid(pmap));
   // delete (QPixmap *)pmap;
}
//______________________________________________________________________________
static inline void FillPixmapAttribute(QPixmap &pixmap, Pixmap_t &pict_mask
                                       , PictureAttributes_t & attr,TQtPixmapGuard &guard)
{
   // static method to avoid a access to the died objects
   attr.fWidth  = pixmap.width();
   attr.fHeight = pixmap.height();
   // Let's see whether the file brought us any mask.
#if QT_VERSION < 0x40000
   if  ( pixmap.mask() && !pixmap.mask()->isNull() ) {
#else /* QT_VERSION */
   if  ( !pixmap.mask().isNull() ) {
#endif /* QT_VERSION */
      QBitmap *pixmask = (QBitmap *)guard.Pixmap(pict_mask,kTRUE);
      if (pixmask) { // fill it with the new value
#if QT_VERSION < 0x40000
         *pixmask = *pixmap.mask();
#else /* QT_VERSION */
         *pixmask = pixmap.mask();
#endif /* QT_VERSION */
      } else {
#if QT_VERSION < 0x40000
         pixmask   = guard.Create(*pixmap.mask());
#else /* QT_VERSION */
         pixmask   = guard.Create(pixmap.mask());
#endif /* QT_VERSION */
         pict_mask = Pixmap_t(TGQt::rootwid(pixmask));
      }
   } else {
      pict_mask = kNone;
   }
}
//______________________________________________________________________________
Bool_t       TGQt::CreatePictureFromFile( Drawable_t /*id*/, const char *filename,
                                        Pixmap_t & pict,
                                        Pixmap_t &pict_mask,
                                        PictureAttributes_t & attr)
{
   // Create a picture pixmap from data on file. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.
   QPixmap *pixmap = 0;
   if (pict  != kNone )
      pixmap = fQPixmapGuard.Pixmap(pict);
   if (!pixmap) {
      // Create the new pixmap
      pixmap = fQPixmapGuard.Create(QString(filename));
      // pixmap = new QPixmap (QString(filename));
      pict = Pixmap_t(rootwid(pixmap));
   } else {
      // reload the old one
      pixmap->load(QString(filename));
   }
   if (! pixmap->isNull() ) {
      FillPixmapAttribute(*pixmap,pict_mask,attr,fQPixmapGuard);
   } else {
      fQPixmapGuard.Delete(pixmap);
      pict= kNone;
      pixmap = 0;
   }
   return pixmap;
}

//______________________________________________________________________________
Bool_t       TGQt::CreatePictureFromData(Drawable_t /*id*/, char **data,
                                        Pixmap_t & pict,
                                        Pixmap_t &pict_mask,
                                        PictureAttributes_t & attr)
{
   // Create a pixture pixmap from data. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.
   QPixmap *pixmap = fQPixmapGuard.Pixmap(pict);
   if (!pixmap) {
      pixmap = fQPixmapGuard.Create((const char **)data);
      pict = Pixmap_t(rootwid(pixmap));
   }  else {
      *pixmap = QPixmap ( (const char **)data);
   }

   if (! pixmap->isNull() ) {
      FillPixmapAttribute(*pixmap,pict_mask,attr,fQPixmapGuard);
   } else {
      fQPixmapGuard.Delete(pixmap);
      pict= kNone;
      pixmap = 0;
   }
   return pixmap;
}
//______________________________________________________________________________
Bool_t       TGQt::ReadPictureDataFromFile(const char *fileName, char ***data)
{
   // Read picture data from file and store in ret_data. Returns kTRUE in
   // case of success, kFALSE otherwise.
   QPixmap *pictureBuffer = fQPixmapGuard.Create(QString(fileName));
   if (pictureBuffer->isNull()){
      fQPixmapGuard.Delete(pictureBuffer);
   }  else {
      // &data = (char **)pictureBuffer;
   }
   if (!data)
      return gSystem->Load(fileName);
   else {
      fprintf(stderr, "I got no idea why do we need this trick yet!\n");
   }
   return kFALSE;
}
//______________________________________________________________________________
 void         TGQt::DeletePictureData(void *data)
 {
    // Delete the QPixmap
    fQPixmapGuard.Delete((QPixmap *)data);
 }
//______________________________________________________________________________
void         TGQt::SetDashes(GContext_t /*gc*/, Int_t /*offset*/, const char * /*dash_list*/,
                        Int_t /*n*/)
{
   // Specify a dash pattertn. Offset defines the phase of the pattern.
   // Each element in the dash_list array specifies the length (in pixels)
   // of a segment of the pattern. N defines the length of the list.

   //  QT has no built-in "user defined dashes"
}
//______________________________________________________________________________
Int_t  TGQt::EventsPending() {
#ifndef R__QTGUITHREAD
    Int_t retCode = fQClientFilterBuffer ? fQClientFilterBuffer->count(): 0;
    return  retCode ? retCode : qApp->hasPendingEvents ();
#endif

   if (fQClientFilterBuffer && fQClientFilterBuffer->isEmpty())
   {
      // We do not need the empty buffer, Let's delete it
#ifndef R__QTGUITHREAD
      if (qApp->hasPendingEvents ()) qApp->processEvents ();
#else
      delete fQClientFilterBuffer;
      fQClientFilterBuffer = 0;
#endif
//      return 0;
   }
   if (!fQClientFilterBuffer)
      fQClientFilterBuffer = fQClientFilter->Queue();
   return fQClientFilterBuffer ? fQClientFilterBuffer->count(): 0;
}
//______________________________________________________________________________
void TGQt::Bell(Int_t percent)
{
   // Sound bell
#ifdef R__QTWIN32
   DWORD dwFreq     = 1000L;         // sound frequency, in hertz
   DWORD dwDuration = 100L+percent;  // sound frequency, in hertz
   Beep(dwFreq,dwDuration);
#else
   if (percent) {}
   QApplication::beep ();
#endif
}
//______________________________________________________________________________
void         TGQt::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                           Int_t src_x, Int_t src_y, UInt_t width,
                           UInt_t height, Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. QPaintDevice  ) to another drawable (pixmap, widget).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.
   assert(qtcontext(gc).HasValid(QtGContext::kROp));
   // fprintf(stderr," TQt::CopyArea this=%p, fROp=%x\n", this, qtcontext(gc).fROp);
   if ( dest && src) {
      //QtGContext qgc = qtcontext(gc);
      QPixmap *pix = dynamic_cast<QPixmap*>(iwid(src));
      QBitmap *mask = qtcontext(gc).fClipMask;
      if (pix && mask && (qtcontext(gc).fMask & QtGContext::kClipMask)) {
         if ((pix->width() != mask->width()) || (pix->height() != mask->height())) {
            pix->resize(mask->width(), mask->height());
        }
         pix->setMask(*mask);
         bitBlt(iwid(dest), dest_x,dest_y,pix, src_x,src_y,width,height, qtcontext(gc).fROp);
      } else {
         bitBlt(iwid(dest), dest_x,dest_y,iwid(src), src_x,src_y,width,height, qtcontext(gc).fROp);
      }
   }
}
//______________________________________________________________________________
void         TGQt::ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr)
{
   // Change window attributes.

   if (!attr || (id == kNone) || (id == kDefault) )  return;

   TQtClientWidget *p = dynamic_cast<TQtClientWidget *>(wid(id));
   assert(p);
   TQtClientWidget &f = *p;
   //   fMask;   // bit mask specifying which fields are valid
   if ( attr->fMask & kWABackPixmap) {
      switch (attr->fBackgroundPixmap) {
          case kNone:
          case kParentRelative:
             break;
          default: f.setPaletteBackgroundPixmap (*(QPixmap *)attr->fBackgroundPixmap);
             break;
      };
   }
   if ( attr->fMask & kWABackPixel) {
      // background pixel
      f.setPaletteBackgroundColor(QtColor(attr->fBackgroundPixel));
      f.setEraseColor(QtColor(attr->fBackgroundPixel));
   }
   if ( attr->fMask & kWABorderPixmap) {
      // fBorderPixmap;         // border of the window
   }
   if ( attr->fMask & kWABorderPixel) {
      // ULong_t    fBorderPixel;          // border pixel value
       // f.setFrameShape( QFrame::PopupPanel );
#if QT_VERSION < 0x40000
       f.setFrameStyle( QFrame::Box | QFrame::Plain );
#else /* QT_VERSION */
       f.setFrameStyle( Q3Frame::Box | Q3Frame::Plain );
#endif /* QT_VERSION */
       // printf("TGQt::ChangeWindowAttributes  kWABorderPixel %p name = %s; shape = %d; margin = %d width=%d \n",&f,(const char*)f.name(),f.frameShape(),f.margin(),f.lineWidth() );
   }
   if ( attr->fMask & kWABorderWidth) {
      // border width in pixels)
       f.setLineWidth(attr->fBorderWidth);
       // printf("TGQt::ChangeWindowAttributes  kWABorderWidth %p %d margin=%d\n",&f, attr->fBorderWidth,f.margin());
   }
   if ( attr->fMask & kWABitGravity) {
      //  Int_t      fBitGravity;           // one of bit gravity values
   }
   if ( attr->fMask & kWAWinGravity) {
      // Int_t      fWinGravity;           // one of the window gravity values
   }
   if ( attr->fMask & kWABackingStore) {
      //  Int_t      fBackingStore;         // kNotUseful, kWhenMapped, kAlways
   }
   if ( attr->fMask & kWABackingPlanes) {
      // ULong_t    fBackingPlanes;        // planes to be preseved if possible
   }
   if ( attr->fMask & kWABackingPixel) {
      // ULong_t    fBackingPixel;         // value to use in restoring planes
   }
   if ( attr->fMask & kWAOverrideRedirect) {
      // Bool_t     fOverrideRedirect;     // boolean value for override-redirect
   }
   if ( attr->fMask & kWASaveUnder) {
      // Bool_t     fSaveUnder;            // should bits under be saved (popups)?
   }
   if ( attr->fMask & kWAEventMask) {
      // Long_t     fEventMask;            // set of events that should be saved
      f.SetAttributeEventMask(attr->fEventMask);
   }
   if ( attr->fMask & kWADontPropagate) {
      // Long_t     fDoNotPropagateMask;   // set of events that should not propagate
   }
   if ( attr->fMask & kWAColormap) {
      // Colormap_t fColormap;             // color map to be associated with window
   }
   if ( attr->fMask & kWACursor) {
      // cursor to be displayed (or kNone)
      if (fCursor != kNone) f.setCursor(*fCursors[fCursor]);
      else f.setCursor(QCursor(Qt::BlankCursor));
   }
}
//______________________________________________________________________________
void         TGQt::ChangeProperty(Window_t, Atom_t, Atom_t, UChar_t *, Int_t) { }
//______________________________________________________________________________
void TGQt::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.
   TQtPainter p(iwid(id),qtcontext(gc));
   p.drawLine ( x1, y1, x2, y2 );
}
//______________________________________________________________________________
void         TGQt::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.
   if (id == kNone || id == kDefault ) return;
   wid(id)->erase (x, y, w, h );
}
//______________________________________________________________________________
Bool_t       TGQt::CheckEvent(Window_t, EGEventType, Event_t &) 
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.
  
    return kFALSE; 
}
//______________________________________________________________________________
void         TGQt::SendEvent(Window_t id, Event_t *ev)
{
   // Send event ev to window id.

   if ( (ev->fType  == kClientMessage || ev->fType  == kDestroyNotify) && ev &&  id != kNone )
   {
      TQUserEvent qEvent(*ev);
      static TQtClientWidget *gMessageDispatcherWidget = 0;
      if (!gMessageDispatcherWidget) {
         gMessageDispatcherWidget = fQClientGuard.Create(0,"messager");
         if (QClientFilter()) {
            gMessageDispatcherWidget->installEventFilter(QClientFilter());
         }
      }
      QObject * receiver = 0;
      if ( id != kDefault) {
         receiver = wid(id);
      } else {
         receiver = gMessageDispatcherWidget;
         // fprintf(stderr, "  TGQt::SendEvent(Window_t id=%d, Event_t *ev=%d) to %p\n", id, ev->fType,  receiver);
      }

      // fprintf(stderr, "  TGQt::SendEvent(Window_t id, Event_t *ev) %p type=%d\n", wid(id), ev->fType);
      QApplication::postEvent(receiver,new TQUserEvent(*ev));
   } else {
      fprintf(stderr,"TQt::SendEvent:: unknown event %d for widget: %p\n",ev->fType,wid(id));
   }
}
//______________________________________________________________________________
 void         TGQt::WMDeleteNotify(Window_t id)
 {
    // WMDeleteNotify causes the filter to treat QEventClose event
    if (id == kNone || id == kDefault ) return;
    ((TQtClientWidget *)wid(id))->SetDeleteNotify();
    // fprintf(stderr,"TGQt::WMDeleteNotify %p\n",id);
 }
//______________________________________________________________________________
 void         TGQt::SetKeyAutoRepeat(Bool_t) { }
//______________________________________________________________________________
 void         TGQt::GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab)
 {
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.
   // A modifiers argument of AnyModifier is equivalent to issuing the
   // request for all possible modifier combinations (including the combination of no modifiers).

    if (id == kNone) return;

    if (grab ) {
      ((TQtClientWidget*)wid(id))->SetKeyMask(keycode,modifier);
    } else {
       ((TQtClientWidget*)wid(id))->UnSetKeyMask(keycode,modifier);
    }
 }
//______________________________________________________________________________
 void         TGQt::GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                       UInt_t evmask, Window_t confine, Cursor_t cursor,
                       Bool_t grab)
 {
   // Establish passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.

    //X11: ButtonPress event is reported if all of the following conditions are true:
    //   ·    The pointer is not grabbed, and the specified button is logically
    //        pressed when the specified modifier keys are logically down,
    //        and no other buttons or modifier keys are logically down.
    //   ·    The grab_window contains the pointer.
    //   ·    The confine_to window (if any) is viewable.
    //   ·    A passive grab on the same button/key combination does not exist
    //        on any ancestor of grab_window.

//    fprintf(stderr,"TGQt::GrabButton \"0x%x\" id=%x QWidget = %p\n"
//       ,evmask,id,((TQtClientWidget*)wid(id)));
    if (id == kNone) return;
    assert(confine==kNone);
    if (grab ) {
//       if (cursor == kNone) {
          ((TQtClientWidget*)wid(id))->SetButtonMask(modifier,button); 
          ((TQtClientWidget*)wid(id))->SetButtonEventMask(evmask,cursor);
    } else {
          ((TQtClientWidget*)wid(id))->UnSetButtonMask();
    }
}

 //______________________________________________________________________________
 void         TGQt::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                        Cursor_t cursor, Bool_t grab, Bool_t owner_events)
 {
    // Establish an active pointer grab. While an active pointer grab is in
    // effect, further pointer events are only reported to the grabbing
    // client window.

    // XGrabPointer(3X11):
       // The XGrabPointer function actively grabs control of the
       // pointer and returns GrabSuccess if the grab was success­
       // ful.  Further pointer events are reported only to the
       // grabbing client.  XGrabPointer overrides any active
       // pointer grab by this client.  If owner_events is False,
       // all generated pointer events are reported with respect to
       // grab_window and are reported only if selected by
       // event_mask.  If owner_events is True and if a generated
       // pointer event would normally be reported to this client,
       // it is reported as usual.  Otherwise, the event is reported
       // with respect to the grab_window and is reported only if
       // selected by event_mask.
       //-------------------------------------------------------------------
       // For either value of owner_events, unreported events are discarded.
       //-------------------------------------------------------------------

    assert(confine==kNone);
    TQtClientWidget *gw = (id == kNone) ?  0: cwid(id);
    // Do we still grabbing anything ?
    if (grab) {
       if ( !gw )  return;
       fPointerGrabber = gw;
       // fprintf(stderr,"TGQt::GrabPointer grabbing with the cursor: owner=%d wid = %x %p\n", owner_events, id, gw);
    } else {
       if (!gw) gw = fPointerGrabber;
       // fprintf(stderr,"TGQt::GrabPointer ungrabbing with the cursor: owner=%d wid =  %x grabber =%p \n"
       //      , owner_events, id, gw );
       fPointerGrabber = 0;
    }
    TQtClientFilter *f = QClientFilter();
    if (f) 
        f->GrabPointer(gw, evmask,0,(QCursor *)cursor, grab, owner_events);
 }
//______________________________________________________________________________
 void         TGQt::SetWindowName(Window_t id, char *name)
 {
    // Set window name.
    if (id == kNone || id == kDefault ) return;
    winid(id)->setCaption(name);
 }
 //______________________________________________________________________________
 void         TGQt::SetIconName(Window_t id, char *name)
 {
    // Set window icon name.
    if (id == kNone || id == kDefault ) return;
    winid(id)->setIconText(name);
 }
//______________________________________________________________________________
void  TGQt::Warp(Int_t ix, Int_t iy, Window_t id) {
   // Set pointer position.
   // ix       : New X coordinate of pointer
   // iy       : New Y coordinate of pointer
   // Coordinates are relative to the origin of the window id
   // or to the origin of the current window if id == 0.
   if (id == kNone) {}
   else {
       QCursor::setPos(wid(id)->mapToGlobal(QPoint(ix,iy)));
   }
}
//______________________________________________________________________________
 void         TGQt::SetClassHints(Window_t, char *, char *)
 {
    // Sets the windows class and resource name.
#ifdef QTDEBUG
    fprintf(stderr,"No implementation: TGQt::SetClassHints(Window_t, char *, char *)\n");
#endif
 }
//______________________________________________________________________________
 void         TGQt::SetMWMHints(Window_t id, UInt_t /*value*/, UInt_t /*funcs*/,
                            UInt_t /*input*/)
 {
    // Sets decoration style.
    // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).
//---- MWM hints stuff
// These constants werer broowed from TGFrame.h to avoid circular depemdency.
// The right place for them is sopmewhere in "base" (guitype.h for example)
enum EMWMHints {
   // functions
   kMWMFuncAll      = BIT(0),
   kMWMFuncResize   = BIT(1),
   kMWMFuncMove     = BIT(2),
   kMWMFuncMinimize = BIT(3),
   kMWMFuncMaximize = BIT(4),
   kMWMFuncClose    = BIT(5),

   // input mode
   kMWMInputModeless                = 0,
   kMWMInputPrimaryApplicationModal = 1,
   kMWMInputSystemModal             = 2,
   kMWMInputFullApplicationModal    = 3,

   // decorations
   kMWMDecorAll      = BIT(0),
   kMWMDecorBorder   = BIT(1),
   kMWMDecorResizeH  = BIT(2),
   kMWMDecorTitle    = BIT(3),
   kMWMDecorMenu     = BIT(4),
   kMWMDecorMinimize = BIT(5),
   kMWMDecorMaximize = BIT(6)
};

   //MWMHintsProperty_t prop;

   //prop.fDecorations = value;
   //prop.fFunctions   = funcs;
   //prop.fInputMode   = input;
   //prop.fFlags       = kMWMHintsDecorations |
   //                    kMWMHintsFunctions   |
   //                    kMWMHintsInputMode;

   //XChangeProperty(fDisplay, (Window) id, gMOTIF_WM_HINTS, gMOTIF_WM_HINTS, 32,
   //                PropModeReplace, (UChar_t *)&prop, kPropMWMHintElements);

    if (id == kNone || id == kDefault ) return;
    // QSizePolicy  thisPolicy = wid(id)->sizePolicy ();
    // wid(id)->setSizePolicy ( thisPolicy );
#ifdef QTDEBUG
    fprintf(stderr,"No implementation: TGQt::SetMWMHints(Window_t, UInt_t, UInt_t, UInt_t)\n");
#endif
 }
//______________________________________________________________________________
void TGQt::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   // Tells the window manager the desired position [x,y] of window "id".
   if (id == kNone || id == kDefault ) return;
#ifdef QTDEBUG
   fprintf(stderr,"No implementation: TGQt::SetWMPosition(Window_t id, Int_t x=%d, Int_t y=%d\n",x,y);
#endif
   wid(id)->move(x,y);
}
//______________________________________________________________________________
void TGQt::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   // Tells window manager the desired size of window "id".
   //
   // w - the width
   // h - the height
   if (id == kNone || id == kDefault ) return;
  // vf to review QSizePolicy  thisPolicy = wid(id)->sizePolicy ();
   wid(id)->setBaseSize(int(w),int(h));
  // fprintf(stderr,"No implementation: TGQt::SetWMSize(Window_t id, UInt_t w=%d, UInt_t h=%d\n",w,h);
  // SafeCallW32(id)->W32_Rescale(id,w, h);
  // wid(id)->setSizePolicy(thisPolicy);
}
//______________________________________________________________________________
 void         TGQt::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax,
                               UInt_t winc, UInt_t hinc)
{
   // Gives the window manager minimum and maximum size hints of the window
   // "id". Also specify via "winc" and "hinc" the resize increments.
   //
   // wmin, hmin - specify the minimum window size
   // wmax, hmax - specify the maximum window size
   // winc, hinc - define an arithmetic progression of sizes into which
   //              the window to be resized (minimum to maximum)
    if (id == kNone || id == kDefault ) return;
    QWidget &w = *wid(id);
    w.setMinimumSize  ( int(wmin), int(hmin) );
    w.setMaximumSize  ( int(wmax), int(hmax) );
    w.setSizeIncrement( int(winc), int(hinc) );
 }
//______________________________________________________________________________
 void         TGQt::SetWMState(Window_t id, EInitialState /*state*/)
{
   // Sets the initial state of the window "id": either kNormalState
   // or kIconicState.
    if (id == kNone || id == kDefault ) return;
#ifdef QTDEBUG
        fprintf(stderr,"No implementation: TGQt::SetWMState( . . . )\n");
#endif
 }
//______________________________________________________________________________
void   TGQt::SetWMTransientHint(Window_t id, Window_t /*main_id*/ )
{
   // Tells window manager that the window "id" is a transient window
   // of the window "main_id". A window manager may decide not to decorate
   // a transient window or may treat it differently in other ways.
    if (id == kNone || id == kDefault ) return;
#ifdef QTDEBUG
    fprintf(stderr,"No implementation: TGQt::SetWMTransientHint( . . . )\n");
#endif
 }
//______________________________________________________________________________
void  TGQt::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                   const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y)
   if (id == kNone) return;

   if (s && s[0] && len) {
      TQtPainter paint(iwid(id),qtcontext(gc));
      //   Pick the font from the context
      const QColor &fontColor = qtcontext(gc).paletteForegroundColor ();
      paint.setPen(fontColor);
      paint.setBrush(fontColor);
      if (qtcontext(gc).fFont)  paint.setFont(*qtcontext(gc).fFont);
      // fprintf(stderr,"TGQt::DrawString  \"%s\":%d with color %s\n",s,len,(const char *)fontColor.name());
      paint.drawText (x, y,  GetTextDecoder()->toUnicode(s),len);
   }
}
//______________________________________________________________________________
Int_t TGQt::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font.

   Int_t textWidth = 0;
   if (len >0 && s && s[0] != 0 ) {
      QFontMetrics metric(*(QFont *)font);
      char* str = new char[len+1];
      memset(str,0,len+1);
      strncpy(str,s,len);
      QString qstr = strncpy(str,s,len);
      delete [] str;
      textWidth = metric.width(qstr,len);
      // fprintf(stderr," TGQt::TextWidth  %d %d <%s> \n", textWidth, len, (const char *)qstr);
   }
   return textWidth;
}
//______________________________________________________________________________
void TGQt::GetFontProperties(FontStruct_t fs, Int_t &max_ascent, Int_t &max_descent)
{
   // The ascent of a font is the distance from the baseline
   //            to the highest position characters extend to.
   // The descent is the distance from the base line to the
   //             lowest point characters extend to
   //             (Note that this is different from X, which adds 1 pixel.)
   QFontMetrics metrics(*(QFont *)fs);
   max_ascent  = metrics.ascent ();
   max_descent = metrics.descent();
}
//______________________________________________________________________________
 void TGQt::GetGCValues(GContext_t  gc, GCValues_t &gval)
 {
    // Get current values from graphics context gc. Which values of the
    // context to get is encoded in the GCValues::fMask member. If fMask = 0
    // then copy all fields.

    assert(gval.fMask == kGCFont);
    gval.fFont = (FontStruct_t)qtcontext(gc).fFont;
 }
//______________________________________________________________________________
 FontStruct_t TGQt::GetFontStruct(FontH_t fh) { return (FontStruct_t)fh; }
//______________________________________________________________________________
 void         TGQt::ClearWindow(Window_t id)
 {
    // Clear window.
   if (id == kNone || id == kDefault ) return;
   wid(id)->erase();
 }

//---- Key symbol mapping
struct KeyQSymbolMap_t {
   Qt::Key fQKeySym;
   EKeySym fKeySym;
};

//---- Mapping table of all non-trivial mappings (the ASCII keys map
//---- one to one so are not included)

static KeyQSymbolMap_t gKeyQMap[] = {
   {Qt::Key_Escape,    kKey_Escape},
   {Qt::Key_Tab,       kKey_Tab},
   {Qt::Key_Backtab,   kKey_Backtab},
#if QT_VERSION < 0x40000
   {Qt::Key_BackSpace, kKey_Backspace},
#else /* QT_VERSION */
   {Qt::Key_Backspace, kKey_Backspace},
#endif /* QT_VERSION */
   {Qt::Key_Return,    kKey_Return},
   {Qt::Key_Insert,    kKey_Insert},
   {Qt::Key_Delete,    kKey_Delete},
   {Qt::Key_Pause,     kKey_Pause},
   {Qt::Key_Print,     kKey_Print},
   {Qt::Key_SysReq,    kKey_SysReq},
   {Qt::Key_Home,      kKey_Home},       // cursor movement
   {Qt::Key_End,       kKey_End},
   {Qt::Key_Left,      kKey_Left},
   {Qt::Key_Up,        kKey_Up},
   {Qt::Key_Right,     kKey_Right},
   {Qt::Key_Down,      kKey_Down},
#if QT_VERSION < 0x40000
   {Qt::Key_Prior,     kKey_Prior},
   {Qt::Key_Next,      kKey_Next},
#else /* QT_VERSION */
   {Qt::Key_PageUp,     kKey_Prior},
   {Qt::Key_PageDown,      kKey_Next},
#endif /* QT_VERSION */
   {Qt::Key_Shift,     kKey_Shift},
   {Qt::Key_Control,   kKey_Control},
   {Qt::Key_Meta,      kKey_Meta},
   {Qt::Key_Alt,       kKey_Alt},
   {Qt::Key_CapsLock,  kKey_CapsLock},
   {Qt::Key_NumLock ,  kKey_NumLock},
   {Qt::Key_ScrollLock, kKey_ScrollLock},
   {Qt::Key_Space,     kKey_Space},  // numeric keypad
   {Qt::Key_Tab,       kKey_Tab},
   {Qt::Key_Enter,     kKey_Enter},
   {Qt::Key_Equal,     kKey_Equal},
   {Qt::Key_F1,        kKey_F1 },
   {Qt::Key_F2,        kKey_F2 },
   {Qt::Key_F3,        kKey_F3 },
   {Qt::Key_F4,        kKey_F4 },
   {Qt::Key_PageUp,    kKey_PageUp },
   {Qt::Key_PageDown,  kKey_PageDown },
   {Qt::Key(0), (EKeySym) 0}   
};
//______________________________________________________________________________________
static inline Int_t MapKeySym(int key, bool toQt=true)
{
   for (int i = 0; gKeyQMap[i].fKeySym; i++) {	// any other keys
      if (toQt) {
        if (key ==  gKeyQMap[i].fKeySym ) {
           return   UInt_t(gKeyQMap[i].fQKeySym);
        }
      } else {
        if (key ==  gKeyQMap[i].fQKeySym) {
           return   UInt_t(gKeyQMap[i].fKeySym);
        }
      }
   }
#if 0
   UInt_t text;
#if QT_VERSION < 0x40000
   QCString r = gQt->GetTextDecoder()->fromUnicode(qev.text());
#else /* QT_VERSION */
   Q3CString r = gQt->GetTextDecoder()->fromUnicode(qev.text());
#endif /* QT_VERSION */
   qstrncpy((char *)&text, (const char *)r,1);
   return text;
#else
   return key;
#endif
}
//______________________________________________________________________________
 Int_t        TGQt::KeysymToKeycode(UInt_t keysym) {
    // Convert a keysym to the appropriate keycode. For example keysym is
    // a letter and keycode is the matching keyboard key (which is dependend
    // on the current keyboard mapping).
    return Int_t(MapKeySym(keysym));
 }
 //______________________________________________________________________________
 void TGQt::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
    UInt_t w, UInt_t h)
 {
    if (id == kNone) return;
    // Draw a filled rectangle. Filling is done according to the gc.
    TQtPainter paint(iwid(id),qtcontext(gc));
    if (qtcontext(gc).HasValid(QtGContext::kTileRect) ) {
       paint.drawTiledPixmap(x,y,w,h,*qtcontext(gc).fTilePixmap);
    } else {
       if (qtcontext(gc).HasValid(QtGContext::kStipple)) {
          // qtcontext(gc).fBrush.setStyle(Qt::FDiagPattern);
          if (qtcontext(gc).HasValid(QtGContext::kBrush) ) {
             // paint.setPen(Qt::black);
             //paint.setBackgroundColor(qtcontext(gc).paletteBackgroundColor());
             paint.setPen(qtcontext(gc).paletteForegroundColor());
          } else {
             paint.setBackgroundColor(Qt::white);
             paint.setPen(Qt::black);
          }
          paint.setBackgroundMode( Qt::OpaqueMode); // Qt::TransparentMode
       }
       paint.fillRect ( x, y, w, h, qtcontext(gc).fBrush );
    }
 }
 //______________________________________________________________________________
 void TGQt::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
    UInt_t w, UInt_t h)
 {
    if (id == kNone) return;
    // Draw a rectangle outline.
    TQtPainter paint(iwid(id),qtcontext(gc));
    paint.setBrush(Qt::NoBrush);
    paint.drawRect ( x, y, w, h);
 }
 //______________________________________________________________________________
 void TGQt::DrawSegments(Drawable_t id, GContext_t gc, Segment_t * seg, Int_t nseg)
 {
    // Draws multiple line segments. Each line is specified by a pair of points.
    if (id == kNone) return;
    TQtPainter paint(iwid(id),qtcontext(gc));
#if QT_VERSION < 0x40000
    QPointArray segments(2*nseg);
#else /* QT_VERSION */
    Q3PointArray segments(2*nseg);
#endif /* QT_VERSION */
    for (int i=0;i<nseg;i++) {
       segments.setPoint (2*i,   seg[i].fX1, seg[i].fY1);
       segments.setPoint (2*i+1, seg[i].fX2, seg[i].fY2);
    }
    paint.drawLineSegments ( segments);
 }
//______________________________________________________________________________
 void         TGQt::SelectInput(Window_t id, UInt_t evmask)
 {
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   // UInt_t xevmask;

   //MapEventMask(evmask, xevmask);

   //XSelectInput(fDisplay, (Window) id, xevmask);

       //The XSelectInput function requests that the X server
       //report the events associated with the specified event
       //mask.  Initially, X will not report any of these events.
       //Events are reported relative to a window.  If a window is
       //not interested in a device event, it usually propagates to
       //the closest ancestor that is interested, unless the
       //do_not_propagate mask prohibits it.

       //Setting the event-mask attribute of a window overrides any
       //previous call for the same window but not for other
       //clients.  Multiple clients can select for the same events
       //on the same window with the following restrictions:

       //·    Multiple clients can select events on the same window
       //     because their event masks are disjoint.  When the X
       //     server generates an event, it reports it to all
       //     interested clients.

       //·    Only one client at a time can select Circu­
       //     lateRequest, ConfigureRequest, or MapRequest events,
       //     which are associated with the event mask Substructur­
       //     eRedirectMask.

       //·    Only one client at a time can select a ResizeRequest
       //     event, which is associated with the event mask Resiz­
       //     eRedirectMask.

       //·    Only one client at a time can select a ButtonPress
       //     event, which is associated with the event mask But­
       //     tonPressMask.

       //The server reports the event to all interested clients.

//       QClientFilter()->RemovePointerGrab(0);
    ((TQtClientWidget*)wid(id))->SelectInput(evmask);
}
//______________________________________________________________________________
Window_t  TGQt::GetInputFocus()
{
   // Returns the window id of the window having the input focus.
   TQtClientWidget *focus = 0;
   QWidget *f = qApp->focusWidget ();
   if (f) {
     focus = dynamic_cast<TQtClientWidget*>(f);
     assert(focus);
   }
   return wid(focus);
}
//______________________________________________________________________________
 void         TGQt::SetInputFocus(Window_t id)
 {
   // Set keyboard input focus to window id.
    if (id == kNone || id == kDefault ) return;
    wid(id)->setFocus ();
 }
//______________________________________________________________________________
void         TGQt::LookupString(Event_t *ev, char *tmp, Int_t /*n*/, UInt_t &keysym)
{
    // Convert the keycode from the event structure to a key symbol (according
    // to the modifiers specified in the event structure and the current
    // keyboard mapping). In buf a null terminated ASCII string is returned
    // representing the string that is currently mapped to the key code.

    keysym = ev->fCode;
    // we have to accomodate the new ROOT GUI logic.
    // the information about the "ctrl" key should provided TWICE nowadays 12.04.2005 vf.
    if (ev->fState & kKeyControlMask) keysym -= keysym > 'Z' ? 96 :64;
//    if (ev->fState & kKeyControlMask) if (isupper(keysym)) keysym += 32;
    *tmp = keysym; tmp++;
    *tmp = '\0';
}
//______________________________________________________________________________
void         TGQt::TranslateCoordinates(Window_t src, Window_t dest,
                                   Int_t src_x, Int_t src_y,
                                   Int_t & dest_x, Int_t & dest_y,
                                   Window_t & child)
{
   // TranslateCoordinates translates coordinates from the frame of
   // reference of one window to another. If the point is contained
   // in a mapped child of the destination, the id of that child is
   // returned as well.
   QWidget *wSrc = wid(src);
   QWidget *wDst = wid(dest);
   child = kNone;
   //Local variables to keep the dest coordinates
   Int_t  destX;
   Int_t  destY;

   if (!wSrc) wSrc = QApplication::desktop();
   if (!wDst) wDst = QApplication::desktop();
   assert(wSrc && wDst);
   if (src == dest) {
      destX = src_x; destY= src_y;
   } else {
      QPoint mapped = wDst->mapFromGlobal(wSrc->mapToGlobal(QPoint(src_x,src_y)));
      destX = mapped.x(); destY = mapped.y();
   }
   TQtClientWidget* tmpW = dynamic_cast<TQtClientWidget*>(wDst->childAt ( destX, destY, false ));
   if (tmpW) {
      child = wid(tmpW);
   }
   dest_x = destX; dest_y = destY;
   // fprintf(stderr," Translate the  coordinate src %d %d, dst %d %d; child = %d \n", src_x, src_y, dest_x, dest_y, child);
 }
//______________________________________________________________________________
void         TGQt::GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).
   
   x =  y = 0;
   if (id == kNone || id == kDefault )
   {
      QDesktopWidget *d = QApplication::desktop();
      w = d->width();   // returns desktop width
      h = d->height();  // returns desktop height
   } else {
         QPixmap *thePix = dynamic_cast<QPixmap*>(iwid(id) );
         if (thePix) {
//            *fQPixmapGuard.Pixmap(pix)
            w = thePix->width();     // returns pixmap width
            h = thePix->height();    // returns pixmap height
         } else {
            TQtClientWidget* theWidget = dynamic_cast<TQtClientWidget*>( wid(id) );
            if (theWidget) {
               const QRect &gWidget=theWidget->frameGeometry ();
               // theWidget->dumpObjectInfo () ;
               x = gWidget.x();
               y = gWidget.y();
               w = gWidget.width();
               h = gWidget.height();
            } else {         
               QDesktopWidget *d = QApplication::desktop();
               w = d->width();     // returns desktop width
               h = d->height();    // returns desktop height
            }
         }
     }
 }
//______________________________________________________________________________
void  TGQt::FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt)
{
   // FillPolygon fills the region closed by the specified path.
   // The path is cosed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.
   if (id == kNone) return;
   if (npnt > 1) {
      TQtPainter paint(iwid(id),qtcontext(gc));
#if QT_VERSION < 0x40000
      QPointArray pa(npnt);
#else /* QT_VERSION */
      Q3PointArray pa(npnt);
#endif /* QT_VERSION */
      Int_t x = points[0].fX;
      Int_t y = points[0].fY;
      pa.setPoint(0,x,y);
      for (int i=1;i<npnt;i++)  pa.setPoint(i,points[i].fX,points[i].fY);
      paint.drawConvexPolygon (pa);
   }
}
//______________________________________________________________________________
void  TGQt::QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                        Int_t &root_x, Int_t &root_y, Int_t &win_x,
                        Int_t &win_y, UInt_t &mask)
{
   //Returns the position of the cursor (hot spot) in global screen coordinates
   if (id == kNone) return;

   QPoint position     = QCursor::pos();
   QWidget *thisWidget = wid(id);
   QWidget *topWiget   = thisWidget->topLevelWidget();

   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   QPoint rootPosition = topWiget->mapFromGlobal( position );
   root_x = rootPosition.x(); root_y = rootPosition.y();
   rootw  = rootwid(topWiget);

   // The pointer coordinates returned to win_x and win_y are relative to
   // the origin of the specified window.
   QPoint win_pos = thisWidget->mapFromGlobal( position );
   win_x = win_pos.x(); win_y = win_pos.y();

   TQtClientWidget *ch = (TQtClientWidget * )thisWidget->childAt (win_x,win_y);
   childw = ch ? wid(ch): kNone;

   // QueryPointer returns the current logical state of the
   // keyboard buttons and the modifier keys in mask.

//> Can you say whether it is possible to query the current button state
//> (Qt::ButtonState ) with no QMouseEvent?
//
//I am afraid that this is not possible using Qt, you will need to use
//platform dependent code to do this I am afraid.
//
//Have a nice day!
//
//Andy
//--
//Technicial Support Technician
//Trolltech AS, Waldemar Thranes gate 98, NO-0175 Oslo, Norway

   mask = 0;
}
//______________________________________________________________________________
void         TGQt::SetBackground(GContext_t gc, ULong_t background)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).
   // The interface is confusing . This function MUST be not here (VF 07/07/2003)
   
   qtcontext(gc).SetBackground(background);
}

//______________________________________________________________________________
void         TGQt::SetForeground(GContext_t gc, ULong_t foreground)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).
   // The interface is confusing . This function MUST be not here (VF 07/07/2003)
   
   qtcontext(gc).SetForeground(foreground);
}
//______________________________________________________________________________
void         TGQt::SetClipRectangles(GContext_t gc, Int_t x, Int_t y,
                                    Rectangle_t * recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.
   // Rectangle structure (maps to the X11 XRectangle structure)
   if (n <=0 ) return;
   Region_t clip  = CreateRegion();
   for (int i=0;i<n;i++)
      UnionRectWithRegion(recs,clip,clip);
   ((QRegion *)clip)->translate(x,y);
   qtcontext(gc).fClipRegion = *(QRegion *)clip;
   SETBIT(qtcontext(gc).fMask,QtGContext::kClipRegion);
   DestroyRegion(clip);
}
//______________________________________________________________________________
void         TGQt::Update(Int_t mode)
{
   // Flush (mode = 0, default) or synchronize (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.
#ifndef R__QTX11
   QApplication::flush ();
#else
   if (mode == 0)
      QApplication::flush ();
   if (mode == 1) {
      QApplication::syncX ();
   }
#endif
}
//------------------------------------------------------------------------------
//
//  Region functions. Using QRegion instead of X-Window regions.
//  Event though they are static "by nature".
//  We can not make them static because they are virtual ones.
//  Written by Yuri

//______________________________________________________________________________
Region_t TGQt::CreateRegion()
{
   // Create a new empty region.

   QRegion *reg = new QRegion();
   return (Region_t) reg;
}
//______________________________________________________________________________
void TGQt::DestroyRegion(Region_t reg)
{
   // Destroy region.

   delete (QRegion*) reg;
}
//______________________________________________________________________________
void TGQt::UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest)
{
   // Union of rectangle with a region.

   if( !rect || src == 0 || dest == 0 )
      return;

   QRegion rc(QRect( rect->fX, rect->fY, rect->fWidth, rect->fHeight ));
   QRegion &rSrc  = *(QRegion*) src;
   QRegion &rDest = *(QRegion*) dest;

   rDest = rSrc + rc;
}
//______________________________________________________________________________
Region_t TGQt::PolygonRegion(Point_t *points, Int_t np, Bool_t winding)
{
   // Create region for the polygon defined by the points array.
   // If winding is true use WindingRule else EvenOddRule as fill rule.
   if( np<0 || !points )
      return 0;

#if QT_VERSION < 0x40000
   QPointArray pa;
#else /* QT_VERSION */
   Q3PointArray pa;
#endif /* QT_VERSION */
   pa.resize( np );
   for(int i=0; i<np; i++)
      pa.setPoint( i, points[i].fX, points[i].fY );

   return (Region_t) new QRegion( pa, winding );
}
//______________________________________________________________________________
void TGQt::UnionRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the union of rega and regb and return result region.
   // The output region may be the same result region.

   if( !rega || !regb || !result )
      return;

   QRegion& a = *(QRegion*) rega;
   QRegion& b = *(QRegion*) regb;
   QRegion& r = *(QRegion*) result;

   r = a + b;
}
//______________________________________________________________________________
void TGQt::IntersectRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the intersection of rega and regb and return result region.
   // The output region may be the same as the result region.

	if( !rega || !regb || !result )
		return;

	QRegion& a = *(QRegion*) rega;
	QRegion& b = *(QRegion*) regb;
	QRegion& r = *(QRegion*) result;

	r = a & b;
}
//______________________________________________________________________________
void TGQt::SubtractRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Subtract rega from regb.

   if( !rega || !regb || !result )
      return;

   QRegion& a = *(QRegion*) rega;
   QRegion& b = *(QRegion*) regb;
   QRegion& r = *(QRegion*) result;

   r = a - b;
}
//______________________________________________________________________________
void TGQt::XorRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Calculate the difference between the union and intersection of
   // two regions.
   if( !rega || !regb || !result )
      return;

   QRegion& a = *(QRegion*) rega;
   QRegion& b = *(QRegion*) regb;
   QRegion& r = *(QRegion*) result;

   r = a ^ b;
}
//______________________________________________________________________________
Bool_t TGQt::EmptyRegion(Region_t reg)
{
   // Return true if the region is empty.

   if( !reg )
		return true;

	QRegion& r = *(QRegion*) reg;

	return r.isEmpty();
}
//______________________________________________________________________________
Bool_t TGQt::PointInRegion(Int_t x, Int_t y, Region_t reg)
{
   // Returns true if the point x,y is in the region.
	if( !reg )
		return false;

	QRegion& r = *(QRegion*) reg;
	return r.contains( QPoint(x, y) );
}
//______________________________________________________________________________
Bool_t TGQt::EqualRegion(Region_t rega, Region_t regb)
{
   // Returns true if two regions are equal.

   if( !rega || !regb )
      return false;

   QRegion& a = *(QRegion*) rega;
   QRegion& b = *(QRegion*) regb;

   return (a == b);
}
//______________________________________________________________________________
void TGQt::GetRegionBox(Region_t reg, Rectangle_t *rect)
{
   // Return smallest enclosing rectangle.

	if( !reg || !rect )
		return;

	QRegion& r = *(QRegion*) reg;
	QRect rc   = r.boundingRect();

	rect->fX      = rc.x();
	rect->fY      = rc.y();
	rect->fWidth  = rc.width();
	rect->fHeight = rc.height();
}
//______________________________________________________________________________
char **TGQt::ListFonts(const char *fontname, Int_t max, Int_t &count)
{
   // Returns list of font names matching fontname regexp, like "-*-times-*".
   // The pattern string can contain any characters, but each asterisk (*)
   // is a wildcard for any number of characters, and each question mark (?)
   // is a wildcard for a single character. If the pattern string is not in
   // the Host Portable Character Encoding, the result is implementation
   // dependent. Use of uppercase or lowercase does not matter. Each returned
   // string is null-terminated.
   //
   // fontname - specifies the null-terminated pattern string that can
   //            contain wildcard characters
   // max      - specifies the maximum number of names to be returned
   // count    - returns the actual number of font names

   // ------------------------------------------------------
   //  ROOT uses nin-portable XLDF font description:
   //  XLFD 
   // ------------------------------------------------------

   // The X Logical Font Descriptor (XLFD) is a text string made up of 13 parts separated by a minus sign, i.e.: 
   // -Misc-Fixed-Medium-R-Normal-13-120-75-75-C-70-ISO8859-1 -Adobe-Helvetica-Medium-R-Normal-12-120-75-75-P-67-ISO8859-1 
   // 
   // 
   // FOUNDRY 
   // text name of font creator 
   // 
   // FAMILY_NAME 
   // name of the font. Related fonts generally have the same base names; i.e. helvetica, helvetica narrow , etc. 
   
   // WEIGHT_NAME 
   // usually one of [light|medium|demibold|bold] but other types may exist 
   // 
   // SLANT 
   // one of [r|i|o]. i and o are used similarly, AFAIK 
   // 
   // SETWIDTH_NAME 
   // [normal|condensed|narrow|double wide] 
   // 
   // ADD_STYLE_NAME 
   // not a classification field, used only for additional differentiation 
   //
   // PIXEL_SIZE 
   // 0 = scalable font; integer typicially height of bounding box 
   //
   // POINT_SIZE 
   // typically height of bounding box in tenths of pixels 
   // 
   // RESOLUTION_X 
   // horizontal dots per inch 
   // 
   // RESOLUTION_Y 
   // vertical dots per inch 
   // 
   // SPACING 
   // [p|m|c] p = proportional, m = monospaced, c = charcell. Charcell is a special case of monospaced where no glyphs have pixels outside the character cell; i.e. there is no kerning (no negative metrics). 
   // 
   // AVERAGE_WIDTH 
   // unweighted arithmetic mean of absolute value of width of each glyph in tenths of pixels 
   // CHARSET_REGISTRY and CHARSET_ENCODING 
   // the chararterset used to encode the font; ISO8859-1 for Latin 1 fonts    fprintf(stderr,"No implementation: TGQt::ListFonts\n");
   
   //  Check whether "Symbol" font is available
    count = 0;
    TXlfd  patternFont(fontname);
    QFontDatabase fdb;
    QStringList xlFonts;
    QStringList families = fdb.families();
    for ( QStringList::Iterator f = families.begin(); f != families.end(); ++f ) {
        QString family = *f;
        QStringList styles = fdb.styles( family );
        for ( QStringList::Iterator s = styles.begin(); s != styles.end(); ++s ) {
            QString style = *s;
            // fprintf(stderr," family %s style = %s\n", (const char *)family, (const char *) style);
            Int_t bold   = fdb.bold  (family, style);
            Int_t italic = fdb.italic(family, style);
            TXlfd currentFont(family,bold,italic);
            if (currentFont != patternFont) continue;
            
#if QT_VERSION < 0x40000
            QValueList<int> sizes = fdb.pointSizes( family, style );
            for ( QValueList<int>::Iterator points = sizes.begin();
#else /* QT_VERSION */
            Q3ValueList<int> sizes = fdb.pointSizes( family, style );
            for ( Q3ValueList<int>::Iterator points = sizes.begin();
#endif /* QT_VERSION */
                  points != sizes.end() && (Int_t)xlFonts.size() < max; ++points ) 
            {
              currentFont.SetPointSize(*points);
              if (currentFont ==  patternFont )  xlFonts.push_back(currentFont.ToString());
            }
        }
    }
    count = xlFonts.size();
    char **listFont = 0;
    if (count) {
       char **list = listFont = new char*[count+1];  list[count] = 0;
       for ( QStringList::Iterator it = xlFonts.begin(); it != xlFonts.end(); ++it ) {
          char *nextFont = new char[(*it).length()+1];
          *list = nextFont; list++;
          strncpy(nextFont,(*it).ascii(),(*it).length());
       }
    }
    return listFont;
}

//______________________________________________________________________________
void TGQt::FreeFontNames(char ** fontlist)
{
   // Frees the specified the array of strings "fontlist".
//   fprintf(stderr,"No implementation: TGQt::FreeFontNames\n");
   char ** list =  fontlist;
   while (*list) {  delete [] *list; list++;  }
   delete [] fontlist;
}

//______________________________________________________________________________
Drawable_t TGQt::CreateImage(UInt_t width, UInt_t height)
{
   // Allocates the memory needed for an drawable.
   //
   // width  - the width of the image, in pixels
   // height - the height of the image, in pixels
   int depth = GetDepth();

   if      (depth > 8) depth = 32;
   else if (depth > 1) depth = 8;
   else                depth = 1;

   QImage *image = new QImage(width,height,depth);
   return Drawable_t(image);
}

//______________________________________________________________________________
void TGQt::GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height)
{
   // Returns the width and height of the image id
   QImage *image = (QImage *)id;
   if (image) {
      width  = image->width();
      height = image->height();
   }
}

//______________________________________________________________________________
void TGQt::PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel)
{
   // Overwrites the pixel in the image with the specified pixel value.
   // The image must contain the x and y coordinates.
   //
   // id    - specifies the image
   // x, y  - coordinates
   // pixel - the new pixel value
   //
   QImage *image = (QImage *)id;
   if (image) image->setPixel(x,y,QtColor(pixel).rgb());
}

//______________________________________________________________________________
void TGQt::PutImage(Drawable_t id, GContext_t gc,Drawable_t img, Int_t dx, Int_t dy,
                         Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Combines an image with a rectangle of the specified drawable. The
   // section of the image defined by the x, y, width, and height arguments
   // is drawn on the specified part of the drawable.
   //
   // id   - the drawable
   // gc   - the GC
   // img  - the image you want combined with the rectangle
   // dx   - the offset in X from the left edge of the image
   // dy   - the offset in Y from the top edge of the image
   // x, y - coordinates, which are relative to the origin of the
   //        drawable and are the coordinates of the subimage
   // w, h - the width and height of the subimage, which define the
   //        rectangle dimensions
   //
   // GC components in use: function, plane-mask, subwindow-mode,
   // clip-x-origin, clip-y-origin, and clip-mask.
   // GC mode-dependent components: foreground and background.
   // (see also the GCValues_t structure)
    const QImage *image = (QImage *)img;
    if (image) {
       TQtPainter pnt(iwid(id),qtcontext(gc));
#if QT_VERSION < 0x40000
       int conversionFlags=0;
#else /* QT_VERSION */
       Qt::ImageConversionFlag conversionFlags=Qt::AutoColor;
#endif /* QT_VERSION */
       pnt.drawImage(dx,dy, *image, x,y,w,h,conversionFlags);
       //   Qt::ImageConversionFlags
       //   The conversion flag is a bitwise-OR of the following values.
       //   The options marked "(default)" are set if no other values
       //   from the list are included (since the defaults are zero):
       //  --  Color/Mono preference (ignored for QBitmap)  --
       //      Qt::AutoColor - (default) - If the image has depth 1 and
       //                      contains only black and white pixels, the
       //                      pixmap becomes monochrome.
       //      Qt::ColorOnly - The pixmap is dithered/converted to the native
       //                      display depth.
       //      Qt::MonoOnly -  The pixmap becomes monochrome. If necessary, it
       //                      is dithered using the chosen dithering algorithm.
       //
       //  --  Dithering mode preference for RGB channels  --
       //      Qt::DiffuseDither   - (default) - A high-quality dither.
       //      Qt::OrderedDither   - A faster, more ordered dither.
       //      Qt::ThresholdDither - No dithering; closest color is used.
       //
       //  --  Dithering mode preference for alpha channel
       //      Qt::ThresholdAlphaDither - (default) - No dithering.
       //      Qt::OrderedAlphaDither   - A faster, more ordered dither.
       //      Qt::DiffuseAlphaDither   - A high-quality dither.
       //      Qt::NoAlpha              - Not supported.
       //
       //  --  Color matching versus dithering preference
       //      Qt::PreferDither - (default when converting to a pixmap)
       //                         - Always dither 32-bit images when the image
       //                           is converted to 8 bits.
       //      Qt::AvoidDither  - (default when converting for the purpose of saving to file)
       //                         - Dither 32-bit images only if the image
       //                           has more than 256 colors and it is being converted to 8 bits.
       //      Qt::AutoDither   - Not supported.
       //
       //  --  The following are not values that are used directly, but masks for the above classes:
       //      Qt::ColorMode_Mask   - Mask for the color mode.
       //      Qt::Dither_Mask      - Mask for the dithering mode for RGB channels.
       //      Qt::AlphaDither_Mask - Mask for the dithering mode for the alpha channel.
       //      Qt::DitherMode_Mask  - Mask for the mode that determines the preference of color
       //                             matching versus dithering.
       //
       //Using 0 as the conversion flag sets all the default options.
    }
}

//______________________________________________________________________________
void TGQt::DeleteImage(Drawable_t img)
{
   // Deallocates the memory associated with the image img
   delete (QImage *)img;
}

#if ROOT_VERSION_CODE < ROOT_VERSION(4,01,02)
//______________________________________________________________________________
ULong_t TGQt::GetWinDC(Window_t wind)
{
   // Returns HDC.
   return (wind == kNone) ? 0 :  ULong_t(wid(wind)->handle());
}
#endif

//______________________________________________________________________________
Bool_t  TGQt::IsHandleValid(Window_t /*id*/)
{
   return true;
//    Bool_t ok = (id == 0) || (id == kDefault) || (Bool_t)fQClientGuard.Find(id);
//    if (!ok) fprintf(stderr,"TGObject::GetId() = %ld\n",id);
//    return ok;
}

//______________________________________________________________________________
void  TGQt::SendDestroyEvent(TQtClientWidget *widget) const
{
      // Send the ROOT kDestroyEvent via Qt event loop
   Event_t destroyEvent;
   memset(&destroyEvent,0,sizeof(Event_t));
   destroyEvent.fType      = kDestroyNotify;
   destroyEvent.fWindow    = rootwid(widget);
   destroyEvent.fSendEvent = kTRUE;
   destroyEvent.fTime      = QTime::currentTime().msec();

   // fprintf(stderr,"---- - - > TGQt::SendDestroyEvent %p  %ld \n", widget, wid(widget) );
   ((TGQt *)this)->SendEvent(TGQt::kDefault,&destroyEvent);
}


// -- V.Onuchine's method to back ASIMage


//______________________________________________________________________________
unsigned char *TGQt::GetColorBits(Drawable_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Returns an array of pixels created from a part of drawable (defined by x, y, w, h)
   // in format:
   // b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
   //
   // Pixels are numbered from left to right and from top to bottom.
   // By default all pixels from the whole drawable are returned.
   //
   // Note that return array is 32-bit aligned

   if (!wid || (int(wid) == -1) ) return 0;

   QPaintDevice &dev = *iwid(wid);
   QPixmap *pix=0;
   switch (dev.devType()) {
   case QInternal::Widget:
     pix = &((TQtWidget*)&dev)->GetBuffer();
     break;

   case QInternal::Pixmap: {
      pix = (QPixmap *)&dev;
      break;
                          }
   case QInternal::Picture:
   case QInternal::Printer:
#if QT_VERSION < 0x40000
   case QInternal::UndefinedDevice:
#else /* QT_VERSION */
   // case QInternal::UndefinedDevice:
#endif /* QT_VERSION */
   default: assert(0);
     break;
   };

   if (pix) {
      // Create intermediate pixmap to stretch the original one if any
      QPixmap outMap(0,0);
      if ( (h == w) && (w == UInt_t(-1) ) ) outMap.resize(pix->size());
      else outMap.resize(w,h);

      QImage img = pix->convertToImage();
      if (!img.isNull()) {
         UInt_t *bits = new UInt_t[w*h];
         UInt_t *ibits = (UInt_t *)img.bits();

         int idx = y;
         int iii = 0;
         for (UInt_t j = 0; j < h; j++) {
            for (UInt_t i = 0; i < w; i++) {
               bits[iii + i] = ibits[idx + x + i];
            }
            idx += w;
            iii += w;
         }
         return (unsigned char *)bits;
      }
   }

   return 0;
}


//______________________________________________________________________________
Pixmap_t TGQt::CreatePixmapFromData(unsigned char * bits, UInt_t width,
                                       UInt_t height)
{
   // create pixmap from RGB data. RGB data is in format :
   // b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
   //
   // Pixels are numbered from left to right and from top to bottom.
   // Note that data must be 32-bit aligned

   QImage img(bits, width, height, 32, 0, 0, QImage::LittleEndian);
   QPixmap *p = new QPixmap(img);
   fQPixmapGuard.Add(p);
   return Pixmap_t(rootwid(p));
}

//______________________________________________________________________________
Window_t TGQt::GetCurrentWindow() const
{
   // Return current/selected window pointer.
   fprintf(stderr, " Qt layer is not ready for GetCurrentWindow \n");
   assert(0);

   return (Window_t)(fSelectedBuffer ? fSelectedBuffer : fSelectedWindow);
}
