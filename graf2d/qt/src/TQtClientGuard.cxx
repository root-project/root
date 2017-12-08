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
#include <assert.h>

#include "TQtClientGuard.h"
#include "TGQt.h"
#include <QPixmap>
#include <QBitmap>

////////////////////////////////////////////////////////////////////////////////
/// add the widget to list of the "guarded" widget

void TQtClientGuard::Add(QWidget *w)
{
   fQClientGuard.prepend(w);
   // fprintf(stderr," TQtClientGuard::Add %d %lp %p \n", TGQt::rootwid(w), TGQt::rootwid(w),w );
   connect(w,SIGNAL(destroyed()),this,SLOT(Disconnect()));
}
////////////////////////////////////////////////////////////////////////////////
/// TQtClientWidget object factory

TQtClientWidget *TQtClientGuard::Create(QWidget* mother, const char* name, Qt::WFlags f)
{
   TQtClientWidget *w =  new TQtClientWidget(this,mother,name,f);
   // w->setBackgroundMode(Qt::NoBackground);
   Add(w);
   return  w;
}
////////////////////////////////////////////////////////////////////////////////
/// Delete and unregister the object

void TQtClientGuard::Delete(QWidget *w)
{
   int found = -1;
#if QT_VERSION < 0x40000
   if (w && ( (found = fQClientGuard.find(w))>=0))
#else
   if (w && ( (found = fQClientGuard.indexOf(w))>=0) )
#endif
   {
      w->hide();
      Disconnect(w,found);
      //((TQtClientWidget *)w)->SetClosing();
      //w->close(true);
      w->deleteLater();
      assert( w != QWidget::mouseGrabber() );
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Disconnect and unregister the object
/// fprintf(stderr, "TQtClientGuard::Disconnecting widget %p\n", w);

void TQtClientGuard::Disconnect(QWidget *w, int found)
{
   if ( (found>=0) ||
#if QT_VERSION < 0x40000
      ( w && ( (found = fQClientGuard.find(w)) >=0 ) )  ) {
#else
      ( w && ((found = fQClientGuard.indexOf(w)) >=0 ) )  ) {
#endif
      // ungrab the poiner just in case
      QWidget *grabber = QWidget::mouseGrabber();
#if QT_VERSION < 0x40000
      fQClientGuard.remove();
#else
      fQClientGuard.removeAt(found);
#endif
      disconnect(w,SIGNAL(destroyed()),this,SLOT(Disconnect()));
      if (grabber == w && gQt->IsRegistered(w) )
         gVirtualX->GrabPointer(TGQt::iwid(w), 0, 0, 0, kFALSE);
   } else {
      fDeadCounter++;
#ifdef QTDEBUG
      printf(" %d Attempt to delete the dead widget %p\n",fDeadCounter, w);
#endif
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Disconnect all children of the registered widget

void TQtClientGuard::DisconnectChildren(TQtClientWidget *w)
{
   if (w) {
#if QT_VERSION < 0x40000
      const QObjectList *childList = w->children();
#else /* QT_VERSION */
      const QObjectList &childList = w->children();
#endif /* QT_VERSION */
#if QT_VERSION < 0x40000
      if (childList) {
         QObjectListIterator next(*childList);
         next.toLast();
#else /* QT_VERSION */
      if (!childList.isEmpty()) {
         QListIterator<QObject *> next(childList);
         next.toBack();
#endif /* QT_VERSION */
         QObject *widget = 0;
         // while ( (widget = *next) )
#if QT_VERSION < 0x40000
         for (widget=next.toLast(); (widget = next.current()); --next)
#else /* QT_VERSION */
         while( next.hasPrevious() )
#endif /* QT_VERSION */
         {
#if QT_VERSION >= 0x40000
            widget = next.previous();
#endif /* QT_VERSION */
            if (dynamic_cast<TQtClientWidget*>(widget)) {
               DisconnectChildren((TQtClientWidget*)widget);
            } else {
                // assert(0);// Layout here
            }
         }
      }
      Disconnect(w);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find the object by ROOT id

QWidget *TQtClientGuard::Find(Window_t id)
{
   // fprintf(stderr," TQtClientGuard::Find %d %lp %p\n", id, id, TGQt::wid(id));
   int found = -1;
#if QT_VERSION < 0x40000
   found = fQClientGuard.find(TGQt::wid(id));
#else
   found = fQClientGuard.indexOf(TGQt::wid(id));
#endif
   return  found >=0 ? fQClientGuard.at(found) : 0;
}
// protected slots:
////////////////////////////////////////////////////////////////////////////////
/// Disconnect object Qt slot

void TQtClientGuard::Disconnect()
{
   QWidget *w = (QWidget *)sender();
   // fprintf(stderr, "Disconnecting  SLOT widget %p\n", w);
   int found = -1;
#if QT_VERSION < 0x40000
   found = fQClientGuard.find(w);
#else
   found = fQClientGuard.indexOf(w);
#endif
   if ( found >= 0 ) {
      if ( w == QWidget::mouseGrabber())
         fprintf(stderr," mouse is still grabbed by the dead wigdet !!!\n");
#if QT_VERSION < 0x40000
      fQClientGuard.remove();
#else
      fQClientGuard.removeAt(found);
#endif
      disconnect(w,SIGNAL(destroyed()),this,SLOT(Disconnect()));
   }
}

//______________________________________________________________________________
//
//      TQtPixmapGuard
////////////////////////////////////////////////////////////////////////////////
/// add the widget to list of the "guarded" widget

void TQtPixmapGuard::Add(QPixmap *w)
{
   fQClientGuard.prepend(w);
   SetCurrent(0);
   // fprintf(stderr," TQtPixmapGuard::Add %d %lp %p \n", TGQt::iwid(w), TGQt::iwid(w),w );
}
////////////////////////////////////////////////////////////////////////////////

QPixmap* TQtPixmapGuard::Create(int w, int h, const uchar *bits, bool isXbitmap)
{
   QPixmap *p = new QBitmap(
         QBitmap::fromData (QSize(w,h), bits, isXbitmap ? QImage::Format_MonoLSB : QImage::Format_Mono));
   Add(p);
   return p;
}
////////////////////////////////////////////////////////////////////////////////

QPixmap* TQtPixmapGuard::Create(int width, int height, int depth)
                                // , Optimization optimization)
{
   if (depth) {/* fool the compiler with  Qt4 */ }
   QPixmap *w =  new QPixmap(width,height); // ,optimization);
   Add(w);
   return  w;
}
////////////////////////////////////////////////////////////////////////////////

QPixmap* TQtPixmapGuard::Create(const QString &fileName, const char *format)
//, ColorMode mode)
{
   // QPixmap object factory
   // Constructs a pixmap from the file fileName.

   QPixmap *w =  new QPixmap(fileName,format); //,mode);
   Add(w);
   return  w;
}
////////////////////////////////////////////////////////////////////////////////
/// QPixmap object factory
/// Constructs a pixmap that is a copy of pixmap.

QPixmap* TQtPixmapGuard::Create(const QPixmap &src)
{
   QPixmap *w =  new QPixmap(src);
   Add(w);
   return  w;
}

////////////////////////////////////////////////////////////////////////////////
/// QBitmap object factory

QBitmap* TQtPixmapGuard::Create(const QBitmap &src)
{
   QBitmap *w =  new QBitmap(src);
   Add(w);
   return  w;
}
////////////////////////////////////////////////////////////////////////////////
/// QPixmap object factory
/// Constructs a pixmap from xpm

QPixmap* TQtPixmapGuard::Create (const char* xpm[])
{
   QPixmap *w =  new QPixmap(xpm);
   Add(w);
   return  w;
}
////////////////////////////////////////////////////////////////////////////////
/// Delete and unregister QPixmap

void TQtPixmapGuard::Delete(QPixmap *w)
{
   if (w)
   {
      Disconnect(w);
      delete w;
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Disconnect QPixmap

void TQtPixmapGuard::Disconnect(QPixmap *w, int found)
{
   if (found <0) found =
#if QT_VERSION < 0x40000
                   fQClientGuard.find(w);
#else
                   fQClientGuard.indexOf(w);
#endif
   if ( found >=0 ) {
#if QT_VERSION < 0x40000
      fQClientGuard.remove();
#else
      fQClientGuard.removeAt(found);
#endif
   } else {
      fDeadCounter++;
#ifdef QTDEBUG
      printf(" %d Attempt to delete the dead pixmap %p\n",fDeadCounter, w);
#endif
   }
   SetCurrent(found);
}
////////////////////////////////////////////////////////////////////////////////
/// Find QPixmap by ROOT pixmap id

QPixmap *TQtPixmapGuard::Pixmap(Pixmap_t id, bool needBitmap)
{
   (void)needBitmap;
   QPixmap *thisPix = 0;
   int found = -1;
   if (id) {
#if QT_VERSION < 0x40000
      found = fQClientGuard.find((QPixmap *)id);
      thisPix  = fQClientGuard.current();
      assert( thisPix &&  (!needBitmap || thisPix->isQBitmap()) ) ;
#else
      found = fQClientGuard.indexOf((QPixmap *)id);
      thisPix = found>=0 ? fQClientGuard[found] : 0;
      assert( thisPix  &&  (!needBitmap || thisPix->isQBitmap()) ) ;
#endif
      (void)needBitmap; // used in asserts above.
   }
   SetCurrent(found);
   return thisPix;
}
////////////////////////////////////////////////////////////////////////////////
/// return the current QPixmap object

QPixmap *TQtPixmapGuard::Find(Window_t /*id*/ )
{
   // fprintf(stderr," TQtPixmapGuard::Find %d %lp %p index=%d\n", id, id, TGQt::wid(id),
   // fQClientGuard.find(TGQt::wid(id));
#if QT_VERSION < 0x40000
   return  fQClientGuard.current();
#else
   return  fLastFound >=0 ? fQClientGuard[fLastFound] : 0;
#endif
}
// protected slots:
////////////////////////////////////////////////////////////////////////////////
/// Disconnect Qt slot

void TQtPixmapGuard::Disconnect()
{
   QPixmap *w = (QPixmap *)sender();
   int found = -1;
#if QT_VERSION < 0x40000
   found = fQClientGuard.find(w);
   if ( found >=0 )   fQClientGuard.remove();
#else
   found = fQClientGuard.indexOf(w);
   if ( found >=0 )   fQClientGuard.removeAt(found);
#endif
   SetCurrent(found);
}
