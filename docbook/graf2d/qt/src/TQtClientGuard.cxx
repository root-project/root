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

//______________________________________________________________________________
void TQtClientGuard::Add(QWidget *w)
{
   // add the widget to list of the "guarded" widget
   fQClientGuard.prepend(w);
   // fprintf(stderr," TQtClientGuard::Add %d %lp %p \n", TGQt::rootwid(w), TGQt::rootwid(w),w );
   connect(w,SIGNAL(destroyed()),this,SLOT(Disconnect()));
}
//______________________________________________________________________________
TQtClientWidget *TQtClientGuard::Create(QWidget* mother, const char* name, Qt::WFlags f)
{
   // TQtClientWidget object factory
   TQtClientWidget *w =  new TQtClientWidget(this,mother,name,f);
   // w->setBackgroundMode(Qt::NoBackground);
   Add(w);
   return  w;
}
//______________________________________________________________________________
void TQtClientGuard::Delete(QWidget *w)
{
   // Delete and unregister the object
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
//______________________________________________________________________________
void TQtClientGuard::Disconnect(QWidget *w, int found)
{
   // Disconnect and unregister the object
   // fprintf(stderr, "TQtClientGuard::Disconnecting widget %p\n", w);
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
//______________________________________________________________________________
void TQtClientGuard::DisconnectChildren(TQtClientWidget *w)
{
   // Disconnect all children of the registered widget
   if (w) {
#if QT_VERSION < 0x40000
      const QObjectList *childList = w->children();
#else /* QT_VERSION */
      const QObjectList &childList = w->children();
#endif /* QT_VERSION */
      int nChild = 0;
#if QT_VERSION < 0x40000
      if (childList) {
         nChild = childList->count();
         QObjectListIterator next(*childList);
         next.toLast();
#else /* QT_VERSION */
      if (!childList.isEmpty()) {
         nChild = childList.count();
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

//______________________________________________________________________________
QWidget *TQtClientGuard::Find(Window_t id)
{
   // Find the object by ROOT id

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
//______________________________________________________________________________
void TQtClientGuard::Disconnect()
{
   // Disconnect object Qt slot
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
//______________________________________________________________________________
void TQtPixmapGuard::Add(QPixmap *w)
{
   // add the widget to list of the "guarded" widget
   fQClientGuard.prepend(w);
   SetCurrent(0);
   // fprintf(stderr," TQtPixmapGuard::Add %d %lp %p \n", TGQt::iwid(w), TGQt::iwid(w),w );
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(int w, int h, const uchar *bits, bool isXbitmap)
{
   QPixmap *p = new QBitmap(
         QBitmap::fromData (QSize(w,h), bits, isXbitmap ? QImage::Format_MonoLSB : QImage::Format_Mono));
   Add(p);
   return p;
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(int width, int height, int depth)
                                // , Optimization optimization)
{
   if (depth) {/* fool the compiler with  Qt4 */ }
   QPixmap *w =  new QPixmap(width,height); // ,optimization);
   Add(w);
   return  w;
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(const QString &fileName, const char *format)
//, ColorMode mode)
{
   // QPixmap object factory
   // Constructs a pixmap from the file fileName.

   QPixmap *w =  new QPixmap(fileName,format); //,mode);
   Add(w);
   return  w;
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(const QPixmap &src)
{
   // QPixmap object factory
   // Constructs a pixmap that is a copy of pixmap.
   QPixmap *w =  new QPixmap(src);
   Add(w);
   return  w;
}

//______________________________________________________________________________
QBitmap* TQtPixmapGuard::Create(const QBitmap &src)
{
  // QBitmap object factory

   QBitmap *w =  new QBitmap(src);
   Add(w);
   return  w;
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create (const char* xpm[])
{
   // QPixmap object factory
   // Constructs a pixmap from xpm
   QPixmap *w =  new QPixmap(xpm);
   Add(w);
   return  w;
}
//______________________________________________________________________________
void TQtPixmapGuard::Delete(QPixmap *w)
{
   // Delete and unregister QPixmap
   if (w)
   {
      Disconnect(w);
      delete w;
   }
}
//______________________________________________________________________________
void TQtPixmapGuard::Disconnect(QPixmap *w, int found)
{
   // Disconnect QPixmap
   
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
//______________________________________________________________________________
QPixmap *TQtPixmapGuard::Pixmap(Pixmap_t id, bool needBitmap)
{
   // Find QPixmap by ROOT pixmap id
   QPixmap *thisPix = 0;
   int found = -1;
   if (id) {
#if QT_VERSION < 0x40000
      found = fQClientGuard.find((QPixmap *)id);
      assert( (thisPix  = fQClientGuard.current()) &&  (!needBitmap || thisPix->isQBitmap()) ) ;
#else
      found = fQClientGuard.indexOf((QPixmap *)id);
      thisPix = found>=0 ? fQClientGuard[found] : 0;
      assert( thisPix  &&  (!needBitmap || thisPix->isQBitmap()) ) ;
#endif
   }
   SetCurrent(found);
   return thisPix;
}
//______________________________________________________________________________
QPixmap *TQtPixmapGuard::Find(Window_t /*id*/ )
{
   // return the current QPixmap object

   // fprintf(stderr," TQtPixmapGuard::Find %d %lp %p index=%d\n", id, id, TGQt::wid(id),
   // fQClientGuard.find(TGQt::wid(id));
#if QT_VERSION < 0x40000
   return  fQClientGuard.current();
#else
   return  fLastFound >=0 ? fQClientGuard[fLastFound] : 0;
#endif
}
// protected slots:
//______________________________________________________________________________
void TQtPixmapGuard::Disconnect()
{
   // Disconnect Qt slot
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
