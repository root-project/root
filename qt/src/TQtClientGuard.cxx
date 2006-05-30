// @(#)root/qt:$Name:  $:$Id: TQtClientGuard.cxx,v 1.8 2006/03/24 15:31:10 antcheva Exp $
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
#if QT_VERSION < 0x40000
#include <qobjectlist.h>
#else /* QT_VERSION */
#include <qobject.h>
#endif /* QT_VERSION */
#include <qbitmap.h>
#if QT_VERSION >= 0x40000
//Added by qt3to4:
#include <QPixmap>
#endif /* QT_VERSION */

//______________________________________________________________________________
void TQtClientGuard::Add(QWidget *w)
{
   // add the widget to list of the "guarded" widget
   fQClientGuard.prepend(w);
   // fprintf(stderr," TQtClientGuard::Add %d %lp %p \n", TGQt::rootwid(w), TGQt::rootwid(w),w );
   connect(w,SIGNAL(destroyed()),this,SLOT(Disconnect()));
}
//______________________________________________________________________________
#if QT_VERSION < 0x40000
TQtClientWidget *TQtClientGuard::Create(QWidget* parent, const char* name, WFlags f)
#else /* QT_VERSION */
TQtClientWidget *TQtClientGuard::Create(QWidget* parent, const char* name, Qt::WFlags f)
#endif /* QT_VERSION */
{
   // TQtClientWidget object factory
   TQtClientWidget *w =  new TQtClientWidget(this,parent,name,f);
   // w->setBackgroundMode(Qt::NoBackground);
   Add(w);
   return  w;
}
//______________________________________________________________________________
void TQtClientGuard::Delete(QWidget *w)
{
   // Delete and unregister the object
   if (w && (fQClientGuard.find(w)>=0))
   {
      w->hide();
      Disconnect(w);
      //((TQtClientWidget *)w)->SetClosing();
      //w->close(true);
      delete w;
      assert( w != QWidget::mouseGrabber() );
   }
}
//______________________________________________________________________________
void TQtClientGuard::Disconnect(QWidget *w)
{
   // Disconnect and unregister the object
   // fprintf(stderr, "TQtClientGuard::Disconnecting widget %p\n", w);
   if ( w && (fQClientGuard.find(w) >=0 ) ) {
      // ungrab the poiner just in case
      QWidget *grabber = QWidget::mouseGrabber();
      fQClientGuard.remove();
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
   fQClientGuard.find(TGQt::wid(id));
   return  fQClientGuard.current();
}
// protected slots:
//______________________________________________________________________________
void TQtClientGuard::Disconnect()
{
   // Disconnect object Qt slot
   QWidget *w = (QWidget *)sender();
   // fprintf(stderr, "Disconnecting  SLOT widget %p\n", w);
   fQClientGuard.find(w);
   if ( fQClientGuard.current() ) {
      if ( w == QWidget::mouseGrabber())
         fprintf(stderr," mouse is still grabbed by the dead wigdet !!!\n");
      fQClientGuard.remove();
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
   // fprintf(stderr," TQtPixmapGuard::Add %d %lp %p \n", TGQt::iwid(w), TGQt::iwid(w),w );
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(int w, int h, const uchar *bits, bool isXbitmap)
{
   QPixmap *p = (QPixmap*)new QBitmap(w,h,bits,isXbitmap);
   Add(p);
   return p;
}
//______________________________________________________________________________
QPixmap* TQtPixmapGuard::Create(int width, int height, int depth)
                                // , Optimization optimization)
{
#if QT_VERSION < 0x40000
   QPixmap *w =  new QPixmap(width,height,depth); // ,optimization);
#else /* QT_VERSION */
   if (depth) {/* fool the compiler wit  Qt4 */ }
   QPixmap *w =  new QPixmap(width,height); // ,optimization);
#endif /* QT_VERSION */
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
void TQtPixmapGuard::Disconnect(QPixmap *w)
{
   // Disconnect QPixmap

   fQClientGuard.find(w);
   if ( fQClientGuard.current() ) {
      fQClientGuard.remove();
   } else {
      fDeadCounter++;
#ifdef QTDEBUG
      printf(" %d Attempt to delete the dead pixmap %p\n",fDeadCounter, w);
#endif
   }
}
//______________________________________________________________________________
QPixmap *TQtPixmapGuard::Pixmap(Pixmap_t id, bool needBitmap)
{
   // Find QPixmap by ROOT pixmap id
   QPixmap *thisPix = 0;
   if (id) {
      fQClientGuard.find((QPixmap *)id);
      assert( (thisPix  = fQClientGuard.current()) &&  (!needBitmap || thisPix->isQBitmap()) ) ;
   }
   return thisPix;
}
//______________________________________________________________________________
QPixmap *TQtPixmapGuard::Find(Window_t /*id*/ )
{
   // return the current QPixmap object

   // fprintf(stderr," TQtPixmapGuard::Find %d %lp %p index=%d\n", id, id, TGQt::wid(id),
   // fQClientGuard.find(TGQt::wid(id));
   return  fQClientGuard.current();
}
// protected slots:
//______________________________________________________________________________
void TQtPixmapGuard::Disconnect()
{
   // Disconnect Qt slot
   QPixmap *w = (QPixmap *)sender();
   fQClientGuard.find(w);
   if ( fQClientGuard.current() ) {
      fQClientGuard.remove();
   }
}
