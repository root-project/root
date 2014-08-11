/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename slots use Qt Designer which will
** update this file, preserving your code. Create an init() slot in place of
** a constructor, and a destroy() slot in place of a destructor.
*****************************************************************************/
#include "TCanvas.h"
#include "TClass.h"
#include "TFile.h"

#if QT_VERSION >= 0x40000
//Added by qt3to4:
#include <QPixmap>
#include <QMimeSource>
#include <Q3MimeSourceFactory>
#include <Q3ImageDrag>
#include <Q3ListViewItem>
typedef Q3ListViewItem QListViewItem;
typedef Q3DragObject QDragObject;
typedef Q3TextDrag QTextDrag;
typedef Q3ImageDrag QImageDrag;
typedef Q3MimeSourceFactory QMimeSourceFactory;
#endif

static QPixmap uic_load_pixmap( const QString &name )
{
   const QMimeSource *m = QMimeSourceFactory::defaultFactory()->data( name );
   if ( !m )
      return QPixmap();
   QPixmap pix;
   QImageDrag::decode( m, pix );
   return pix;
}

void qtrootexample1::init()
{
   TKey *key;
   TFile *fxDiskFile;

   (TQRootCanvas1->GetCanvas())->Divide(2,2);
   fxDiskFile = new TFile("test.root");
   TIter next(fxDiskFile->GetListOfKeys());
   while ((key = (TKey*) next())) {
      AddItemToListView1(key->ReadObj());
   }
}

void qtrootexample1::destroy()
{

}

void qtrootexample1::ListView1_mouseButtonPressed( int, QListViewItem *SelectedItem, const QPoint &, int )
{
   if (SelectedItem!=0) {
      QDragObject *d = new QTextDrag(SelectedItem->text(0),ListView1);
      d->dragCopy();
   }
}

void qtrootexample1::AddItemToListView1(TObject *Key)
{
   if( Key->IsA()->InheritsFrom("TH2") ) {
      QListViewItem * item1 = new QListViewItem(ListView1, Key->GetName() ,"TH2");
      item1->setPixmap( 0, uic_load_pixmap( "h2_t.png" ) );
   }
   else if (Key->IsA()->InheritsFrom("TH1")) {
      QListViewItem * item1 = new QListViewItem(ListView1, Key->GetName(),"TH1");
      item1->setPixmap( 0, uic_load_pixmap( "h1_t.png" ) );
   }
}

