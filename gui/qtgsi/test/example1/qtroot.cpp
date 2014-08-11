/****************************************************************************
** Copyright ( C ) 2000 denis Bertini.  All rights reserved.
*****************************************************************************/

#include "qtroot.h"

#if QT_VERSION >= 0x40000
//Added by qt3to4:
#include <QCloseEvent>
#include "q3toolbar.h"
#include "q3filedialog.h"
#include "q3strlist.h"
#include "q3popupmenu.h"
#include "q3intdict.h"
#include "q3action.h"
#include "q3toolbar.h"
#include "qnamespace.h"
#include "q3filedialog.h"
#include "q3whatsthis.h"
#include "Q3MimeSourceFactory"
#else
#include "qfiledialog.h"
#include "qstrlist.h"
#include "qpopupmenu.h"
#include "qintdict.h"
#include "qaction.h"
#include "qtoolbar.h"
#include "qnamespace.h"
#include "qfiledialog.h"
#include "qwhatsthis.h"
#include "qkeycode.h"
typedef QToolBar Q3ToolBar;
typedef QPopupMenu Q3PopupMenu;
typedef QAction Q3Action;
typedef QWhatsThis Q3WhatsThis;
typedef QFileDialog Q3FileDialog;
typedef QMimeSourceFactory Q3MimeSourceFactory;
#endif

#include "stdlib.h"
#include "qevent.h"
#include "qpainter.h"
#include "qprinter.h"
#include "qtoolbutton.h"
#include "qspinbox.h"
#include "qtooltip.h"
#include "qrect.h"
#include "qpoint.h"
#include "qcolordialog.h"
#include "qcursor.h"
#include "qimage.h"
#include "qpushbutton.h"
#include "Riostream.h"
using namespace std;
#include "qdial.h"
#include "qapplication.h"
#include "qimage.h"
#include "qpixmap.h"
#include "qtoolbutton.h"
#include "qmenubar.h"
#include "qfile.h"
#include "qstatusbar.h"
#include "qmessagebox.h"
#include "qdialog.h"
#include "qlabel.h"


#include "filesave.xpm"
#include "fileopen.xpm"
#include "fileprint.xpm"
#include "qtbuttonsupdate.xpm"
#include "qtbuttonsclear.xpm"

#include "TPad.h"
#include "TList.h"
#include "TObject.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TString.h"
#include "TH1.h"
#include "TList.h"
#include "TIterator.h"
#include "TMethod.h"
#include "TCanvas.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TPad.h"
#include "TObjArray.h"
#include "TIterator.h"
#include "TRandom.h"
#include "TFrame.h"
#include "TGraph.h"
#include "TMath.h"

// global menus

const char * fileOpenText = "<img source=\"fileopen\"> "
"Click this button to open a <em>new file</em>. <br><br>"
"You can also select the <b>Open command</b> from the File menu.";
const char * fileSaveText = "Click this button to save the file you are "
"editing.  You will be prompted for a file name.\n\n"
"You can also select the Save command from the File menu.\n\n"
"Note that implementing this function is left as an exercise for the reader.";
const char * filePrintText = "Click this button to print the file you "
"are editing.\n\n"
"You can also select the Print command from the File menu.";
const char* updateHisto = " update histograms ";
const char* clearHisto = " clear histograms ";


//---------------------------------------------------------------------------
//            Qt &Root user Application window example
//---------------------------------------------------------------------------


ApplicationWindow::ApplicationWindow()
    : Q3MainWindow( 0, "example application main window", Qt::WDestructiveClose )
{
   // create a printer
   printer = new QPrinter;
   // create user interface actions

   Q3Action *fileNewAction = new Q3Action( "New", "&New", Qt::CTRL+Qt::Key_N, this, "new" );

   connect( fileNewAction, SIGNAL( activated() ) , this, SLOT( newDoc() ) );

   Q3Action *fileOpenAction = new Q3Action( "Open File", QPixmap( fileopen ), "&Open", Qt::CTRL+Qt::Key_O, this, "open" );
   connect( fileOpenAction, SIGNAL( activated() ) , this, SLOT( load() ) );
   Q3MimeSourceFactory::defaultFactory()->setPixmap( "fileopen", QPixmap( fileopen ) );
   fileOpenAction->setWhatsThis( fileOpenText );

   Q3Action *fileSaveAction = new Q3Action( "Save File", QPixmap( filesave ), "&Save", Qt::CTRL+Qt::Key_S, this, "save" );
   connect( fileSaveAction, SIGNAL( activated() ) , this, SLOT( save() ) );
   fileSaveAction->setWhatsThis( fileSaveText );

   Q3Action *fileSaveAsAction = new Q3Action( "Save File As", "Save &as", 0,  this, "save as" );
   connect( fileSaveAsAction, SIGNAL( activated() ) , this, SLOT( saveAs() ) );
   fileSaveAsAction->setWhatsThis( fileSaveText );

   Q3Action *filePrintAction = new Q3Action( "Print File", QPixmap( fileprint ), "&Print", Qt::CTRL+Qt::Key_P, this, "print" );
   connect( filePrintAction, SIGNAL( activated() ) , this, SLOT( print() ) );
   filePrintAction->setWhatsThis( filePrintText );

   Q3Action *fileCloseAction = new Q3Action( "Close", "&Close", Qt::CTRL+Qt::Key_W, this, "close" );
   connect( fileCloseAction, SIGNAL( activated() ) , this, SLOT( close() ) );

   Q3Action *fileQuitAction = new Q3Action( "Quit", "&Quit", Qt::CTRL+Qt::Key_Q, this, "quit" );
   connect( fileQuitAction, SIGNAL( activated() ) , qApp, SLOT( quit() ) );

   // create button for histo handling
   Q3Action *Update_histo = new Q3Action("Update Histo",QPixmap("qtbuttonsupdate.xpm"),"&Update", Qt::CTRL+Qt::Key_0, this, "update");
   connect( Update_histo, SIGNAL( activated() ) , this, SLOT( execute() ) );
   Q3MimeSourceFactory::defaultFactory()->setPixmap( "update", QPixmap("qtbuttonsupdate.xpm" ) );
   Update_histo->setWhatsThis( updateHisto );

   Q3Action *clear_histo = new Q3Action("Clear Histo",QPixmap("qtbuttonsclear.xpm"),"&Clear", Qt::CTRL+Qt::Key_0, this, "clear");   connect( clear_histo, SIGNAL( activated() ) , this, SLOT( clear_histo() ) );
   Q3MimeSourceFactory::defaultFactory()->setPixmap( "clear", QPixmap("qtbuttonsclear.xpm" ) );
   clear_histo->setWhatsThis( clearHisto );

   // populate a tool bar with some actions

   Q3ToolBar* fileTools = new Q3ToolBar( this, "file operations" );
   fileTools->setLabel( tr( "File Operations" ) );
   fileOpenAction->addTo( fileTools );
   fileSaveAction->addTo( fileTools );
   filePrintAction->addTo( fileTools );
   Update_histo->addTo ( fileTools );
   clear_histo->addTo ( fileTools );
   (void)Q3WhatsThis::whatsThisButton( fileTools );
   // popuplate a menu with all actions

   Q3PopupMenu * file = new Q3PopupMenu( this );
   menuBar()->insertItem( "&File", file );
   fileNewAction->addTo( file );
   fileOpenAction->addTo( file );
   fileSaveAction->addTo( file );
   fileSaveAsAction->addTo( file );
   file->insertSeparator();
   filePrintAction->addTo( file );
   file->insertSeparator();
   fileCloseAction->addTo( file );
   fileQuitAction->addTo( file );

   // add a help menu

   Q3PopupMenu * help = new Q3PopupMenu( this );
   menuBar()->insertSeparator();
   menuBar()->insertItem( "&Help", help );
   help->insertItem( "&About", this, SLOT(about()), Qt::Key_F1 );
   help->insertItem( "About &Qt", this, SLOT(aboutQt()) );
   help->insertSeparator();
   help->insertItem( "What's &This", this, SLOT(whatsThis()), Qt::SHIFT+Qt::Key_F1 );

   // create and define the ROOT Canvas central widget
   tab = new QTabWidget(this);
   tab->show();
   setCentralWidget( tab );

   Q3MainWindow *win1 = new Q3MainWindow( 0, "tab1 main window", Qt::WDestructiveClose );
   Q3MainWindow *win2 = new Q3MainWindow( 0, "tab2 main window", Qt::WDestructiveClose );
   aCanvas = new TQRootCanvas(this, win1,"Qt&Root");
   aCanvas2 = new TQRootCanvas(this, win2,"Qt&Root");

   win1->setCentralWidget(aCanvas);
   win2->setCentralWidget(aCanvas2);


   tab->addTab(win1,"page1");
   tab->addTab(win2,"page2");

   win1->show();
   win2->show();

   // with no QTabWidget
   //    aCanvas = new TQRootCanvas(this,"Qt&Root");
   //    setCentralWidget( aCanvas );
   resize( 450, 500 );

   // put here some ROOT Specifics ...
   if (aCanvas->GetCanvas()) {

      aCanvas->GetCanvas()->Resize();
      aCanvas->GetCanvas()->SetFillColor(40);
      aCanvas->GetCanvas()->cd();

      pad1 = new TPad("pad1","The pad with the function",0.05,0.50,0.95,0.95,21);
      pad1->Draw();
      pad1->cd();
      pad1->SetGridx();
      pad1->SetGridy();
      pad1->GetFrame()->SetFillColor(42);
      pad1->GetFrame()->SetBorderMode(-1);
      pad1->GetFrame()->SetBorderSize(5);

      histo= new TH1F("hppx","Gaussian distribution",100,-4,4);
      histo->SetFillColor(0);
      histo->Draw();

      aCanvas->GetCanvas()->cd();
      pad2 = new TPad("pad2","The pad with the histogram",0.05,0.05,0.95,0.45,21);
      pad2->Draw();
      pad2->cd();

      form1 = new TFormula("form1","abs(sin(x)/x)");
      sqroot = new TF1("sqroot","x*gaus(0) + [3]*form1",0,10);
      sqroot->SetParameters(10,4,1,20);
      sqroot->SetLineColor(4);
      sqroot->SetLineWidth(6);
      sqroot->Draw();
   } // ! aCAnvas

   if (aCanvas2) {

      TCanvas *c1 = aCanvas2->GetCanvas();
      c1->Resize();
      c1->SetFillColor(42);
      c1->GetFrame()->SetFillColor(21);
      c1->GetFrame()->SetBorderSize(12);
      c1->cd();

      //graph example
      const Int_t n = 20;
      Double_t x[n], y[n];
      for (Int_t i=0;i<n;i++) {
         x[i] = i*0.1;
         y[i] = 10*TMath::Sin(x[i]+0.2);
         //  printf(" i %i %f %f \n",i,x[i],y[i]);
      }
      TGraph* gr = new TGraph(n,x,y);
      gr->SetLineColor(2);
      gr->SetLineWidth(4);
      gr->SetMarkerColor(4);
      gr->SetMarkerStyle(21);
      gr->SetTitle("a simple graph");
      gr->Draw("ACP");

      gr->GetHistogram()->SetXTitle("X title");
      gr->GetHistogram()->SetYTitle("Y title");
      c1->Modified();
      c1->Update();
    } //!aCanvas2
}

ApplicationWindow::~ApplicationWindow()
{
   qDebug(" ~ApplicationWindow() \n");
   if (aCanvas)  delete aCanvas;
   if (printer) delete printer;
   if (histo) delete histo;
}

void ApplicationWindow::clear_histo()
{
   // clear histo and update
   if (histo) {
      histo->Reset();
      histo->Draw();
      gROOT->GetSelectedPad()->Modified();
      gROOT->GetSelectedPad()->Update();
  }
}

void ApplicationWindow::execute()
{
   //fill histograms and update for monitoring
   if ( histo ) {
      aCanvas->GetCanvas()->cd();
      pad1->cd();
      float px,py;
      const int kUPDATE = 1000;
      //internal event loop
      for (Int_t i = 0; i < 25000; i++) {
         gRandom->Rannor(px,py);
         float random = gRandom->Rndm(1);
         histo->Fill(px);
         if (i && (i%kUPDATE) == 0) {
            if (i == kUPDATE) histo->Draw("same");
            aCanvas->GetCanvas()->Modified();
            aCanvas->GetCanvas()->Update();
         }
      }
      histo->Draw("same");
      aCanvas->GetCanvas()->Modified();
      aCanvas->GetCanvas()->Update();
   }
}

void ApplicationWindow::newDoc()
{
   ApplicationWindow *ed = new ApplicationWindow;
   ed->show();
}

void ApplicationWindow::load()
{
   QString fn = Q3FileDialog::getOpenFileName(QString::null, QString::null, this);
   if ( !fn.isEmpty() )
      load( fn );
   else
      statusBar()->message( "Loading aborted", 2000 );
}

void ApplicationWindow::load( const char *fileName )
{
}

void ApplicationWindow::save()
{
}

void ApplicationWindow::saveAs()
{
}

void ApplicationWindow::print()
{
}

void ApplicationWindow::closeEvent( QCloseEvent* ce )
{
   int testvar =  QMessageBox::information( 0, "Qt Application Example",
                                            "Do you want to close QtRoot? "
                                            "", "Save", "Cancel", "Close",
                                            0, 1 );
   switch (testvar) {
      case 0: // here we should save
         // data
         save();
         ce->accept();
         break;
      case 1:
      default: // just for sanity
         ce->ignore();
         break;
      case 2: // Here i close all windows
         // do now an explicit release of Histogram's
         // child windows
         TList *lc = (TList*)gROOT->GetListOfCanvases();
         TObject *fitpanel  =  lc->FindObject("R__fitpanel");
         TObject *drawpanel =  lc->FindObject("R__drawpanelhist");
         if (fitpanel) {
            qDebug("detecting fitpanel %x \n",fitpanel);
            delete fitpanel;
         }
         if (drawpanel) {
            qDebug("detecting drawpanel %x \n",drawpanel);
            delete drawpanel;
         }
         ce->accept();
         break;
   }
}

void ApplicationWindow::about()
{
   QMessageBox::about( this, "Qt&ROOT Application Example",
                       "This example demonstrates simple use of "
                       "QMainWindow,\nQMenuBar and QToolBar.");
}

void ApplicationWindow::aboutQt()
{
   QMessageBox::aboutQt( this, "Qt Application Example" );
}
