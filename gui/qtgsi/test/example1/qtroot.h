#ifndef QTROOT_H
#define QTROOT_H

#include "Riostream.h"
using namespace std;

#include "qprinter.h"
#include "qstring.h"
#include "qthread.h"
#include "qtabwidget.h"

#if QT_VERSION >= 0x40000
//Added by qt3to4:
#include <QPaintEvent>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QCloseEvent>
#include "q3mainwindow.h"
#else
#include "qmainwindow.h"
typedef QMainWindow Q3MainWindow;
#endif

#include "TObject.h"
#include "TCanvas.h"
#include "TVirtualX.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TH1.h"
#include "TF1.h"
#include "TFormula.h"
#include "TPad.h"
#include "TSystem.h"

#include "TQRootCanvas.h"


class QPrinter;
class QMouseEvent;
class QResizeEvent;
class QPaintEvent;

/**
* 
*           Main Qt&Root Application Window:
*
*           Simple Example how to use a QMainWindow class 
*           embedding a ROOT Canvas attached as its Central Widget.      
*
*
* @short Simple example using 
*        QMainWindow together with an QRootCanvas  
*        
*  
* Services:
* Simple example  
* @li Creates a Menu for file actions (save, browse, close, ect...)  
* @li Creates a Toolbar with actions managed by QAction class
* @li Creates a QRootCanvas and stick it  as central Widget in QMainWindows
* @li Actions ( execute(), clear()) are defined as slots managed by this
*              QMainApplication. This slots functions acts on the ROOT Objects
*               directly ( histograms, ntuples, ect...)
*
* @see QRootCanvas, QMainWindow, QPopupMenu, QAction, QString
*
* @authors Denis Bertini <d.bertini@gsi.de> 
* @version 2.0
*
*/


class ApplicationWindow: public Q3MainWindow
{
    Q_OBJECT
public:
    ApplicationWindow();
    ~ApplicationWindow();

protected:
    void closeEvent( QCloseEvent* ce);
    
protected slots:
      void myclose(){
	   close(); 
      }
   /**
      New incoming functionality documentation
    */   
    void newDoc();
   /**
     IO  Loading:
             @li ascii data files (Qt or ROOT streamers)
             @li Root binary files(ROOT streamer)
             @ etc...
   */  
 
    void load();
   /**
     IO  Loading function:

             @li ascii data files (Qt or ROOT streamers)
             @li Root binary files(ROOT streamer)
             @ etc...
   */  
    void load( const char *fileName );
   /**
      Saving results function:

             @li ascii data files
             @li histograms, ntuples, Objects using
                           @li ROOT based IO (Root format)
                           @li Qt Based  IO (Qt format)?

             @ etc...
   */ 

    void save();
   /**
       saving pictures, in different supported formats
   */ 
    void saveAs();
    /**
       print results in a specified format
    */ 
    void print();
    /**
       online help about Qt&Root Main Application
    */ 
    void about();
    /**
       online help about this Qt based Application
    */ 
    void aboutQt();
    /**
      Main slot action upon ROOT classes
             @li histograms
             @li ntuples
    */   
    void execute();

     /**
      Main slot to reinitialize the objects
            @li histograms
            @li ntuples
     */ 

    void clear_histo(); 
    void closeQtRoot(){
      qDebug("QtRoot:  closing qt wins \n");
      close(); 
      qDebug("QtRoot:  exiting root \n");
      gSystem->Exit(0); 
    }

private:
    QString filename;
    QPrinter *printer;
    TQRootCanvas *aCanvas,*aCanvas2;     
    TH1F* histo; 
    TF1 *sqroot; 
    TFormula *form1; 
    TPad* pad1,*pad2; 
    QTabWidget* tab;
    QWidget* central;
};

#endif
