/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
**
** Qt tutorial 7
**
****************************************************************/

#ifdef __CINT__

#pragma include "test7.dll"
#pragma include "qtcint.dll"

#else

#include <qapplication.h>
#include <qpushbutton.h>
#include <qlcdnumber.h>
#include <qfont.h>
#include <qvbox.h>
#include <qgrid.h>

#include "lcdrange.h"

#endif

class MyWidget : public QVBox
{
public:
    MyWidget( QWidget *parent=0, const char *name=0 );
};


MyWidget::MyWidget( QWidget *parent, const char *name )
        : QVBox( parent, name )
{
    QPushButton *quit = new QPushButton( "Quit", this, "quit" );
    quit->setFont( QFont( "Times", 18, QFont::Bold ) );

    connect( quit, SIGNAL(clicked()), qApp, SLOT(quit()) );

    QGrid *grid = new QGrid( 4, this );

    LCDRange *previous = 0;
    for( int r = 0 ; r < 4 ; r++ ) {
	for( int c = 0 ; c < 4 ; c++ ) {
	    LCDRange* lr = new LCDRange( grid );
	    if ( previous ) {
#ifdef __CINT__
		connect( lr, "2valueChanged(int)",
			 previous, "1setValue(int)" );
#else
		connect( lr, SIGNAL(valueChanged(int)),
			 previous, SLOT(setValue(int)) );
#endif
            }
	    previous = lr;
	}
    }
}


int main( int argc, char **argv )
{
    QApplication a( argc, argv );

#ifdef __CINT__
    //Cint workaround, don't know why linker can not find qApp 
    //Set a dummy version declared in qtstatic.cxx
    qApp = &a; 
#endif

    MyWidget w;
    a.setMainWidget( &w );
    w.show();
    return a.exec();
}
