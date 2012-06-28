/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
**
** Qt tutorial 8
**
****************************************************************/

#ifdef __CINT__

#pragma include "test8.dll"
#pragma include "qtcint.dll"

#else

#include <qapplication.h>
#include <qpushbutton.h>
#include <qlcdnumber.h>
#include <qfont.h>
#include <qlayout.h>

#include "lcdrange.h"
#include "cannon.h"

#endif

class MyWidget: public QWidget
{
public:
    MyWidget( QWidget *parent=0, const char *name=0 );
};


MyWidget::MyWidget( QWidget *parent, const char *name )
        : QWidget( parent, name )
{
    QPushButton *quit = new QPushButton( "Quit", this, "quit" );
    quit->setFont( QFont( "Times", 18, QFont::Bold ) );

    connect( quit, SIGNAL(clicked()), qApp, SLOT(quit()) );

    LCDRange *angle = new LCDRange( this, "angle" );
    angle->setRange( 5, 70 );

    CannonField *cannonField 
	= new CannonField( this, "cannonField" );

#ifdef __CINT__
    connect( angle, "2valueChanged(int)",
	     cannonField, "1setAngle(int)" );
    connect( cannonField, "2angleChanged(int)",
	     angle, "1setValue(int)" );
#else
    connect( angle, SIGNAL(valueChanged(int)),
	     cannonField, SLOT(setAngle(int)) );
    connect( cannonField, SIGNAL(angleChanged(int)),
	     angle, SLOT(setValue(int)) );
#endif

    QGridLayout *grid = new QGridLayout( this, 2, 2, 10 );
    //2x2, 10 pixel border

    grid->addWidget( quit, 0, 0 );
    grid->addWidget( angle, 1, 0, Qt::AlignTop );
    grid->addWidget( cannonField, 1, 1 );
    grid->setColStretch( 1, 10 );

    angle->setValue( 60 );
    angle->setFocus();
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
    w.setGeometry( 100, 100, 500, 355 );
    a.setMainWidget( &w );
    w.show();
    return a.exec();
}
