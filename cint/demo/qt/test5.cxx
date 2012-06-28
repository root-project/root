/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
**
** Qt tutorial 5
**
****************************************************************/

#ifdef __CINT__

#pragma include <qtcint.dll>

#else

#include <qapplication.h>
#include <qpushbutton.h>
#include <qslider.h>
#include <qlcdnumber.h>
#include <qfont.h>

#include <qvbox.h>

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

    QLCDNumber *lcd  = new QLCDNumber( 2, this, "lcd" );

    QSlider * slider = new QSlider( Horizontal, this, "slider" );
    slider->setRange( 0, 99 );
    slider->setValue( 0 );

#ifdef __CINT__
    connect( slider, "2valueChanged(int)", lcd, "1display(int)" );
#else
    connect( slider, SIGNAL(valueChanged(int)), lcd, SLOT(display(int)) );
#endif
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

