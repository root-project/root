/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifdef __CINT__

#pragma include <qtcint.dll>

#else

#include <qapplication.h>
#include <qpushbutton.h>
#include <qfont.h>
#include <qvbox.h>

#endif

int main( int argc, char **argv )
{
    QApplication a( argc, argv );

    QVBox box;
    box.resize( 200, 120 );

    QPushButton quit( "Quit", &box );
    quit.setFont( QFont( "Times", 18, QFont::Bold ) );

    QObject::connect( &quit, SIGNAL(clicked()), &a, SLOT(quit()) );

    a.setMainWidget( &box );
    box.show();

    return a.exec();
}
