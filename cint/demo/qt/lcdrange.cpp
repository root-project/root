/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
**
** Implementation of LCDRange class, Qt tutorial 8
**
****************************************************************/

#include "lcdrange.h"

#include <qslider.h>
#include <qlcdnumber.h>

LCDRange::LCDRange( QWidget *parent, const char *name )
        : QVBox( parent, name )
{
    QLCDNumber *lcd  = new QLCDNumber( 2, this, "lcd"  );
    slider = new QSlider( Horizontal, this, "slider" );
    slider->setRange( 0, 99 );
    slider->setValue( 0 );
    connect( slider, SIGNAL(valueChanged(int)),
	     lcd, SLOT(display(int)) );
    connect( slider, SIGNAL(valueChanged(int)),
	     SIGNAL(valueChanged(int)) );

    setFocusProxy( slider );
}

int LCDRange::value() const
{
    return slider->value();
}

void LCDRange::setValue( int value )
{
    slider->setValue( value );
}

void LCDRange::setRange( int minVal, int maxVal )
{
    if ( minVal < 0 || maxVal > 99 || minVal > maxVal ) {
      qWarning( "LCDRange::setRange(%d,%d)\n"
	       "\tRange must be 0..99\n"
	       "\tand minVal must not be greater than maxVal",
	       minVal, maxVal );
      return;
    }
    slider->setRange( minVal, maxVal );
}
