/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef TEST8_H
#define TEST8_H

#ifdef __CINT__

#pragma include <qtcint.dll>

/////////////////////////////////////////////////////////////
class LCDRange : public QVBox
{
 public:
    LCDRange( QWidget *parent=0, const char *name=0 );
    int value() const;
    void setValue( int );
    void setRange( int minVal, int maxVal );
  protected:
    void valueChanged( int );
  private:
    LCDRange(const LCDRange& x) { }
};

#pragma link off function LCDRange::operator=(const LCDRange&);


/////////////////////////////////////////////////////////////
class CannonField : public QWidget
{
public:
    CannonField( QWidget *parent=0, const char *name=0 );

    int angle() const { return ang; }
    QSizePolicy sizePolicy() const;

    void setAngle( int degrees );

protected:
    void angleChanged( int );

protected:
    void paintEvent( QPaintEvent * );

private:
    int ang;
    CannonField(const CannonField& x) { }
};

/////////////////////////////////////////////////////////////

#else

#include "lcdrange.h"
#include "cannon.h"

#endif


#endif
