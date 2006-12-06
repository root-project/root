/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef TEST7_H
#define TEST7_H

#ifdef __CINT__

#pragma include <qtcint.dll>

class LCDRange : public QVBox
{
 public:
    LCDRange( QWidget *parent=0, const char *name=0 );
    int value() const;
    void setValue( int );
  protected:
    void valueChanged( int );
  private:
    LCDRange(const LCDRange& x) { }
};

#pragma link off function LCDRange::operator=(const LCDRange&);

#else

#include "lcdrange.h"

#endif


#endif
