/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
**
** Definition of CannonField class, Qt tutorial 8
**
****************************************************************/

#ifndef CANNON_H
#define CANNON_H

#include <qwidget.h>


class CannonField : public QWidget
{
    Q_OBJECT
public:
    CannonField( QWidget *parent=0, const char *name=0 );

    int angle() const { return ang; }
    QSizePolicy sizePolicy() const;

public slots:
    void setAngle( int degrees );

signals:
    void angleChanged( int );

protected:
    void paintEvent( QPaintEvent * );

private:
    int ang;
};


#endif // CANNON_H
