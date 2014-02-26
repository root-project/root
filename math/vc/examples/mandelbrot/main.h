/*
    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

*/

#ifndef MAIN_H
#define MAIN_H

#include <QImage>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QWidget>

#include "mandel.h"

class MainWindow : public QWidget
{
    Q_OBJECT
    public:
        MainWindow(QWidget *parent = 0);

    protected:
        void paintEvent(QPaintEvent *);
        void resizeEvent(QResizeEvent *);
        void mousePressEvent(QMouseEvent *);
        void mouseMoveEvent(QMouseEvent *);
        void mouseReleaseEvent(QMouseEvent *);
        void wheelEvent(QWheelEvent *);

    private slots:
        void vcImage(const QImage &, quint64);
        void scalarImage(const QImage &, quint64);

    private:
        void recreateImage();
        void updateTitle();

        float m_x; // left
        float m_y; // top
        float m_scale;
        QImage m_img1;
        QImage m_img2;
        QRect m_rect1;
        QRect m_rect2;
        QPoint m_dragStart;
        QPoint m_dragDelta;

        float m_cycles1, m_cycles2;

        Mandel<VcImpl> m_mandelVc;
        Mandel<ScalarImpl> m_mandelScalar;
};
#endif // MAIN_H
