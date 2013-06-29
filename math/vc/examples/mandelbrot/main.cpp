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

#include "main.h"

#include <QApplication>
#include <QPainter>
//#include <QProgressBar>

MainWindow::MainWindow(QWidget *_parent)
    : QWidget(_parent),
    m_scale(0.01f)
{
    m_x = width() * m_scale * -0.667f;
    m_y = height() * m_scale * -0.5f;

    m_rect1 = m_rect2 = rect();
    m_rect1.setWidth(m_rect1.width() / 2);
    m_rect2.setX(m_rect1.width());

    qRegisterMetaType<QImage>();
    qRegisterMetaType<quint64>();
    connect(&m_mandelVc, SIGNAL(ready(QImage, quint64)), SLOT(vcImage(QImage, quint64)));
    connect(&m_mandelScalar, SIGNAL(ready(QImage, quint64)), SLOT(scalarImage(QImage, quint64)));

    setWindowTitle(tr("Mandelbrot"));
    setCursor(Qt::CrossCursor);
}

void MainWindow::vcImage(const QImage &img, quint64 cycles)
{
    m_img1 = img;
    update(m_rect1);
    if (cycles > 1) {
        m_cycles1 = cycles;
        updateTitle();
    }
    if (QCoreApplication::arguments().contains("--benchmark")) {
        m_mandelScalar.brot(m_rect2.size(), m_x, m_y, m_scale);
    }
}

void MainWindow::scalarImage(const QImage &img, quint64 cycles)
{
    m_img2 = img;
    update(m_rect2);
    if (cycles > 1) {
        m_cycles2 = cycles;
        updateTitle();
    }
}

void MainWindow::updateTitle()
{
    setWindowTitle(tr("Mandelbrot [Speedup: %1] [%2]").arg(m_cycles2 / m_cycles1).arg(m_img1 == m_img2 ? "Equal" : "Not Equal"));
}

void MainWindow::paintEvent(QPaintEvent *e)
{
    QPainter p(this);
    QRect r1 = m_rect1 & e->rect();
    p.drawImage(r1, m_img1, r1.translated(m_dragDelta));
    QRect r2 = m_rect2 & e->rect();
    p.drawImage(r2, m_img2, QRect(QPoint(), r2.size()).translated(m_dragDelta));
}

void MainWindow::mousePressEvent(QMouseEvent *e)
{
    m_dragStart = e->pos();
}

void MainWindow::mouseMoveEvent(QMouseEvent *e)
{
    m_dragDelta = m_dragStart - e->pos();
    update();
}

void MainWindow::mouseReleaseEvent(QMouseEvent *e)
{
    m_dragDelta = m_dragStart - e->pos();
    // translate m_x, m_y accordingly and recreate the image
    m_x += m_dragDelta.x() * m_scale;
    m_y += m_dragDelta.y() * m_scale;
    recreateImage();
    m_dragDelta = QPoint();
}

void MainWindow::wheelEvent(QWheelEvent *e)
{
    if (e->delta() < 0 && width() * m_scale > 3.f && height() * m_scale > 2.f) {
        return;
    }
    const float xx = e->x() >= m_rect1.width() ? e->x() - m_rect1.width() : e->x();
    const float constX = m_x + m_scale * xx;
    const float constY = m_y + m_scale * e->y();
    if (e->delta() > 0) {
        m_scale *= 1.f / (1.f + e->delta() * 0.001f);
    } else {
        m_scale *= 1.f - e->delta() * 0.001f;
    }
    m_x = constX - m_scale * xx;
    m_y = constY - m_scale * e->y();
    recreateImage();
    //update();
}

void MainWindow::resizeEvent(QResizeEvent *e)
{
    if (e->oldSize().isValid()) {
        m_x += 0.25f * m_scale * (e->oldSize().width() - e->size().width());
        m_y += 0.5f  * m_scale * (e->oldSize().height() - e->size().height());
    } else {
        m_x = e->size().width() * m_scale * -0.333f;
        m_y = e->size().height() * m_scale * -0.5f;
    }

    m_rect1 = m_rect2 = QRect(QPoint(), e->size());
    m_rect1.setWidth(m_rect1.width() / 2);
    m_rect2.setX(m_rect1.width());

    recreateImage();
    update();
}

void MainWindow::recreateImage()
{
    if (!QCoreApplication::arguments().contains("--benchmark")) {
        m_mandelScalar.brot(m_rect2.size(), m_x, m_y, m_scale);
    }
    m_mandelVc.brot(m_rect1.size(), m_x, m_y, m_scale);
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    MainWindow w;
    w.resize(600, 200);
    w.show();
    return app.exec();
}
