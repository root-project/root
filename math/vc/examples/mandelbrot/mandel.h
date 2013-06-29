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

#include <QImage>
#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>
#include <complex>

enum MandelImpl {
    VcImpl, ScalarImpl
};

class MandelBase : public QThread
{
    Q_OBJECT
    public:
        void brot(const QSize &size, float x, float y, float scale);

    protected:
        MandelBase(QObject* _parent = 0);
        ~MandelBase();
        void emitImage(const QImage &image, quint64 cycles) { emit ready(image, cycles); }

        void run();
        virtual void mandelMe(QImage &image, float x, float y, float scale, int maxIterations) = 0;
        inline bool restart() const { return m_restart; }

    signals:
        void ready(const QImage &image, quint64 cycles);

    private:
        QMutex m_mutex;
        QWaitCondition m_wait;
        QSize m_size;
        float m_x, m_y, m_scale;
        bool m_restart;
        bool m_abort;
};

template<MandelImpl Impl>
class Mandel : public MandelBase
{
    public:
        Mandel(QObject *_parent = 0);

    protected:
        void mandelMe(QImage &image, float x, float y, float scale, int maxIterations);
};

