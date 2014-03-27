/*
    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#include "mandel.h"
#include <QMutexLocker>
#include <QtCore/QtDebug>
#include "../tsc.h"

#include <Vc/vector.h>
#include <Vc/common/macros.h>

using Vc::float_v;
using Vc::float_m;
using Vc::uint_v;
using Vc::uint_m;

template<MandelImpl Impl>
Mandel<Impl>::Mandel(QObject *_parent)
    : MandelBase(_parent)
{
}

MandelBase::MandelBase(QObject *_parent)
    : QThread(_parent),
    m_restart(false), m_abort(false)
{
}

MandelBase::~MandelBase()
{
    m_mutex.lock();
    m_abort = true;
    m_wait.wakeOne();
    m_mutex.unlock();

    wait();
}

void MandelBase::brot(const QSize &size, float x, float y, float scale)
{
    QMutexLocker lock(&m_mutex);

    m_size = size;
    m_x = x;
    m_y = y;
    m_scale = scale;

    if (!isRunning()) {
        start(LowPriority);
    } else {
        m_restart = true;
        m_wait.wakeOne();
    }
}

void MandelBase::run()
{
    while (!m_abort) {
        // first we copy the parameters to our local data so that the main main thread can give a
        // new task while we're working
        m_mutex.lock();
        // destination image, RGB is good - no need for alpha
        QImage image(m_size, QImage::Format_RGB32);
        float x = m_x;
        float y = m_y;
        float scale = m_scale;
        m_mutex.unlock();

        // benchmark the number of cycles it takes
        TimeStampCounter timer;
        timer.Start();

        // calculate the mandelbrot set/image
        mandelMe(image, x, y, scale, 255);

        timer.Stop();

        // if no new set was requested in the meantime - return the finished image
        if (!m_restart) {
            emit ready(image, timer.Cycles());
        }

        // wait for more work
        m_mutex.lock();
        if (!m_restart) {
            m_wait.wait(&m_mutex);
        }
        m_restart = false;
        m_mutex.unlock();
    }
}

static const float S = 4.f;

/**
 * std::complex is way too slow for our limited purposes:
 *
 * norm is implemented as std::abs(z) * std::abs(z) for float
 * z * z is implemented as multiplication & lots of branches looking for NaN and inf
 *
 * since we know that we require the square of r and i for norm and multiplication we can
 * explicitely cache it in the object
 */
//! [MyComplex]
template<typename T>
class MyComplex
{
    public:
        MyComplex(T r, T i)
            : m_real(r), m_imag(i),
            m_real2(r * r), m_imag2(i * i)
        {
        }

        MyComplex squaredPlus(T r, T i) const
        {
            return MyComplex(
                    m_real2 + r - m_imag2,
                    (m_real + m_real) * m_imag + i
                    );
        }

        T norm() const
        {
            return m_real2 + m_imag2;
        }

    private:
        T m_real, m_imag;
        T m_real2, m_imag2;
};
//! [MyComplex]

//! [P function]
template<typename T> inline MyComplex<T> P(MyComplex<T> z, T c_real, T c_imag)
{
    return z.squaredPlus(c_real, c_imag);
}
//! [P function]

template<> void Mandel<VcImpl>::mandelMe(QImage &image, float x0,
        float y0, float scale, int maxIt)
{
    typedef MyComplex<float_v> Z;
    const unsigned int height = image.height();
    const unsigned int width = image.width();
    const float_v colorScale = 0xff / static_cast<float>(maxIt);
    for (unsigned int y = 0; y < height; ++y) {
        unsigned int *VC_RESTRICT line = reinterpret_cast<unsigned int *>(image.scanLine(y));
        const float_v c_imag = y0 + y * scale;
        uint_m toStore;
        for (uint_v x = uint_v::IndexesFromZero(); !(toStore = x < width).isEmpty();
                x += float_v::Size) {
            const float_v c_real = x0 + x * scale;
            Z z(c_real, c_imag);
            float_v n = 0.f;
            float_m inside = z.norm() < S;
            while (!(inside && n < maxIt).isEmpty()) {
                z = P(z, c_real, c_imag);
                ++n(inside);
                inside = z.norm() < S;
            }
            uint_v colorValue = static_cast<uint_v>((maxIt - n) * colorScale) * 0x10101;
            if (toStore.isFull()) {
                colorValue.store(line, Vc::Unaligned);
                line += uint_v::Size;
            } else {
                colorValue.store(line, toStore, Vc::Unaligned);
                break; // we don't need to check again wether x[0] + float_v::Size < width to break out of the loop
            }
        }
        if (restart()) {
            break;
        }
    }
}

template<> void Mandel<ScalarImpl>::mandelMe(QImage &image, float x0,
        float y0, float scale, int maxIt)
{
    typedef MyComplex<float> Z;
    const int height = image.height();
    const int width = image.width();
    const float colorScale = 0xff / static_cast<float>(maxIt);
    for (int y = 0; y < height; ++y) {
        unsigned int *VC_RESTRICT line = reinterpret_cast<unsigned int *>(image.scanLine(y));
        const float c_imag = y0 + y * scale;
        for (int x = 0; x < width; ++x) {
            const float c_real = x0 + x * scale;
            Z z(c_real, c_imag);
            int n = 0;
            for (; z.norm() < S && n < maxIt; ++n) {
                z = P(z, c_real, c_imag);
            }
            *line++ = static_cast<unsigned int>((maxIt - n) * colorScale) * 0x10101;
        }
        if (restart()) {
            break;
        }
    }
}

template class Mandel<VcImpl>;
template class Mandel<ScalarImpl>;

// vim: sw=4 sts=4 et tw=100
