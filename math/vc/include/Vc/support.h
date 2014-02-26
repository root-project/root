/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_COMMON_SUPPORT_H
#define VC_COMMON_SUPPORT_H

#ifndef VC_GLOBAL_H
#error "Vc/global.h must be included first!"
#endif

#include <Vc/cpuid.h>

#if defined(VC_GCC) && VC_GCC >= 0x40400
#define VC_TARGET_NO_SIMD __attribute__((target("no-sse2,no-avx")))
#else
#define VC_TARGET_NO_SIMD
#endif

namespace ROOT {
namespace Vc
{

/**
 * \name Micro-Architecture Feature Tests
 */
//@{
/**
 * \ingroup Utilities
 * \headerfile support.h <Vc/support.h>
 * Determines the extra instructions supported by the current CPU.
 *
 * \return A combination of flags from Vc::ExtraInstructions that the current CPU supports.
 */
VC_TARGET_NO_SIMD
unsigned int extraInstructionsSupported();

/**
 * \ingroup Utilities
 * \headerfile support.h <Vc/support.h>
 *
 * Tests whether the given implementation is supported by the system the code is executing on.
 *
 * \return \c true if the OS and hardware support execution of instructions defined by \p impl.
 * \return \c false otherwise
 *
 * \param impl The SIMD target to test for.
 */
VC_TARGET_NO_SIMD
bool isImplementationSupported(Vc::Implementation impl);

/**
 * \internal
 * \ingroup Utilities
 * \headerfile support.h <Vc/support.h>
 *
 * Tests whether the given implementation is supported by the system the code is executing on.
 *
 * \code
 * if (!isImplementationSupported<Vc::CurrentImplementation>()) {
 *   std::cerr << "This code was compiled with features that this system does not support.\n";
 *   return EXIT_FAILURE;
 * }
 * \endcode
 *
 * \return \c true if the OS and hardware support execution of instructions defined by \p impl.
 * \return \c false otherwise
 *
 * \tparam Impl The SIMD target to test for.
 */
template<typename Impl>
VC_TARGET_NO_SIMD
static inline bool isImplementationSupported()
{
    return isImplementationSupported(static_cast<Vc::Implementation>(Impl::Implementation)) &&
        (extraInstructionsSupported() & Impl::ExtraInstructions) == Impl::ExtraInstructions;
}

/**
 * \ingroup Utilities
 * \headerfile support.h <Vc/support.h>
 *
 * Determines the best supported implementation for the current system.
 *
 * \return The enum value for the best implementation.
 */
VC_TARGET_NO_SIMD
Vc::Implementation bestImplementationSupported();

#ifndef VC_COMPILE_LIB
/**
 * \ingroup Utilities
 * \headerfile support.h <Vc/support.h>
 *
 * Tests that the CPU and Operating System support the vector unit which was compiled for. This
 * function should be called before any other Vc functionality is used. It checks whether the program
 * will work. If this function returns \c false then the program should exit with a useful error
 * message before the OS has to kill it because of an invalid instruction exception.
 *
 * If the program continues and makes use of any vector features not supported by
 * hard- or software then the program will crash.
 *
 * Example:
 * \code
 * int main()
 * {
 *   if (!Vc::currentImplementationSupported()) {
 *     std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
 *     exit -1;
 *   }
 *   ...
 * }
 * \endcode
 *
 * \return \c true if the OS and hardware support execution of the currently selected SIMD
 *                 instructions.
 * \return \c false otherwise
 */
VC_TARGET_NO_SIMD
#ifndef DOXYGEN
static
#endif
inline bool currentImplementationSupported()
{
    return isImplementationSupported<Vc::CurrentImplementation>();
}
#endif // VC_COMPILE_LIB
//@}

} // namespace Vc
} // namespace ROOT

#undef VC_TARGET_NO_SIMD

#endif // VC_COMMON_SUPPORT_H
