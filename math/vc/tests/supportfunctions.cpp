/*{{{
    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#include "unittest.h"

void testCompiledImplementation()
{
    VERIFY(Vc::currentImplementationSupported());
}

void testIsSupported()
{
    using Vc::CpuId;
    VERIFY(Vc::isImplementationSupported(Vc::ScalarImpl));
    COMPARE(Vc::isImplementationSupported(Vc::SSE2Impl ), CpuId::hasSse2());
    COMPARE(Vc::isImplementationSupported(Vc::SSE3Impl ), CpuId::hasSse3());
    COMPARE(Vc::isImplementationSupported(Vc::SSSE3Impl), CpuId::hasSsse3());
    COMPARE(Vc::isImplementationSupported(Vc::SSE41Impl), CpuId::hasSse41());
    COMPARE(Vc::isImplementationSupported(Vc::SSE42Impl), CpuId::hasSse42());
    COMPARE(Vc::isImplementationSupported(Vc::AVXImpl  ), CpuId::hasOsxsave() && CpuId::hasAvx());
    COMPARE(Vc::isImplementationSupported(Vc::AVX2Impl ), false);
}

void testBestImplementation()
{
    // when building with a recent and fully featured compiler the following should pass
    // but - old GCC versions have to fall back to Scalar, even though SSE is supported by the CPU
    //     - ICC/MSVC can't use XOP/FMA4
    //COMPARE(Vc::bestImplementationSupported(), VC_IMPL);
}

void testExtraInstructions()
{
    using Vc::CpuId;
    unsigned int extra = Vc::extraInstructionsSupported();
    COMPARE(!(extra & Vc::Float16cInstructions), !CpuId::hasF16c());
    COMPARE(!(extra & Vc::XopInstructions), !CpuId::hasXop());
    COMPARE(!(extra & Vc::Fma4Instructions), !CpuId::hasFma4());
    COMPARE(!(extra & Vc::PopcntInstructions), !CpuId::hasPopcnt());
    COMPARE(!(extra & Vc::Sse4aInstructions), !CpuId::hasSse4a());
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

    runTest(testCompiledImplementation);
    runTest(testIsSupported);
    runTest(testBestImplementation);
    runTest(testExtraInstructions);

    return 0;
}
