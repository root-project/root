#include <Vc/global.h>
#if !(defined VC_GCC && VC_GCC < 0x40400) && !defined VC_MSVC
#include <x86intrin.h>
#endif
#include <Vc/Vc>
#include <Vc/IO>
#include <Vc/support.h>

using namespace Vc;
float_v foo0(float_v::AsArg a)
{
    const float_v b = sin(a + float_v::One());
    std::cerr << b;
    return b;
}
