#include <Vc/Vc>
#include <Vc/IO>

#define CAT(a, b) a##b
#define name(a, b) CAT(a, b)

using namespace Vc;
float_v
#ifdef VC_MSVC
__declspec(dllexport)
#endif
name(fooLib0, POSTFIX)(float_v::AsArg a)
{
    const float_v b = sin(a + float_v::One());
    std::cerr << b;
    return b;
}
