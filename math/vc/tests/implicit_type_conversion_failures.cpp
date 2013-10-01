#include <Vc/vector.h>

#if !defined(TYPE_A) || !defined(TEST_OP) || !defined(TYPE_B)
#error "Need to define TYPE_A, TEST_OP, and TYPE_B"
#endif

using namespace Vc;

int main()
{
    TYPE_A() TEST_OP TYPE_B();
    return 0;
}
