#include <stdio.h>
#include <TClassEdit.h>

namespace A1 { namespace B2 { namespace C3 { typedef int what; } } }
namespace NS { typedef int IntNS_t; }

int execResolveTypedef()
{
   printf("The following should be 'int': %s\n",TClassEdit::ResolveTypedef("A1::B2::C3::what").c_str());
   printf("The following should be 'const int': %s\n",TClassEdit::ResolveTypedef("const NS::IntNS_t").c_str());
   return 0;
}
