#include <stdio.h>
#include <TClassEdit.h>

namespace A1 { namespace B2 { namespace C3 { typedef int what; } } }

int execResolveTypedef()
{
   printf("The following should be 'int': %s\n",TClassEdit::ResolveTypedef("A1::B2::C3::what").c_str());
   return 0;
}
