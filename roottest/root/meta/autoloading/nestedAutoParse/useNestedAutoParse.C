#include <vector>

int useNestedAutoParse()
{
   // Payload is only forward declared here. Instantiating the members of
   // std::vector<Payload> needs the complete type, so its payload is
   // autoparsed from within a template instantiation.
   std::vector<NestedAutoParse::Inner::Payload> v(1);
   return v[0].get();
}
