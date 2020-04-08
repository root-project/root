#include <RooFitHS3/JSONInterface.h>

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}
