#include "JSONInterface.h"

using RooFit::Detail::JSONNode;

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}
