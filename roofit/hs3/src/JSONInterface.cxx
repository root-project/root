#include <RooFitHS3/JSONInterface.h>

std::ostream &operator<<(std::ostream &os, RooFit::Experimental::JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}

template <>
int RooFit::Experimental::JSONNode::val_t<int>() const
{
   return val_int();
}
template <>
float RooFit::Experimental::JSONNode::val_t<float>() const
{
   return val_float();
}
template <>
double RooFit::Experimental::JSONNode::val_t<double>() const
{
   return val_float();
}
template <>
bool RooFit::Experimental::JSONNode::val_t<bool>() const
{
   return val_bool();
}
template <>
std::string RooFit::Experimental::JSONNode::val_t<std::string>() const
{
   return val();
}
