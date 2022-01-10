#include "RooFitHS3/JSONInterface.h"

using RooFit::Detail::JSONNode;

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}

template <>
int JSONNode::val_t<int>() const
{
   return val_int();
}
template <>
float JSONNode::val_t<float>() const
{
   return val_float();
}
template <>
double JSONNode::val_t<double>() const
{
   return val_float();
}
template <>
bool JSONNode::val_t<bool>() const
{
   return val_bool();
}
template <>
std::string JSONNode::val_t<std::string>() const
{
   return val();
}
