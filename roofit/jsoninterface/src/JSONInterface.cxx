/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/Detail/JSONInterface.h>

#include "JSONParser.h"
#ifdef ROOFIT_WITH_RYML
#include "RYMLParser.h"
#endif

namespace {
template <class Nd>
class ChildItImpl final : public RooFit::Detail::JSONNode::child_iterator_t<Nd>::Impl {
public:
   using child_iterator = RooFit::Detail::JSONNode::child_iterator_t<Nd>;
   ChildItImpl(Nd &n, size_t p) : node(n), pos(p) {}
   ChildItImpl(const ChildItImpl &other) : node(other.node), pos(other.pos) {}
   std::unique_ptr<typename child_iterator::Impl> clone() const override
   {
      return std::make_unique<ChildItImpl>(node, pos);
   }
   void forward() override { ++pos; }
   void backward() override { --pos; }
   Nd &current() override { return node.child(pos); }
   bool equal(const typename child_iterator::Impl &other) const override
   {
      auto it = dynamic_cast<const ChildItImpl<Nd> *>(&other);
      return it && &(it->node) == &(this->node) && (it->pos) == this->pos;
   }

private:
   Nd &node;
   size_t pos;
};
} // namespace

namespace RooFit {
namespace Detail {

template class JSONNode::child_iterator_t<JSONNode>;
template class JSONNode::child_iterator_t<const JSONNode>;

JSONNode::children_view JSONNode::children()
{
   return {child_iterator(std::make_unique<::ChildItImpl<JSONNode>>(*this, 0)),
           child_iterator(std::make_unique<::ChildItImpl<JSONNode>>(*this, this->num_children()))};
}
JSONNode::const_children_view JSONNode::children() const
{
   return {const_child_iterator(std::make_unique<::ChildItImpl<const JSONNode>>(*this, 0)),
           const_child_iterator(std::make_unique<::ChildItImpl<const JSONNode>>(*this, this->num_children()))};
}

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}

std::unique_ptr<JSONTree> JSONTree::create()
{
   if (getBackendEnum() == Backend::Ryml) {
#ifdef ROOFIT_WITH_RYML
      return std::make_unique<TRYMLTree>();
#else
      throw std::runtime_error(
         "Requesting JSON tree with rapidyaml backend, but rapidyaml could not be found by ROOT when it was compiled.");
#endif
   }
   return std::make_unique<TJSONTree>();
}

std::unique_ptr<JSONTree> JSONTree::create(std::istream &is)
{
   if (getBackendEnum() == Backend::Ryml) {
#ifdef ROOFIT_WITH_RYML
      return std::make_unique<TRYMLTree>(is);
#else
      throw std::runtime_error(
         "Requesting JSON tree with rapidyaml backend, but rapidyaml could not be found by ROOT when it was compiled.");
#endif
   }
   return std::make_unique<TJSONTree>(is);
}

/// Check if ROOT was compiled with support for a certain JSON backend library.
/// \param[in] name Name of the backend.
bool JSONTree::hasBackend(std::string const &name)
{
   if (name == "rapidyaml") {
#ifdef ROOFIT_WITH_RYML
      return true;
#else
      return false;
#endif
   }
   if (name == "nlohmann-json")
      return true;
   return false;
}

JSONTree::Backend &JSONTree::getBackendEnum()
{
   static Backend backend = Backend::NlohmannJson;
   return backend;
}

/// Returns the name of the library that serves as the backend for the JSON
/// interface, which is either `"nlohmann-json"` or `"rapidyaml"`.
/// \return Backend name as a string.
std::string JSONTree::getBackend()
{
   return getBackendEnum() == Backend::Ryml ? "rapidyaml" : "nlohmann-json";
}

/// Set the library that serves as the backend for the JSON interface. Note that the `"rapidyaml"` backend is only
/// supported if rapidyaml was found on the system when ROOT was compiled. \param[in] name Name of the backend, can be
/// either `"nlohmann-json"` or `"rapidyaml"`.
void JSONTree::setBackend(std::string const &name)
{
   if (name == "rapidyaml")
      getBackendEnum() = Backend::Ryml;
   if (name == "nlohmann-json")
      getBackendEnum() = Backend::NlohmannJson;
}

} // namespace Detail
} // namespace RooFit
