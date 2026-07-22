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

#include <sstream>

namespace {
template <class Node_t>
class ChildItImpl final : public RooFit::Detail::JSONNode::child_iterator_t<Node_t>::Impl {
public:
   using child_iterator = RooFit::Detail::JSONNode::child_iterator_t<Node_t>;
   ChildItImpl(Node_t &n, size_t p) : node(n), pos(p) {}
   ChildItImpl(const ChildItImpl &other) : node(other.node), pos(other.pos) {}
   std::unique_ptr<typename child_iterator::Impl> clone() const override
   {
      return std::make_unique<ChildItImpl>(node, pos);
   }
   void forward() override { ++pos; }
   void backward() override { --pos; }
   Node_t &current() override { return node.child(pos); }
   bool equal(const typename child_iterator::Impl &other) const override
   {
      auto it = dynamic_cast<const ChildItImpl<Node_t> *>(&other);
      return it && &(it->node) == &(this->node) && (it->pos) == this->pos;
   }

private:
   Node_t &node;
   size_t pos;
};
} // namespace

namespace RooFit {
namespace Detail {

template class JSONNode::child_iterator_t<JSONNode>;
template class JSONNode::child_iterator_t<const JSONNode>;

double JSONNode::val_double() const
{
   double out;
   std::stringstream ss{val()};
   ss >> out;
   return out;
}

// Default fallback for backends that don't provide native type introspection:
// a node is considered numeric if its textual value parses completely as a
// floating-point number. Containers, null and non-numeric scalars (strings,
// booleans) are rejected.
bool JSONNode::is_number() const
{
   if (is_container() || is_null()) {
      return false;
   }
   const std::string text = val();
   if (text.empty()) {
      return false;
   }
   try {
      std::size_t consumed = 0;
      std::stod(text, &consumed);
      return consumed == text.size();
   } catch (...) {
      return false;
   }
}

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

template <typename... Args>
std::unique_ptr<JSONTree> JSONTree::createImpl(Args &&...args)
{
   return std::make_unique<TJSONTree>(std::forward<Args>(args)...);
}

std::unique_ptr<JSONTree> JSONTree::create()
{
   return createImpl();
}

std::unique_ptr<JSONTree> JSONTree::create(std::istream &is)
{
   return createImpl(is);
}

std::unique_ptr<JSONTree> JSONTree::create(std::string const &str)
{
   std::stringstream ss{str};
   return JSONTree::create(ss);
}

} // namespace Detail
} // namespace RooFit
