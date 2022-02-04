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

#include "JSONParser.h"

#include <sstream>

#include "nlohmann/json.hpp"

// TJSONTree interface

TJSONTree::Node &TJSONTree::rootnode()
{
   return root;
}

// TJSONTree methods

TJSONTree::TJSONTree() : root(this){};

TJSONTree::TJSONTree(std::istream &is) : root(this, is){};

TJSONTree::~TJSONTree()
{
   TJSONTree::_nodecache.clear();
};

TJSONTree::Node &TJSONTree::incache(const TJSONTree::Node &n)
{
   _nodecache.push_back(n);
   return _nodecache.back();
}

TJSONTree *TJSONTree::Node::get_tree()
{
   return tree;
}

const TJSONTree::Node::Impl &TJSONTree::Node::get_node() const
{
   return *node;
}

const TJSONTree *TJSONTree::Node::get_tree() const
{
   return tree;
}

TJSONTree::Node::Impl &TJSONTree::Node::get_node()
{
   return *node;
}

void TJSONTree::clearcache()
{
   TJSONTree::_nodecache.clear();
}

// TJSONTree::Node implementation

class TJSONTree::Node::Impl {
public:
   std::string _key;
   std::string const &key() const { return _key; }
   virtual nlohmann::json &get() = 0;
   virtual const nlohmann::json &get() const = 0;
   class BaseNode;
   class NodeRef;
   Impl(const std::string &k) : _key(k) {}
   virtual ~Impl() = default;
   static TJSONTree::Node &mkNode(TJSONTree *t, const std::string &k, nlohmann::json &n);
   static const TJSONTree::Node &mkNode(const TJSONTree *t, const std::string &k, const nlohmann::json &n);
};

class TJSONTree::Node::Impl::BaseNode : public TJSONTree::Node::Impl {
   nlohmann::json node;

public:
   nlohmann::json &get() override { return node; }
   const nlohmann::json &get() const override { return node; }
   BaseNode(std::istream &is) : Impl(""), node(nlohmann::json::parse(is)) {}
   BaseNode() : Impl("") {}
};

class TJSONTree::Node::Impl::NodeRef : public TJSONTree::Node::Impl {
   nlohmann::json &node;

public:
   nlohmann::json &get() override { return node; }
   const nlohmann::json &get() const override { return node; }
   NodeRef(const std::string &k, nlohmann::json &n) : Impl(k), node(n) {}
   NodeRef(const NodeRef &other) : Impl(other.key()), node(other.node) {}
};

TJSONTree::Node &TJSONTree::Node::Impl::mkNode(TJSONTree *t, const std::string &k, nlohmann::json &n)
{
   Node::Impl::NodeRef ref(k, n);
   return t->incache(Node(t, ref));
}

const TJSONTree::Node &TJSONTree::Node::Impl::mkNode(const TJSONTree *t, const std::string &k, const nlohmann::json &n)
{
   // not so nice to use const_cast here, but the non-const version will only live in the cache
   Node::Impl::NodeRef ref(k, const_cast<nlohmann::json &>(n));
   return const_cast<TJSONTree *>(t)->incache(Node(const_cast<TJSONTree *>(t), ref));
}

TJSONTree::Node::Node(TJSONTree *t, std::istream &is) : tree(t), node(std::make_unique<Impl::BaseNode>(is)) {}

TJSONTree::Node::Node(TJSONTree *t) : tree(t), node(std::make_unique<Impl::BaseNode>()) {}

TJSONTree::Node::Node(TJSONTree *t, Impl &other)
   : tree(t), node(std::make_unique<Impl::NodeRef>(other.key(), other.get()))
{
}

TJSONTree::Node::Node(const Node &other) : Node(other.tree, *other.node) {}

TJSONTree::Node::~Node() {}

// TJSONNode interface

void TJSONTree::Node::writeJSON(std::ostream &os) const
{
   os << node->get();
}

TJSONTree::Node &TJSONTree::Node::operator<<(std::string const &s)
{
   node->get() = s;
   return *this;
}

TJSONTree::Node &TJSONTree::Node::operator<<(int i)
{
   node->get() = i;
   return *this;
}

TJSONTree::Node &TJSONTree::Node::operator<<(double d)
{
   node->get() = d;
   return *this;
}

const TJSONTree::Node &TJSONTree::Node::operator>>(std::string &v) const
{
   v = node->get().get<std::string>();
   return *this;
}

TJSONTree::Node &TJSONTree::Node::operator[](std::string const &k)
{
   return Impl::mkNode(tree, k, node->get()[k]);
}

TJSONTree::Node &TJSONTree::Node::operator[](size_t pos)
{
   return Impl::mkNode(tree, "", node->get()[pos]);
}

const TJSONTree::Node &TJSONTree::Node::operator[](std::string const &k) const
{
   return Impl::mkNode(tree, k, node->get()[k]);
}

const TJSONTree::Node &TJSONTree::Node::operator[](size_t pos) const
{
   return Impl::mkNode(tree, "", node->get()[pos]);
}

bool TJSONTree::Node::is_container() const
{
   return node->get().is_array() || node->get().is_object();
}

bool TJSONTree::Node::is_map() const
{
   return node->get().is_object();
}

bool TJSONTree::Node::is_seq() const
{
   return node->get().is_array();
}

namespace {

// To check whether it's allowed to reset the type of an object. We allow
// this for nodes that have no type yet, or nodes with an empty string.
bool isResettingPossible(nlohmann::json const &node)
{

   if (node.type() == nlohmann::json::value_t::null) {
      return true;
   }

   if (node.type() == nlohmann::json::value_t::string) {
      if (node.get<std::string>() == "") {
         return true;
      }
   }
   return false;
}
} // namespace

void TJSONTree::Node::set_map()
{
   if (node->get().type() == nlohmann::json::value_t::object)
      return;

   if (isResettingPossible(node->get())) {
      node->get() = nlohmann::json::object();
   } else {
      throw std::runtime_error("cannot declare " + this->key() + " to be of map-type, already of type " +
                               node->get().type_name());
   }
}

void TJSONTree::Node::set_seq()
{
   if (node->get().type() == nlohmann::json::value_t::array)
      return;

   if (isResettingPossible(node->get())) {
      node->get() = nlohmann::json::array();
   } else {
      throw std::runtime_error("cannot declare " + this->key() + " to be of seq-type, already of type " +
                               node->get().type_name());
   }
}

std::string TJSONTree::Node::key() const
{
   return node->key();
}

namespace {
std::string itoa(int i)
{
   std::stringstream ss;
   ss << i;
   return ss.str();
}
std::string ftoa(float f)
{
   std::stringstream ss;
   ss << f;
   return ss.str();
}
} // namespace

std::string TJSONTree::Node::val() const
{
   switch (node->get().type()) {
   case nlohmann::json::value_t::string: return node->get().get<std::string>();
   case nlohmann::json::value_t::boolean: return node->get().get<bool>() ? "true" : "false";
   case nlohmann::json::value_t::number_integer: return ::itoa(node->get().get<int>());
   case nlohmann::json::value_t::number_unsigned: return ::itoa(node->get().get<unsigned int>());
   case nlohmann::json::value_t::number_float: return ::ftoa(node->get().get<float>());
   default:
      throw std::runtime_error(std::string("node " + node->key() + ": implicit string conversion for type " +
                                           node->get().type_name() + " not supported!"));
   }
}

int TJSONTree::Node::val_int() const
{
   return node->get().get<int>();
}
float TJSONTree::Node::val_float() const
{
   return node->get().get<float>();
}
bool TJSONTree::Node::val_bool() const
{
   auto const &nd = node->get();

   // Attempting to convert zeroes and ones to bools.
   if (nd.type() == nlohmann::json::value_t::number_unsigned) {
      auto val = nd.get<unsigned int>();
      if (val == 0)
         return false;
      if (val == 1)
         return true;
   }

   return nd.get<bool>();
}

bool TJSONTree::Node::has_key() const
{
   return node->key().size() > 0;
}

bool TJSONTree::Node::has_val() const
{
   return node->get().is_primitive();
}

bool TJSONTree::Node::has_child(std::string const &c) const
{
   return node->get().find(c) != node->get().end();
}

TJSONTree::Node &TJSONTree::Node::append_child()
{
   node->get().push_back("");
   return Impl::mkNode(tree, "", node->get().back());
}

size_t TJSONTree::Node::num_children() const
{
   return node->get().size();
}

TJSONTree::Node &TJSONTree::Node::child(size_t pos)
{
   return Impl::mkNode(tree, "", node->get().at(pos));
}

const TJSONTree::Node &TJSONTree::Node::child(size_t pos) const
{
   return Impl::mkNode(tree, "", node->get().at(pos));
}

using json_iterator = nlohmann::basic_json<>::iterator;
using const_json_iterator = nlohmann::basic_json<>::const_iterator;

template <class Nd, class NdType, class json_it>
class TJSONTree::Node::ChildItImpl final : public RooFit::Experimental::JSONNode::child_iterator_t<Nd>::Impl {
public:
   enum class POS { BEGIN, END };
   ChildItImpl(NdType &n, POS p)
      : node(n), iter(p == POS::BEGIN ? n.get_node().get().begin() : n.get_node().get().end()){};
   ChildItImpl(NdType &n, json_it it) : node(n), iter(it) {}
   ChildItImpl(const ChildItImpl &other) : node(other.node), iter(other.iter) {}
   using child_iterator = RooFit::Experimental::JSONNode::child_iterator_t<Nd>;
   std::unique_ptr<typename child_iterator::Impl> clone() const override
   {
      return std::make_unique<ChildItImpl>(node, iter);
   }
   void forward() override { ++iter; }
   void backward() override { --iter; }
   Nd &current() override
   {
      if (node.is_seq()) {
         return TJSONTree::Node::Impl::mkNode(node.get_tree(), "", iter.value());
      } else {
         return TJSONTree::Node::Impl::mkNode(node.get_tree(), iter.key(), iter.value());
      }
   }
   bool equal(const typename child_iterator::Impl &other) const override
   {
      auto it = dynamic_cast<const ChildItImpl<Nd, NdType, json_it> *>(&other);
      return it && it->iter == this->iter;
   }

private:
   NdType &node;
   json_it iter;
};

RooFit::Experimental::JSONNode::children_view TJSONTree::Node::children()
{
   using childIt = TJSONTree::Node::ChildItImpl<RooFit::Experimental::JSONNode, TJSONTree::Node, json_iterator>;
   return {child_iterator(std::make_unique<childIt>(*this, childIt::POS::BEGIN)),
           child_iterator(std::make_unique<childIt>(*this, childIt::POS::END))};
}
RooFit::Experimental::JSONNode::const_children_view TJSONTree::Node::children() const
{
   using childConstIt =
      TJSONTree::Node::ChildItImpl<const RooFit::Experimental::JSONNode, const TJSONTree::Node, const_json_iterator>;
   return {const_child_iterator(std::make_unique<childConstIt>(*this, childConstIt::POS::BEGIN)),
           const_child_iterator(std::make_unique<childConstIt>(*this, childConstIt::POS::END))};
}
