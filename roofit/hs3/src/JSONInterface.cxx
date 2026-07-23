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

#include <nlohmann/json.hpp>

#include <istream>
#include <list>
#include <memory>
#include <sstream>

// This is the only translation unit where the JSON parsing library is used.
// Hence, the JSON engine backing the RooFit::Detail::JSONTree could in
// principle be swapped out by changing only this file.

namespace {

class TJSONTree : public RooFit::Detail::JSONTree {
public:
   class Node : public RooFit::Detail::JSONNode {
   protected:
      TJSONTree *tree;
      class Impl;
      template <class Nd, class NdType, class json_it>
      class ChildItImpl;
      friend TJSONTree;
      std::unique_ptr<Impl> node;

      const TJSONTree *get_tree() const { return tree; }
      TJSONTree *get_tree() { return tree; }
      const Impl &get_node() const;
      Impl &get_node();

   public:
      void writeJSON(std::ostream &os) const override;

      Node(TJSONTree *t, std::istream &is);
      Node(TJSONTree *t, Impl &other);
      Node(TJSONTree *t);
      Node(const Node &other);
      virtual ~Node();
      Node &operator<<(std::string const &s) override;
      Node &operator<<(int i) override;
      Node &operator<<(double d) override;
      Node &operator<<(bool b) override;
      const Node &operator>>(std::string &v) const override;
      Node &operator[](std::string const &k) override;
      const Node &operator[](std::string const &k) const override;
      bool is_container() const override;
      bool is_map() const override;
      bool is_seq() const override;
      bool is_null() const override;
      bool is_number() const override;
      Node &set_map() override;
      Node &set_seq() override;
      Node &set_null() override;
      void clear() override;
      std::string key() const override;
      std::string val() const override;
      int val_int() const override;
      double val_double() const override;
      bool val_bool() const override;
      bool has_key() const override;
      bool has_val() const override;
      bool has_child(std::string const &) const override;
      Node &append_child() override;
      size_t num_children() const override;
      Node &child(size_t pos) override;
      const Node &child(size_t pos) const override;

      children_view children() override;
      const_children_view children() const override;
   };

protected:
   Node root;
   std::list<Node> _nodecache;

public:
   TJSONTree();
   ~TJSONTree() override;
   TJSONTree(std::istream &is);
   TJSONTree::Node &incache(const TJSONTree::Node &n);

   Node &rootnode() override { return root; }
};

inline nlohmann::json parseWrapper(std::istream &is)
{
   try {
      return nlohmann::json::parse(is);
   } catch (const nlohmann::json::exception &ex) {
      throw std::runtime_error(ex.what());
   }
}

// TJSONTree methods

TJSONTree::TJSONTree() : root(this) {};

TJSONTree::TJSONTree(std::istream &is) : root(this, is) {};

TJSONTree::~TJSONTree()
{
   TJSONTree::_nodecache.clear();
};

TJSONTree::Node &TJSONTree::incache(const TJSONTree::Node &n)
{
   _nodecache.push_back(n);
   return _nodecache.back();
}

const TJSONTree::Node::Impl &TJSONTree::Node::get_node() const
{
   return *node;
}

TJSONTree::Node::Impl &TJSONTree::Node::get_node()
{
   return *node;
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
   BaseNode(std::istream &is) : Impl(""), node(parseWrapper(is)) {}
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

TJSONTree::Node::~Node() = default;

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

TJSONTree::Node &TJSONTree::Node::operator<<(bool b)
{
   node->get() = b;
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

const TJSONTree::Node &TJSONTree::Node::operator[](std::string const &k) const
{
   return Impl::mkNode(tree, k, node->get()[k]);
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

bool TJSONTree::Node::is_null() const
{
   return node->get().is_null();
}

bool TJSONTree::Node::is_number() const
{
   return node->get().is_number();
}

// To check whether it's allowed to reset the type of an object. We allow
// this for nodes that have no type yet, or nodes with an empty string.
bool isResettingPossible(nlohmann::json const &node)
{

   if (node.type() == nlohmann::json::value_t::null) {
      return true;
   }

   if (node.type() == nlohmann::json::value_t::string) {
      if (node.get<std::string>().empty()) {
         return true;
      }
   }
   return false;
}

TJSONTree::Node &TJSONTree::Node::set_map()
{
   if (node->get().type() == nlohmann::json::value_t::object)
      return *this;

   if (isResettingPossible(node->get())) {
      node->get() = nlohmann::json::object();
   } else {
      throw std::runtime_error("cannot declare \"" + this->key() + "\" to be of map - type, already of type " +
                               node->get().type_name());
   }
   return *this;
}

TJSONTree::Node &TJSONTree::Node::set_seq()
{
   if (node->get().type() == nlohmann::json::value_t::array)
      return *this;

   if (isResettingPossible(node->get())) {
      node->get() = nlohmann::json::array();
   } else {
      throw std::runtime_error("cannot declare \"" + this->key() + "\" to be of seq - type, already of type " +
                               node->get().type_name());
   }
   return *this;
}

TJSONTree::Node &TJSONTree::Node::set_null()
{
   node->get() = nullptr;
   return *this;
}

void TJSONTree::Node::clear()
{
   node->get().clear();
}

std::string TJSONTree::Node::key() const
{
   return node->key();
}

std::string TJSONTree::Node::val() const
{
   switch (node->get().type()) {
   case nlohmann::json::value_t::string: return node->get().get<std::string>();
   case nlohmann::json::value_t::boolean: return node->get().get<bool>() ? "true" : "false";
   case nlohmann::json::value_t::number_integer: return std::to_string(node->get().get<int>());
   case nlohmann::json::value_t::number_unsigned: return std::to_string(node->get().get<unsigned int>());
   case nlohmann::json::value_t::number_float: {
      std::stringstream ss;
      ss << node->get().get<double>();
      return ss.str();
   }
   default:
      throw std::runtime_error("node \"" + node->key() + "\": implicit string conversion for type " +
                               node->get().type_name() + " not supported!");
   }
}

int TJSONTree::Node::val_int() const
{
   return node->get().get<int>();
}
double TJSONTree::Node::val_double() const
{
   return node->get().get<double>();
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
   return !node->key().empty();
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
class TJSONTree::Node::ChildItImpl final : public RooFit::Detail::JSONNode::child_iterator_t<Nd>::Impl {
public:
   enum class POS {
      BEGIN,
      END
   };
   ChildItImpl(NdType &n, POS p)
      : node(n), iter(p == POS::BEGIN ? n.get_node().get().begin() : n.get_node().get().end()) {};
   ChildItImpl(NdType &n, json_it it) : node(n), iter(it) {}
   ChildItImpl(const ChildItImpl &other) : node(other.node), iter(other.iter) {}
   using child_iterator = RooFit::Detail::JSONNode::child_iterator_t<Nd>;
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
      // We can use static_cast here because we never compare Iterators for
      // different JSON node types.
      auto it = static_cast<const ChildItImpl<Nd, NdType, json_it> *>(&other);
      return it && it->iter == this->iter;
   }

private:
   NdType &node;
   json_it iter;
};

RooFit::Detail::JSONNode::children_view TJSONTree::Node::children()
{
   using childIt = TJSONTree::Node::ChildItImpl<RooFit::Detail::JSONNode, TJSONTree::Node, json_iterator>;
   return {child_iterator(std::make_unique<childIt>(*this, childIt::POS::BEGIN)),
           child_iterator(std::make_unique<childIt>(*this, childIt::POS::END))};
}
RooFit::Detail::JSONNode::const_children_view TJSONTree::Node::children() const
{
   using childConstIt =
      TJSONTree::Node::ChildItImpl<const RooFit::Detail::JSONNode, const TJSONTree::Node, const_json_iterator>;
   return {const_child_iterator(std::make_unique<childConstIt>(*this, childConstIt::POS::BEGIN)),
           const_child_iterator(std::make_unique<childConstIt>(*this, childConstIt::POS::END))};
}

// Iterator implementation that is agnostic of the JSON parsing backend, used
// for the default implementation of JSONNode::children().
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
