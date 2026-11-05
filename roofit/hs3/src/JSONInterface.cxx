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

#include <deque>
#include <istream>
#include <memory>
#include <sstream>

// This is the only translation unit where the JSON parsing library is used.
// Hence, the JSON engine backing the RooFit::Detail::JSONTree could in
// principle be swapped out by changing only this file.

namespace RooFit {
namespace Detail {

namespace {

nlohmann::json &json(void *value)
{
   return *static_cast<nlohmann::json *>(value);
}

inline nlohmann::json parseWrapper(std::istream &is)
{
   try {
      return nlohmann::json::parse(is);
   } catch (const nlohmann::json::exception &ex) {
      throw std::runtime_error(ex.what());
   }
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

} // namespace

// The nodes live as long as the tree, because the interface hands out references to them.
class JSONNode::Storage {
public:
   // Not root{value}, which would be an array that contains the value.
   explicit Storage(nlohmann::json value) : root(std::move(value)), rootNode{*this, &root, ""} {}

   JSONNode &makeNode(nlohmann::json &value, std::string key)
   {
      nodes.push_back(std::unique_ptr<JSONNode>{new JSONNode{*this, &value, std::move(key)}});
      return *nodes.back();
   }

   // The children are created up front, such that the iterators can be pointers into this list.
   std::vector<JSONNode *> const &makeChildren(nlohmann::json &value)
   {
      childLists.emplace_back();
      std::vector<JSONNode *> &list = childLists.back();
      if (!value.is_structured()) {
         return list;
      }
      list.reserve(value.size());
      for (auto it = value.begin(); it != value.end(); ++it) {
         list.push_back(&makeNode(it.value(), value.is_array() ? "" : it.key()));
      }
      return list;
   }

   nlohmann::json root;
   JSONNode rootNode;
   std::vector<std::unique_ptr<JSONNode>> nodes;
   std::deque<std::vector<JSONNode *>> childLists;
};

// JSONNode

JSONNode::JSONNode(Storage &storage, void *value, std::string key)
   : _storage{&storage}, _value{value}, _key{std::move(key)}
{
}

void JSONNode::writeJSON(std::ostream &os) const
{
   os << json(_value);
}

JSONNode &JSONNode::operator<<(std::string const &s)
{
   json(_value) = s;
   return *this;
}

JSONNode &JSONNode::operator<<(int i)
{
   json(_value) = i;
   return *this;
}

JSONNode &JSONNode::operator<<(double d)
{
   json(_value) = d;
   return *this;
}

JSONNode &JSONNode::operator<<(bool b)
{
   json(_value) = b;
   return *this;
}

const JSONNode &JSONNode::operator>>(std::string &v) const
{
   v = json(_value).get<std::string>();
   return *this;
}

JSONNode &JSONNode::operator[](std::string const &k)
{
   return _storage->makeNode(json(_value)[k], k);
}

const JSONNode &JSONNode::operator[](std::string const &k) const
{
   return _storage->makeNode(json(_value)[k], k);
}

bool JSONNode::is_container() const
{
   return json(_value).is_array() || json(_value).is_object();
}

bool JSONNode::is_map() const
{
   return json(_value).is_object();
}

bool JSONNode::is_seq() const
{
   return json(_value).is_array();
}

bool JSONNode::is_null() const
{
   return json(_value).is_null();
}

bool JSONNode::is_number() const
{
   return json(_value).is_number();
}

JSONNode &JSONNode::set_map()
{
   if (json(_value).type() == nlohmann::json::value_t::object)
      return *this;

   if (isResettingPossible(json(_value))) {
      json(_value) = nlohmann::json::object();
   } else {
      throw std::runtime_error("cannot declare \"" + this->key() + "\" to be of map - type, already of type " +
                               json(_value).type_name());
   }
   return *this;
}

JSONNode &JSONNode::set_seq()
{
   if (json(_value).type() == nlohmann::json::value_t::array)
      return *this;

   if (isResettingPossible(json(_value))) {
      json(_value) = nlohmann::json::array();
   } else {
      throw std::runtime_error("cannot declare \"" + this->key() + "\" to be of seq - type, already of type " +
                               json(_value).type_name());
   }
   return *this;
}

JSONNode &JSONNode::set_null()
{
   json(_value) = nullptr;
   return *this;
}

void JSONNode::clear()
{
   json(_value).clear();
}

void JSONNode::erase(std::string const &key)
{
   json(_value).erase(key);
}

std::string JSONNode::key() const
{
   return _key;
}

std::string JSONNode::val() const
{
   nlohmann::json const &value = json(_value);
   switch (value.type()) {
   case nlohmann::json::value_t::string: return value.get<std::string>();
   case nlohmann::json::value_t::boolean: return value.get<bool>() ? "true" : "false";
   case nlohmann::json::value_t::number_integer: return std::to_string(value.get<int>());
   case nlohmann::json::value_t::number_unsigned: return std::to_string(value.get<unsigned int>());
   case nlohmann::json::value_t::number_float: {
      std::stringstream ss;
      ss << value.get<double>();
      return ss.str();
   }
   default:
      throw std::runtime_error("node \"" + _key + "\": implicit string conversion for type " + value.type_name() +
                               " not supported!");
   }
}

int JSONNode::val_int() const
{
   return json(_value).get<int>();
}
double JSONNode::val_double() const
{
   return json(_value).get<double>();
}
bool JSONNode::val_bool() const
{
   auto const &nd = json(_value);

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

bool JSONNode::has_key() const
{
   return !_key.empty();
}

bool JSONNode::has_val() const
{
   return json(_value).is_primitive();
}

bool JSONNode::has_child(std::string const &c) const
{
   return json(_value).find(c) != json(_value).end();
}

JSONNode &JSONNode::append_child()
{
   json(_value).push_back("");
   return _storage->makeNode(json(_value).back(), "");
}

size_t JSONNode::num_children() const
{
   return json(_value).size();
}

JSONNode &JSONNode::child(size_t pos)
{
   return _storage->makeNode(json(_value).at(pos), "");
}

const JSONNode &JSONNode::child(size_t pos) const
{
   return _storage->makeNode(json(_value).at(pos), "");
}

JSONNode::children_view JSONNode::children()
{
   std::vector<JSONNode *> const &list = _storage->makeChildren(json(_value));
   return {child_iterator{list.data()}, child_iterator{list.data() + list.size()}};
}

JSONNode::const_children_view JSONNode::children() const
{
   std::vector<JSONNode *> const &list = _storage->makeChildren(json(_value));
   return {const_child_iterator{list.data()}, const_child_iterator{list.data() + list.size()}};
}

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}

// JSONTree

JSONTree::JSONTree(std::unique_ptr<JSONNode::Storage> storage) : _storage{std::move(storage)} {}

JSONTree::~JSONTree() = default;

JSONNode &JSONTree::rootnode()
{
   return _storage->rootNode;
}

std::unique_ptr<JSONTree> JSONTree::create()
{
   return std::unique_ptr<JSONTree>{new JSONTree{std::make_unique<JSONNode::Storage>(nlohmann::json{})}};
}

std::unique_ptr<JSONTree> JSONTree::create(std::istream &is)
{
   return std::unique_ptr<JSONTree>{new JSONTree{std::make_unique<JSONNode::Storage>(parseWrapper(is))}};
}

std::unique_ptr<JSONTree> JSONTree::create(std::string const &str)
{
   std::stringstream ss{str};
   return JSONTree::create(ss);
}

} // namespace Detail
} // namespace RooFit
