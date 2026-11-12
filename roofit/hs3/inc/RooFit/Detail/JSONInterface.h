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

#ifndef RooFit_Detail_JSONInterface_h
#define RooFit_Detail_JSONInterface_h

#include <iostream>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace RooFit {
namespace Detail {

class JSONNode {
public:
   template <class Nd>
   class child_iterator_t {
   public:
      explicit child_iterator_t(Nd *const *pos) : _pos(pos) {}

      child_iterator_t &operator++()
      {
         ++_pos;
         return *this;
      }
      child_iterator_t &operator--()
      {
         --_pos;
         return *this;
      }
      Nd &operator*() const { return **_pos; }
      Nd &operator->() const { return **_pos; }

      friend bool operator!=(child_iterator_t const &lhs, child_iterator_t const &rhs) { return lhs._pos != rhs._pos; }
      friend bool operator==(child_iterator_t const &lhs, child_iterator_t const &rhs) { return lhs._pos == rhs._pos; }

   private:
      Nd *const *_pos;
   };

   using child_iterator = child_iterator_t<JSONNode>;
   using const_child_iterator = child_iterator_t<const JSONNode>;

   template <class Nd>
   class children_view_t {
      child_iterator_t<Nd> b, e;

   public:
      inline children_view_t(child_iterator_t<Nd> const &b_, child_iterator_t<Nd> const &e_) : b(b_), e(e_) {}

      inline child_iterator_t<Nd> begin() const { return b; }
      inline child_iterator_t<Nd> end() const { return e; }
   };

   // A node refers to a value of its tree. Copying it would copy that reference, not the value.
   JSONNode(JSONNode const &) = delete;
   JSONNode &operator=(JSONNode const &) = delete;

public:
   void writeJSON(std::ostream &os) const;

public:
   JSONNode &operator<<(std::string const &s);
   inline JSONNode &operator<<(const char *s) { return *this << std::string(s); }
   JSONNode &operator<<(int i);
   JSONNode &operator<<(double d);
   JSONNode &operator<<(bool b);
   const JSONNode &operator>>(std::string &v) const;
   JSONNode &operator[](std::string const &k);
   const JSONNode &operator[](std::string const &k) const;
   bool is_container() const;
   bool is_map() const;
   bool is_seq() const;
   bool is_null() const;
   bool is_number() const;
   JSONNode &set_map();
   JSONNode &set_seq();
   JSONNode &set_null();
   void clear();
   /// Remove a child of a map, if it exists.
   void erase(std::string const &key);

   std::string key() const;
   std::string val() const;
   int val_int() const;
   double val_double() const;
   bool val_bool() const;
   template <class T>
   T val_t() const;
   bool has_key() const;
   bool has_val() const;
   bool has_child(std::string const &) const;
   JSONNode &append_child();
   size_t num_children() const;

   using children_view = children_view_t<JSONNode>;
   using const_children_view = children_view_t<const JSONNode>;

   children_view children();
   const_children_view children() const;
   JSONNode &child(size_t pos);
   const JSONNode &child(size_t pos) const;

   template <typename Collection>
   void fill_seq(Collection const &coll)
   {
      set_seq();
      for (auto const &item : coll) {
         append_child() << item;
      }
   }

   template <typename Collection>
   void fill_seq(Collection const &coll, size_t nmax)
   {
      set_seq();
      size_t n = 0;
      for (auto const &item : coll) {
         if (n >= nmax)
            break;
         append_child() << item;
         ++n;
      }
   }

   template <typename Collection, typename TransformationFunc>
   void fill_seq(Collection const &coll, TransformationFunc func)
   {
      set_seq();
      for (auto const &item : coll) {
         append_child() << func(item);
      }
   }

   template <typename Matrix>
   void fill_mat(Matrix const &mat)
   {
      set_seq();
      for (int i = 0; i < mat.GetNrows(); ++i) {
         auto &row = append_child();
         row.set_seq();
         for (int j = 0; j < mat.GetNcols(); ++j) {
            row.append_child() << mat(i, j);
         }
      }
   }

   JSONNode const *find(std::string const &key) const
   {
      auto &n = *this;
      return n.has_child(key) ? &n[key] : nullptr;
   }

   template <typename... Keys_t>
   JSONNode const *find(std::string const &key, Keys_t const &...keys) const
   {
      auto &n = *this;
      return n.has_child(key) ? n[key].find(keys...) : nullptr;
   }

   JSONNode &get(std::string const &key)
   {
      auto &n = *this;
      n.set_map();
      return n[key];
   }

   template <typename... Keys_t>
   JSONNode &get(std::string const &key, Keys_t const &...keys)
   {
      auto &next = get(key);
      next.set_map();
      return next.get(keys...);
   }

private:
   friend class JSONTree;
   friend class JSONReadRecorder;

   // Owns the values and the nodes of a tree. It is defined in the only source file that knows the JSON library.
   class Storage;

   JSONNode(Storage &storage, void *value, std::string key);

   Storage *_storage;
   void *_value; // of the value type of the JSON library
   std::string _key;
};

class JSONTree {
public:
   ~JSONTree();

   JSONNode &rootnode();

   static std::unique_ptr<JSONTree> create();
   static std::unique_ptr<JSONTree> create(std::istream &is);
   static std::unique_ptr<JSONTree> create(std::string const &str);

private:
   explicit JSONTree(std::unique_ptr<JSONNode::Storage> storage);

   std::unique_ptr<JSONNode::Storage> _storage;
};

std::ostream &operator<<(std::ostream &os, RooFit::Detail::JSONNode const &s);

template <class T>
std::vector<T> &operator<<(std::vector<T> &v, RooFit::Detail::JSONNode::children_view const &cv)
{
   for (const auto &e : cv) {
      v.push_back(e.val_t<T>());
   }
   return v;
}

template <class T>
std::vector<T> &operator<<(std::vector<T> &v, RooFit::Detail::JSONNode::const_children_view const &cv)
{
   for (const auto &e : cv) {
      v.push_back(e.val_t<T>());
   }
   return v;
}

template <class T>
std::vector<T> &operator<<(std::vector<T> &v, RooFit::Detail::JSONNode const &n)
{
   if (!n.is_seq()) {
      throw std::runtime_error("node " + n.key() + " is not of sequence type!");
   }
   v << n.children();
   return v;
}

template <class Key, class T, class Hash, class KeyEqual, class Allocator>
RooFit::Detail::JSONNode &
operator<<(RooFit::Detail::JSONNode &n, const std::unordered_map<Key, T, Hash, KeyEqual, Allocator> &m)
{
   n.set_map();
   for (const auto &it : m) {
      n[it.first] << it.second;
   }
   return n;
}

template <class Key, class T, class Compare, class Allocator>
RooFit::Detail::JSONNode &operator<<(RooFit::Detail::JSONNode &n, const std::map<Key, T, Compare, Allocator> &m)
{
   n.set_map();
   for (const auto &it : m) {
      n[it.first] << it.second;
   }
   return n;
}

template <>
inline int JSONNode::val_t<int>() const
{
   return val_int();
}
template <>
inline double JSONNode::val_t<double>() const
{
   return val_double();
}
template <>
inline bool JSONNode::val_t<bool>() const
{
   return val_bool();
}
template <>
inline std::string JSONNode::val_t<std::string>() const
{
   return val();
}

} // namespace Detail
} // namespace RooFit

#endif
