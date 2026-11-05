/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_HS3Schema_h
#define RooFit_HS3Schema_h

#include "HS3Document.h"

#include <RooFit/Detail/JSONInterface.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace RooFit {
namespace HS3 {

/// A position in the JSON tree. Keeps track of how it got there, in order to create the paths in error messages only
/// when they are needed.
class Cursor {
public:
   Cursor(Detail::JSONNode const &node, Diagnostics &diagnostics) : _node{node}, _diagnostics{diagnostics} {}
   Cursor(Detail::JSONNode const &node, Cursor const &parent, const char *key)
      : _node{node}, _diagnostics{parent._diagnostics}, _parent{&parent}, _key{key}
   {
   }
   Cursor(Detail::JSONNode const &node, Cursor const &parent, std::size_t index)
      : _node{node}, _diagnostics{parent._diagnostics}, _parent{&parent}, _index{index}
   {
   }

   Detail::JSONNode const &node() const { return _node; }
   std::string path() const;
   void error(std::string const &message) const;
   void warning(std::string const &message) const;
   std::size_t numErrors() const;

private:
   Detail::JSONNode const &_node;
   Diagnostics &_diagnostics;
   Cursor const *_parent = nullptr;
   const char *_key = nullptr;
   std::size_t _index = 0;
};

/// Reads the components of a struct. The functions take the same arguments as the ones of the Writer, such that a
/// single function can describe both directions.
class Reader {
public:
   explicit Reader(Cursor const &cursor);
   /// Reports the components that were not read, unless reading the struct gave an error.
   ~Reader();
   Reader(Reader const &) = delete;
   Reader &operator=(Reader const &) = delete;

   /// Read a required component, or an optional one if the field is a std::optional.
   template <class T>
   void operator()(const char *key, T &field);
   template <class T>
   void operator()(const char *key, std::optional<T> &field);

   /// Read an array that can be omitted, which is equivalent to an empty one.
   template <class T>
   void list(const char *key, std::vector<T> &field);

   /// Read the components of a struct that is nested in a component.
   template <class Fn>
   void nested(const char *key, Fn &&fn);

   /// Whether the struct has a component. Checking for a component counts as reading it.
   bool has(const char *key);

   Cursor const &cursor() const { return _cursor; }

private:
   Cursor const &_cursor;
   std::vector<std::string> _seen;
   std::size_t _nErrorsBefore = 0;
};

void readValue(Cursor const &c, std::string &out);
void readValue(Cursor const &c, double &out);
void readValue(Cursor const &c, int &out);
void readValue(Cursor const &c, bool &out);
void readValue(Cursor const &c, ParameterValue &out);
void readValue(Cursor const &c, Data &out);
void readValue(Cursor const &c, Interpolation &out);

template <class Tag>
void readValue(Cursor const &c, Ref<Tag> &out)
{
   readValue(c, out.name);
}

template <class T>
void readValue(Cursor const &c, std::vector<T> &out)
{
   Detail::JSONNode const &node = c.node();
   if (!node.is_seq()) {
      c.error("must be an array");
      return;
   }
   out.clear();
   out.reserve(node.num_children());
   std::size_t i = 0;
   for (Detail::JSONNode const &child : node.children()) {
      out.emplace_back();
      readValue(Cursor{child, c, i}, out.back());
      ++i;
   }
}

/// Everything that has a describe() function is read from a struct.
template <class T>
auto readValue(Cursor const &c, T &out) -> std::void_t<decltype(describe(std::declval<Reader &>(), out))>
{
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   Reader reader{c};
   describe(reader, out);
}

template <class T>
void Reader::operator()(const char *key, T &field)
{
   if (!has(key)) {
      _cursor.error(std::string("does not define the required component \"") + key + "\"");
      return;
   }
   readValue(Cursor{_cursor.node()[key], _cursor, key}, field);
}

template <class T>
void Reader::operator()(const char *key, std::optional<T> &field)
{
   if (!has(key)) {
      return;
   }
   Detail::JSONNode const &node = _cursor.node()[key];
   if (node.is_null()) {
      return;
   }
   field.emplace();
   readValue(Cursor{node, _cursor, key}, *field);
}

template <class T>
void Reader::list(const char *key, std::vector<T> &field)
{
   if (!has(key)) {
      return;
   }
   Detail::JSONNode const &node = _cursor.node()[key];
   if (node.is_null()) {
      return;
   }
   readValue(Cursor{node, _cursor, key}, field);
}

template <class Fn>
void Reader::nested(const char *key, Fn &&fn)
{
   if (!has(key)) {
      _cursor.error(std::string("does not define the required component \"") + key + "\"");
      return;
   }
   Cursor child{_cursor.node()[key], _cursor, key};
   if (!child.node().is_map()) {
      child.error("must be a struct");
      return;
   }
   Reader reader{child};
   fn(reader);
}

} // namespace HS3
} // namespace RooFit

#endif
