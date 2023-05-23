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

#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <RooFit/Detail/JSONInterface.h>

#include <istream>
#include <memory>
#include <list>

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
      Node &set_map() override;
      Node &set_seq() override;
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
   void clearcache();

public:
   TJSONTree();
   ~TJSONTree() override;
   TJSONTree(std::istream &is);
   TJSONTree::Node &incache(const TJSONTree::Node &n);

   Node &rootnode() override { return root; }
};
#endif
