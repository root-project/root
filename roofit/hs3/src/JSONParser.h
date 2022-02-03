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

#include <RooFitHS3/JSONInterface.h>

#include <istream>
#include <memory>
#include <list>

class TJSONTree : public JSONTree {
public:
   class Node : public RooFit::Experimental::JSONNode {
   protected:
      TJSONTree *tree;
      class Impl;
      template <class Nd, class NdType, class json_it>
      class ChildItImpl;
      friend TJSONTree;
      std::unique_ptr<Impl> node;

      const TJSONTree *get_tree() const;
      TJSONTree *get_tree();
      const Impl &get_node() const;
      Impl &get_node();

   public:
      virtual void writeJSON(std::ostream &os) const override;

      Node(TJSONTree *t, std::istream &is);
      Node(TJSONTree *t, Impl &other);
      Node(TJSONTree *t);
      Node(const Node &other);
      virtual ~Node();
      virtual Node &operator<<(std::string const &s) override;
      virtual Node &operator<<(int i) override;
      virtual Node &operator<<(double d) override;
      virtual const Node &operator>>(std::string &v) const override;
      virtual Node &operator[](std::string const &k) override;
      virtual Node &operator[](size_t pos) override;
      virtual const Node &operator[](std::string const &k) const override;
      virtual const Node &operator[](size_t pos) const override;
      virtual bool is_container() const override;
      virtual bool is_map() const override;
      virtual bool is_seq() const override;
      virtual void set_map() override;
      virtual void set_seq() override;
      virtual std::string key() const override;
      virtual std::string val() const override;
      virtual int val_int() const override;
      virtual float val_float() const override;
      virtual bool val_bool() const override;
      virtual bool has_key() const override;
      virtual bool has_val() const override;
      virtual bool has_child(std::string const &) const override;
      virtual Node &append_child() override;
      virtual size_t num_children() const override;
      virtual Node &child(size_t pos) override;
      virtual const Node &child(size_t pos) const override;

      children_view children() override;
      const_children_view children() const override;
   };

protected:
   Node root;
   std::list<Node> _nodecache;
   void clearcache();

public:
   TJSONTree();
   virtual ~TJSONTree();
   TJSONTree(std::istream &is);
   TJSONTree::Node &incache(const TJSONTree::Node &n);

   virtual Node &rootnode() override;
};
#endif
