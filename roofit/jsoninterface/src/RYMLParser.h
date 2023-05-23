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

#ifndef RYML_PARSER_H
#define RYML_PARSER_H

#include <RooFit/Detail/JSONInterface.h>

#include <RConfigure.h>

#include <list>
#include <istream>
#include <memory>

class TRYMLTree : public RooFit::Detail::JSONTree {
protected:
   class Impl;
   std::unique_ptr<Impl> tree;

public:
   class Node : public RooFit::Detail::JSONNode {
   protected:
      TRYMLTree *tree;
      class Impl;
      friend TRYMLTree;
      std::unique_ptr<Impl> node;

   public:
      void writeJSON(std::ostream &os) const override;
      void writeYML(std::ostream &) const override;

      Node(TRYMLTree *t, const Impl &other);
      Node(const Node &other);
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
      bool has_key() const override;
      bool has_val() const override;
      bool has_child(std::string const &) const override;
      Node &append_child() override;
      size_t num_children() const override;
      Node &child(size_t pos) override;
      const Node &child(size_t pos) const override;
   };

protected:
   std::list<std::string> _strcache;
   std::list<Node> _nodecache;

public:
   Node &incache(const Node &n);
   const char *incache(const std::string &str);
   void clearcache();

public:
   TRYMLTree();
   ~TRYMLTree() override;
   TRYMLTree(std::istream &is);
   Node &rootnode() override;
};

#endif
