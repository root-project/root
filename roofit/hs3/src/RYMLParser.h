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

#include "JSONInterface.h"
#include "RConfigure.h"

#include <list>
#include <istream>
#include <memory>

class TRYMLTree : public JSONTree {
protected:
   class Impl;
   std::unique_ptr<Impl> tree;

public:
   class Node : public RooFit::Experimental::JSONNode {
   protected:
      TRYMLTree *tree;
      class Impl;
      friend TRYMLTree;
      std::unique_ptr<Impl> node;

   public:
      virtual void writeJSON(std::ostream &os) const override;
      virtual void writeYML(std::ostream &) const override;

      Node(TRYMLTree *t, const Impl &other);
      Node(const Node &other);
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
      virtual bool has_key() const override;
      virtual bool has_val() const override;
      virtual bool has_child(std::string const &) const override;
      virtual Node &append_child() override;
      virtual size_t num_children() const override;
      virtual Node &child(size_t pos) override;
      virtual const Node &child(size_t pos) const override;
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
   ~TRYMLTree();
   TRYMLTree(std::istream &is);
   Node &rootnode();
};

#endif
