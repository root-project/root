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

#ifndef RooFitHS3_JSONInterface_h
#define RooFitHS3_JSONInterface_h

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace RooFit {
namespace Experimental {

class JSONNode {
public:
   template <class Nd>
   class child_iterator_t {
   public:
      class Impl {
      public:
         virtual ~Impl() = default;
         virtual std::unique_ptr<Impl> clone() const = 0;
         virtual void forward() = 0;
         virtual void backward() = 0;
         virtual Nd &current() = 0;
         virtual bool equal(const Impl &other) const = 0;
      };

   private:
      std::unique_ptr<Impl> it;

   public:
      child_iterator_t(std::unique_ptr<Impl> impl) : it(std::move(impl)) {}
      child_iterator_t(const child_iterator_t &other) : it(std::move(other.it->clone())) {}

      child_iterator_t &operator++()
      {
         it->forward();
         return *this;
      }
      child_iterator_t &operator--()
      {
         it->backward();
         return *this;
      }
      Nd &operator*() const { return it->current(); }
      Nd &operator->() const { return it->current(); }

      bool operator!=(const child_iterator_t &that) const { return !this->it->equal(*that.it); }
      bool operator==(const child_iterator_t &that) const { return this->it->equal(*that.it); }
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

public:
   virtual void writeJSON(std::ostream &os) const = 0;
   virtual void writeYML(std::ostream &) const { throw std::runtime_error("YML not supported"); }

public:
   virtual JSONNode &operator<<(std::string const &s) = 0;
   virtual JSONNode &operator<<(int i) = 0;
   virtual JSONNode &operator<<(double d) = 0;
   template <class T>
   JSONNode &operator<<(const std::vector<T> &v)
   {
      this->set_seq();
      for (const auto &e : v) {
         this->append_child() << e;
      };
      return *this;
   }
   virtual const JSONNode &operator>>(std::string &v) const = 0;
   virtual JSONNode &operator[](std::string const &k) = 0;
   virtual JSONNode &operator[](size_t pos) = 0;
   virtual const JSONNode &operator[](std::string const &k) const = 0;
   virtual const JSONNode &operator[](size_t pos) const = 0;
   virtual bool is_container() const = 0;
   virtual bool is_map() const = 0;
   virtual bool is_seq() const = 0;
   virtual void set_map() = 0;
   virtual void set_seq() = 0;

   virtual std::string key() const = 0;
   virtual std::string val() const = 0;
   virtual int val_int() const { return atoi(this->val().c_str()); }
   virtual float val_float() const { return atof(this->val().c_str()); }
   virtual bool val_bool() const { return atoi(this->val().c_str()); }
   template <class T>
   T val_t() const;
   virtual bool has_key() const = 0;
   virtual bool has_val() const = 0;
   virtual bool has_child(std::string const &) const = 0;
   virtual JSONNode &append_child() = 0;
   virtual size_t num_children() const = 0;

   using children_view = children_view_t<JSONNode>;
   using const_children_view = children_view_t<const JSONNode>;

   virtual children_view children();
   virtual const_children_view children() const;
   virtual JSONNode &child(size_t pos) = 0;
   virtual const JSONNode &child(size_t pos) const = 0;
};

} // namespace Experimental
} // namespace RooFit

class JSONTree {
   virtual RooFit::Experimental::JSONNode &rootnode() = 0;
};

std::ostream &operator<<(std::ostream &os, RooFit::Experimental::JSONNode const &s);
template <class T>
std::vector<T> &operator<<(std::vector<T> &v, RooFit::Experimental::JSONNode const &n)
{
   if (!n.is_seq()) {
      throw std::runtime_error("node " + n.key() + " is not of sequence type!");
   }
   for (const auto &e : n.children()) {
      v.push_back(e.val_t<T>());
   }
   return v;
}

#endif
