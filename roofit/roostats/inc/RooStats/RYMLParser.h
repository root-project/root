#ifndef RYML_PARSER_H
#define RYML_PARSER_H
#include "JSONInterface.h"

#ifdef INCLUDE_RYML
#include <list>
#include <istream>


#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>
#include <c4/yml/common.hpp>

class TRYMLParser {
 protected:
  ryml::Tree tree;
 public:
  class Node : public TJSONNode {
 protected:
   c4::yml::NodeRef node;
 public:
   virtual void writeJSON(std::ostream& os) const override;
   virtual void writeYML(std::ostream&) const override;
   
   Node(c4::yml::NodeRef n);
   Node(const Node& other);
   virtual Node& operator<< (std::string const& s) override ;
   virtual Node& operator<< (int i) override;
   virtual Node& operator<< (double d) override;       
   virtual const Node& operator>> (std::string &v) const override ;
   virtual Node& operator[] (std::string const& k) override ;
   virtual Node& operator[] (size_t pos) override ;
   virtual const Node& operator[] (std::string const& k) const override ;
   virtual const Node& operator[] (size_t pos) const override ;
   virtual bool is_container() const override ;
   virtual bool is_map() const override ;
   virtual bool is_seq() const override ;
   virtual void set_map() override;
   virtual void set_seq() override;  
   virtual std::string key() const override;
   virtual std::string val() const override;   
   virtual bool has_key() const override;
   virtual bool has_val() const override;  
   virtual bool has_child(std::string const&) const override;
   virtual TJSONNode& append_child() override;     
   virtual size_t num_children() const override;
   virtual Node& child(size_t pos) override;
   virtual const Node& child(size_t pos) const override;

 };
 protected:
  static std::list<std::string> _strcache;
  static std::list<Node> _nodecache;
 public:
  static Node& incache(const Node& n);
  static const char* incache(const std::string& str);
  static void clearcache();
  
 public:
  TRYMLParser();  
  TRYMLParser(std::istream& is);
  Node& rootnode();
};

#endif

#endif
