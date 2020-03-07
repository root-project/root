#ifndef JSON_PARSER_H
#define JSON_PARSER_H
#include "JSONInterface.h"

#include <istream>
#include <memory>

class TJSONTree : public JSONTree {
 protected:
  class Impl;
  std::unique_ptr<Impl> tree;
 public:
  class Node : public JSONNode {
 protected:
   TJSONTree* tree;
   class Impl;
   friend TJSONTree;
   std::unique_ptr<Impl> node;
 public:
   virtual void writeJSON(std::ostream& os) const override;

   Node(TJSONTree* t,const Impl& other);   
   Node(const Node& other);
   virtual Node& operator<< (std::string const& s) override;
   virtual Node& operator<< (int i) override;
   virtual Node& operator<< (double d) override;       
   virtual const Node& operator>> (std::string &v) const override;
   virtual Node& operator[] (std::string const& k) override;
   virtual Node& operator[] (size_t pos) override;
   virtual const Node& operator[] (std::string const& k) const override;
   virtual const Node& operator[] (size_t pos) const override;
   virtual bool is_container() const override;
   virtual bool is_map() const override;
   virtual bool is_seq() const override;
   virtual void set_map() override;
   virtual void set_seq() override;  
   virtual std::string key() const override;
   virtual std::string val() const override;   
   virtual bool has_key() const override;
   virtual bool has_val() const override;  
   virtual bool has_child(std::string const&) const override;
   virtual Node& append_child() override;     
   virtual size_t num_children() const override;
   virtual Node& child(size_t pos) override;
   virtual const Node& child(size_t pos) const override;
 };
  
 public:
  TJSONTree();  
  TJSONTree(std::istream& is);
  virtual Node& rootnode() override;
};
#endif
