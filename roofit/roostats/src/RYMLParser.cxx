#include "RooStats/RYMLParser.h"
#include <sstream>

#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>
#include <c4/yml/common.hpp>

// TRYMLTree Implementation
class TRYMLTree::Impl {
  ryml::Tree tree;
public:
  Impl(){};
  Impl(std::istream& is){
    std::string s(std::istreambuf_iterator<char>(is), {});
    tree = c4::yml::parse(c4::to_csubstr(s.c_str()));
  } 
  inline ryml::Tree& get(){
    return tree;
  }
  inline const ryml::Tree& get() const {
    return tree;
  }
};

// TRYMLNode Implementation
class TRYMLTree::Node::Impl {
  c4::yml::NodeRef node;
public:
  inline c4::yml::NodeRef& get(){
    return node;
  }
  inline const c4::yml::NodeRef& get() const {
    return node;
  }
  Impl(const c4::yml::NodeRef& n) : node(n) {};
  inline static TRYMLTree::Node& mkNode(TRYMLTree* t,c4::yml::NodeRef node){
    return t->incache(TRYMLTree::Node(t,TRYMLTree::Node::Impl(node)));
  }  
};

// JSONTree interface implementation

void TRYMLTree::Node::writeJSON(std::ostream& os) const {
  os << c4::yml::as_json(node->get());
}

void TRYMLTree::Node::writeYML(std::ostream& os) const {
  os << node->get();
}

void TRYMLTree::Node::set_map() {
  node->get() |= c4::yml::MAP;
}

void TRYMLTree::Node::set_seq() {
  node->get() |= c4::yml::SEQ;
}

TRYMLTree::TRYMLTree(std::istream& is) : tree(new Impl(is)) {};

TRYMLTree::TRYMLTree() : tree(new Impl()) {};

TRYMLTree::~TRYMLTree(){ clearcache(); };

// JSONNode interface implementation

TRYMLTree::Node::Node(TRYMLTree* t, const TRYMLTree::Node::Impl& imp) : tree(t), node(new Impl(imp)) {}

TRYMLTree::Node::Node(const Node& other) : Node(other.tree,other.node->get()) {}   

TRYMLTree::Node& TRYMLTree::rootnode(){
  return Node::Impl::mkNode(this,tree->get().rootref());
}

TRYMLTree::Node& TRYMLTree::Node::operator<< (std::string const& s) {
  node->get() << s;
  return *this;
}

TRYMLTree::Node& TRYMLTree::Node::operator<< (int i) {
  node->get() << i;
  return *this;
}

TRYMLTree::Node& TRYMLTree::Node::operator<< (double d) {
  node->get() << d;
  return *this;
}

const TRYMLTree::Node& TRYMLTree::Node::operator>> (std::string &v) const {
  node->get() >> v;
  return *this;
}

TRYMLTree::Node& TRYMLTree::Node::operator[] (std::string const& k) {
  return Impl::mkNode(tree,node->get()[c4::to_csubstr(tree->incache(k))]);
}

TRYMLTree::Node& TRYMLTree::Node::operator[] (size_t pos) {
  return Impl::mkNode(tree,node->get()[pos]);     
}

const TRYMLTree::Node& TRYMLTree::Node::operator[] (std::string const& k) const {
  return Impl::mkNode(tree,node->get()[c4::to_csubstr(tree->incache(k))]);     
}

const TRYMLTree::Node& TRYMLTree::Node::operator[] (size_t pos) const {
  return Impl::mkNode(tree,node->get()[pos]);          
}

bool TRYMLTree::Node::is_container() const {
  return node->get().is_container();
}

bool TRYMLTree::Node::is_map() const {
  return node->get().is_map();
}

bool TRYMLTree::Node::is_seq() const {
  return node->get().is_seq();
}

std::string TRYMLTree::Node::key() const {
  std::stringstream ss;    
  ss << node->get().key();
  return ss.str();
}

std::string TRYMLTree::Node::val() const {;
  std::stringstream ss;    
  ss << node->get().val();
  return ss.str();
}

TRYMLTree::Node& TRYMLTree::Node::append_child(){
  return Impl::mkNode(tree,node->get().append_child());
}

bool TRYMLTree::Node::has_key() const {
  return node->get().has_key();
}

bool TRYMLTree::Node::has_val() const {
  return node->get().has_val();
}

bool TRYMLTree::Node::has_child(std::string const&s) const {
  return node->get().has_child(c4::to_csubstr(s.c_str()));
}

size_t TRYMLTree::Node::num_children() const {
  return node->get().num_children();
}

TRYMLTree::Node& TRYMLTree::Node::child(size_t pos){
  return Impl::mkNode(tree,node->get().child(pos));
}

const TRYMLTree::Node& TRYMLTree::Node::child(size_t pos) const {
  return Impl::mkNode(tree,node->get().child(pos));
}

// specific functions

namespace {
  void error_cb(const char* msg, size_t msg_len, void *user_data){
    throw std::runtime_error(msg);
  }
  
  bool setcallbacks(){
    c4::yml::set_callbacks(c4::yml::Callbacks(c4::yml::get_callbacks().m_user_data,
                                              c4::yml::get_callbacks().m_allocate,
                                              c4::yml::get_callbacks().m_free,
                                              &::error_cb));
    return true;
  }
  bool ok = setcallbacks();
}

const char* TRYMLTree::incache(const std::string& str){
  _strcache.push_back(str);
  return _strcache.back().c_str();
}

TRYMLTree::Node& TRYMLTree::incache(const TRYMLTree::Node& n){
  _nodecache.push_back(n);
  return _nodecache.back();
}

void TRYMLTree::clearcache(){
  TRYMLTree::_strcache.clear();
  TRYMLTree::_nodecache.clear();  
}

