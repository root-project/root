#include "RooStats/RYMLParser.h"
#include <sstream>

void TRYMLParser::Node::writeJSON(std::ostream& os) const {
  os << c4::yml::as_json(node);
}
void TRYMLParser::Node::writeYML(std::ostream& os) const {
  os << node;
}

std::list<std::string> TRYMLParser::_strcache = std::list<std::string>();
std::list<TRYMLParser::Node> TRYMLParser::_nodecache = std::list<TRYMLParser::Node>();

void TRYMLParser::Node::set_map() {
  node |= c4::yml::MAP;
}
void TRYMLParser::Node::set_seq() {
  node |= c4::yml::SEQ;
}

TRYMLParser::TRYMLParser(std::istream& is){
  std::string s(std::istreambuf_iterator<char>(is), {});
  this->tree = c4::yml::parse(c4::to_csubstr(s.c_str()));
}
TRYMLParser::TRYMLParser(){}

TRYMLParser::Node& TRYMLParser::rootnode(){
  return TRYMLParser::incache(Node(tree.rootref()));
}
TRYMLParser::Node::Node(const Node& other)  : node(other.node) {}   

TRYMLParser::Node::Node(c4::yml::NodeRef n) : node(n) {}

TRYMLParser::Node& TRYMLParser::Node::operator<< (std::string const& s) {
  node << s;
  return *this;
}
TRYMLParser::Node& TRYMLParser::Node::operator<< (int i) {
  node << i;
  return *this;
}
TRYMLParser::Node& TRYMLParser::Node::operator<< (double d) {
  node << d;
  return *this;
}

const TRYMLParser::Node& TRYMLParser::Node::operator>> (std::string &v) const {
  node >> v;
  return *this;
}
TRYMLParser::Node& TRYMLParser::Node::operator[] (std::string const& k) {
  return TRYMLParser::incache(Node(node[c4::to_csubstr(TRYMLParser::incache(k))]));
}
TRYMLParser::Node& TRYMLParser::Node::operator[] (size_t pos) {
  return TRYMLParser::incache(Node(node[pos]));     
}
const TRYMLParser::Node& TRYMLParser::Node::operator[] (std::string const& k) const {
  return TRYMLParser::incache(Node(node[c4::to_csubstr(TRYMLParser::incache(k))]));     
}
const TRYMLParser::Node& TRYMLParser::Node::operator[] (size_t pos) const {
  return TRYMLParser::incache(Node(node[pos]));          
}
bool TRYMLParser::Node::is_container() const {
  return node.is_container();
}
bool TRYMLParser::Node::is_map() const {
  return node.is_map();
}
bool TRYMLParser::Node::is_seq() const {
  return node.is_seq();
}

std::string TRYMLParser::Node::key() const {
  std::stringstream ss;    
  ss << node.key();
  return ss.str();
}
std::string TRYMLParser::Node::val() const {;
  std::stringstream ss;    
  ss << node.val();
  return ss.str();
}

TJSONNode& TRYMLParser::Node::append_child(){
  return TRYMLParser::incache(Node(node.append_child()));
}

bool TRYMLParser::Node::has_key() const {
  return node.has_key();
}
bool TRYMLParser::Node::has_val() const {
  return node.has_val();
}

bool TRYMLParser::Node::has_child(std::string const&s) const {
  return node.has_child(c4::to_csubstr(s.c_str()));
}
size_t TRYMLParser::Node::num_children() const {
  return node.num_children();
}
TRYMLParser::Node& TRYMLParser::Node::child(size_t pos){
  return TRYMLParser::incache(Node(node.child(pos)));
}
const TRYMLParser::Node& TRYMLParser::Node::child(size_t pos) const {
  return TRYMLParser::incache(Node(node.child(pos)));
}

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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// obtain a const pointer from the string cache
const char* TRYMLParser::incache(const std::string& str){
  TRYMLParser::_strcache.push_back(str);
  return   TRYMLParser::_strcache.back().c_str();
}

TRYMLParser::Node& TRYMLParser::incache(const TRYMLParser::Node& n){
  TRYMLParser::_nodecache.push_back(n);
  return _nodecache.back();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// clear the string cache
void TRYMLParser::clearcache(){
  TRYMLParser::_strcache.clear();
  TRYMLParser::_nodecache.clear();  
}

