#include "RooStats/JSONParser.h"

#include "json.hpp"



class TJSONTree::Impl {

};

class TJSONTree::Node::Impl {

};
  
void TJSONTree::Node::writeJSON(std::ostream& os) const {

}

TJSONTree::Node::Node(TJSONTree* t,const Impl& other){

}

TJSONTree::Node::Node(const Node& other){

}

TJSONTree::Node& TJSONTree::Node::operator<< (std::string const& s) {

}

TJSONTree::Node& TJSONTree::Node::operator<< (int i) {

}

TJSONTree::Node& TJSONTree::Node::operator<< (double d) {

}

const TJSONTree::Node& TJSONTree::Node::operator>> (std::string &v) const {

}

TJSONTree::Node& TJSONTree::Node::operator[] (std::string const& k) {

}

TJSONTree::Node& TJSONTree::Node::operator[] (size_t pos) {

}

const TJSONTree::Node& TJSONTree::Node::operator[] (std::string const& k) const {

}

const TJSONTree::Node& TJSONTree::Node::operator[] (size_t pos) const {

}

bool TJSONTree::Node::is_container() const {

}

bool TJSONTree::Node::is_map() const {

}

bool TJSONTree::Node::is_seq() const {

}

void TJSONTree::Node::set_map() {

}

void TJSONTree::Node::set_seq() {

}

std::string TJSONTree::Node::key() const {

}

std::string TJSONTree::Node::val() const {

}

bool TJSONTree::Node::has_key() const {

}

bool TJSONTree::Node::has_val() const {

}

bool TJSONTree::Node::has_child(std::string const&) const {

}

TJSONTree::Node& TJSONTree::Node::append_child() {

}

size_t TJSONTree::Node::num_children() const {

}

TJSONTree::Node& TJSONTree::Node::child(size_t pos) {

}

const TJSONTree::Node& TJSONTree::Node::child(size_t pos) const {

}
