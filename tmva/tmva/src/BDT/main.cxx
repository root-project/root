#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>

#include "bdt.h"

using json = nlohmann::json;

json read_file(const std::string &filename) {
  std::ifstream i(filename);
  json j;
  i >> j;
  //std::cout << "Read file: " << j.type();
  return j;
}

std::string read_file_string(const std::string &filename){
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

void print(const std::string message){
  std::cout << message << std::endl;
}

void print_json_type(json j){
  std::string my_type = "unknown";
  if (j.type() == json::value_t::null){my_type = "null";}
  else if (j.type() == json::value_t::boolean){my_type = "boolean";}
  else if (j.type() == json::value_t::number_integer){my_type = "number_integer";}
  else if (j.type() == json::value_t::number_unsigned){my_type = "number_unsigned";}
  else if (j.type() == json::value_t::number_float){my_type = "number_float";}
  else if (j.type() == json::value_t::object){my_type = "object";}
  else if (j.type() == json::value_t::array){my_type = "array";}
  else if (j.type() == json::value_t::string){my_type = "string";}
  std::cout << "Type: " << my_type << '\n';
}




int main() {
  json config = read_file("model.json");
  std::string my_config = read_file_string("model.json");
  std::cout << "String: " << my_config << std::endl;

  auto j3 = json::parse(my_config);
  std::cout << "json stringed" << j3.dump() << std::endl;
  std::cout << "Size: " <<  j3.size() << std::endl;

  auto my_type = j3.type();
  //std::cout << my_type.get<std::string>();

  int count = 0;
  for (auto &tree : j3){
    std::cout << tree << "\n\n";
    count++;
  }
  std::cout << "\n *************************** \n\n";

  //print(j3[0].size().dump());
  std::cout << j3[0].size();
  std::cout << j3[0]["children"];

  print_json_type(j3);
  print_json_type(j3[0]);
  print_json_type(j3[0]["children"]);
  std::cout << j3[0]["children"].size();

  std::cout << "\n" << j3[1] << std::endl;

  int base_count = 0;
  for (auto &tree : j3){
    base_count++;
    std::cout << base_count << "[";
    int sub_count=0;
    for (auto &objs : tree){
      sub_count++;
      std::cout << sub_count << "{";
      int sscount = 0;
      for (auto &stree : objs){
        sscount++;
        std::cout << sscount << ",";
      }
      std::cout << "}\n";
    }
    std::cout << "]\n\n";
  }

  //"split":"f0","split_condition"
  /*
  [{"children":
    [{"children":
      [{"children":
        [{"leaf":-0.167088613,"nodeid":7},{"leaf":0.0222222228,"nodeid":8}],"depth":2,"missing":7,"no":8,"nodeid":3,"split":"f0","split_condition":11.5,"yes":7}
      ,{"children":
        [{"leaf":-0.11260505,"nodeid":9} ,{"leaf":-0.0258064512,"nodeid":10}]
        ,"depth":2,"missing":9,"no":10,"nodeid":4,"split":"f0","split_condition":4.5,"yes":9}
      ],"depth":1,"missing":3,"no":4,"nodeid":1,"split":"f1","split_condition":101.5,"yes":3},
      {"children":[{"children":[{"leaf":-0.0141414134,"nodeid":11},{"leaf":0.0769230798,"nodeid":12}],
      "depth":2,"missing":11,"no":12,"nodeid":5,"split":"f0","split_condition":7.5,"yes":11},{"children":[{"leaf":0.0600000024,"nodeid":13},{"leaf":0.132075474,"nodeid":14}],
      "depth":2,"missing":13,"no":14,"nodeid":6,"split":"f1","split_condition":166.5,"yes":13}],
      "depth":1,"missing":5,"no":6,"nodeid":2,"split":"f1","split_condition":159.5,"yes":5}],
      "depth":0,"missing":1,"no":2,"nodeid":0,"split":"f1","split_condition":127.5,"yes":1},{"children":[{"children":[{"children":[{"leaf":-0.159606487,"nodeid":7},{"leaf":-0.100172974,"nodeid":8}],
      "depth":2,"missing":7,"no":8,"nodeid":3,"split":"f1","split_condition":103.5,"yes":7},{"children":[{"leaf":-0.112435006,"nodeid":9},{"leaf":-0.0204244778,"nodeid":10}],
      "depth":2,"missing":9,"no":10,"nodeid":4,"split":"f1","split_condition":99.5,"yes":9}],
      "depth":1,"missing":3,"no":4,"nodeid":1,"split":"f0","split_condition":4.5,"yes":3},{"children":[{"children":[{"leaf":-0.0127850501,"nodeid":11},{"leaf":0.0705055743,"nodeid":12}],
      "depth":2,"missing":11,"no":12,"nodeid":5,"split":"f0","split_condition":7.5,"yes":11},{"children":[{"leaf":0.0552411862,"nodeid":13},{"leaf":0.120366327,"nodeid":14}],
      "depth":2,"missing":13,"no":14,"nodeid":6,"split":"f1","split_condition":166.5,"yes":13}],
      "depth":1,"missing":5,"no":6,"nodeid":2,"split":"f1","split_condition":159.5,"yes":5}],
      "depth":0,"missing":1,"no":2,"nodeid":0,"split":"f1","split_condition":127.5,"yes":1}]
      */
  //Node a;
  //std::cout << "Child nums: " << a.get_child_num() << std::endl;

  // Define nodes // todo: automatize
  Node n1; Node n2; Node n3;
  LeafNode n4; LeafNode n5;
  LeafNode n6; LeafNode n7;

  Node normalNodes[3] = {n1,n2,n3};
  LeafNode leafNodes[4] = {n4,n5,n6,n7};

  // define childs of nodes // TODO: automatize
  n1.child_1 = &n2; n1.child_2 = &n3;
  n2.child_1 = &n4; n2.child_2 = &n5;
  n3.child_1 = &n6; n3.child_2 = &n6;


  std::cout << n2.kind << std::endl;
  std::cout << n7.kind << std::endl;
  //Node n2;  Node n3;  Node n4;  Node n5;  Node n6;  Node n7;

  std::map<unsigned int, AbstractNode> level_0;

  Tree my_tree;
  /*
  std::map<unsigned int, AbstractNode> level_1;
  std::map<unsigned int, AbstractNode> level_2;
  */

  //my_tree.nodes[]
  std::cout << "\n***** END *****" << std::endl;
} // End main


/*
int count = 0;
for (auto &tree : config){
  std::cout << count << std::endl;
  std::cout << tree << "\n\n";
  count++;

}
std::cout << " *************************** \n";
//std::cout << config.type();
// explicit conversion to string
std::string s = config.dump();
// serialization with pretty printing
// pass in the amount of spaces to indent
std::cout << config.dump(3) << std::endl;
std::cout << "config[0] \n"
          << config[0] << std::endl;
std::cout << "config[1] \n"
          << config[10] << std::endl;
std::cout << " ---- \n";
std::cout << config[0].size() << '\n';
std::cout << " ---- \n";
std::cout << config[0] << '\n';
std::cout << " ---- \n"
                    << config[0]["children"][0]["children"] << std::endl;
*/
