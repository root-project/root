#include <string>
#include <iostream>

struct Data { int i; };

struct Final : public Data, public std::string 
{ 
   void SetString(const char *input) { *(std::string*)this = input; }
   void print(std::ostream &out) {
      out << i << endl;
      out << c_str() << endl;
   }
};

