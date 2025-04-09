#include <string>
#include <iostream>

bool t01(std::string &s) {
   std::cout << "t01 received --" << s << "--\n";
   if (s == "test string") return true;
   else return false;
}

bool t01val(std::string s) {
   std::cout << "t01 received --" << s << "--\n";
   if (s == "test string") return true;
   else return false;
}

bool t01p(std::string *s) {
   std::cout << "t01 received --" << *s << "--\n";
   if (*s == "test string") return true;
   else return false;
}
