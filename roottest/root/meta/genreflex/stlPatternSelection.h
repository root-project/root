#include <map>
#include <string>

std::map<std::string, bool> a1;
std::map<char, bool> a2;
std::map<const char*, bool> a3;
std::map<int, bool> a11;
std::map<int, const double*> a21;
std::map<int, unsigned long int> a31;
std::map<double, bool> b12;
std::map<double, const double*> b22;
std::map<double, unsigned long int> b32;

std::multimap<std::string, bool> a12;
std::multimap<char, bool> a22;
std::multimap<const char*, bool> a32;
std::multimap<int, bool> a112;
std::multimap<int, const double*> a212;
std::multimap<int, unsigned long int> a312;
std::multimap<double, bool> a123;
std::multimap<double, const double*> a223;
std::multimap<double, unsigned long int> a323;
