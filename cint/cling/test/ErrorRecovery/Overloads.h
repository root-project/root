#include <fstream>
#include <string>

int main () {
  std::string s;
  std::ofstream("file.txt") << s << std::endl;
  return 0;
}
