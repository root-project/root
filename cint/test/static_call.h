// Minimal test code to prove a cint behavior that may be a bug
#include <string>

class A {
public:
   static void SetConfig(const std::string s);
   static std::string &GetConfig() {
      return configStr;
   }
private:
  static std::string configStr;
};

class B {
public:
  B();
  ~B();
  static int callThrough(const std::string s);
private:
  A* myA;
};

