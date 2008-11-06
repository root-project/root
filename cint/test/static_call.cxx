// Minimal test code to prove a cint behavior that may be a bug
#include "static_call.h"

std::string A::configStr = std::string();
void A::SetConfig(const std::string s){
   configStr=s;
}

B::B():
  myA(0)
{}
B::~B(){
  if (myA){
    delete myA;
    myA=0;
  };
}

int B::callThrough(const std::string s)
{
   A::SetConfig(s);
   return s == A::GetConfig();
};

int main()
{
   return !B::callThrough("config_info");
}

