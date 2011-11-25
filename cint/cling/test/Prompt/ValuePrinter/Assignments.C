// RUN: cat %s | %cling | FileCheck %s

int a = 12;
a // CHECK: (int) 12

const char* b = "b" // CHECK: (const char *) "b"

struct C {int d;} E = {22};
E // CHECK: (struct C) @0x{{[0-9A-Fa-f].*}}
E.d // CHECK: (int) 22

#include <string>
std::string s("xyz") 
// CHECK: (std::string) @0x{{[0-9A-Fa-f]{8,}.}}
// CHECK: c_str: "xyz"
class Outer { 
public: 
  struct Inner { 
    enum E{ 
      A = 12, 
      B = 2, 
      C = 2} ABC; 
  }; 
};
Outer::Inner::C
// CHECK: (enum Outer::Inner::E const) @0x{{[0-9A-Fa-f].*}}
// CHECK: (Outer::Inner::E::B) ? (Outer::Inner::E::C) : (int) 2

.q
