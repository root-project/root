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
      // Note max and min ints are platform dependent since we cannot use 
      //limits.h
      A = 2147483647,
      B = 2, 
      C = 2,
      D = -2147483648 
    } ABC; 
  }; 
};
Outer::Inner::C
// CHECK: (enum Outer::Inner::E const) @0x{{[0-9A-Fa-f].*}}
// CHECK: (Outer::Inner::E::B) ? (Outer::Inner::E::C) : (int) 2
Outer::Inner::D
// CHECK: (enum Outer::Inner::E const) @0x{{[0-9A-Fa-f].*}}
// CHECK: (Outer::Inner::E::D) : (int) -2147483648
.q
