#ifndef SENTINEL
#define SENTINEL
#include <iostream>

class Sentinel {
public:
   Sentinel() : x(0) {
      std::cout << "ctor called" << std::endl;
   }

   Sentinel(const Sentinel& o) : x(o.x) {
      std::cout << "copy-ctor called" << std::endl;
   }

   ~Sentinel() {
      std::cout << "dtor called" << std::endl;
   }

   void set(int _x) { x = _x; }
   int get() const { return x; }

private:
   int x;
};

#endif // SENTINEL
