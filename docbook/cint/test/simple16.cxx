#ifndef __CINT__
#include <stdio.h>
#endif

class top {
public:
   void fromtop() {
      printf("fromtop\n");
   }
};

class middle : public top {
public:
   void frommiddle() {
      printf("frommiddle\n");
   }
};

class bottom : public middle {
public:
   void frombottom() {
      printf("frombottom\n");
   }
};

int main() {
   bottom b;
   b.frombottom();
   b.frommiddle();
   b.fromtop();
   return 0;
}
