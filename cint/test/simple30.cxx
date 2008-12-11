#include <stdio.h>

class top {};
class bottom : public top {};

void func(top *&) {
   fprintf(stderr,"top\n");
}
void func(const bottom*) {
   fprintf(stderr,"bottom\n");
}

int main() {
   top *p = 0;
   bottom *b = 0;

   func(b);
   return 0;
}
