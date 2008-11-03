#include <iostream>
using namespace std;

class Object {
public:
   int value;
   Object() : value(1) {
#if defined(DEBUG)
      fprintf(stderr,"default constructor for %d\n",this);
#endif
   }
   Object(int val) : value(val) {
#if defined(DEBUG)
      fprintf(stderr,"constructor with int for %d\n",this);
#endif
   }
   ~Object() {
#if defined(DEBUG)
      fprintf(stderr,"default destructor for %d\n",this);
#endif
   }
};