#include <stdio.h>


void func(int a) {
   fprintf(stderr,"running func with %d\n",a);
}

typedef void (*funcptr)(int);
typedef void (funcval)(int);

// This seems to be functionaly equivalent to 
// wrinting the prototype 'void val(int)'.
extern funcval val; 

#ifdef __CINT__
funcptr ptr = func;
#else
funcptr ptr(func);
#endif

template <funcval generator> class proxy {
public:
   void exec(int a) { generator(a); }
};

// syntax error: error: `retval' declared as function returning a function
// funcval retval() { return func; }

// correct return value;
funcval &retval() { return func; }
funcval &retval2() { return val; }


funcptr retptr() { return func; }
funcptr retptr2() { return val; }


int main() {
   // val = func;
   val(3);
   ptr(4);
   ptr = val;
   ptr(5);

   proxy< val > p;
   p.exec(6);
}

void val(int a) {
   fprintf(stderr,"running val with %d\n",a);
}
