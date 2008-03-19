namespace onehand {
   int myvar;
}

namespace otherhand {
   using onehand::myvar;
}

class Top {
public:
   int getPriv() { return mypriv; } 
protected:
   int mypriv;
};

class Bottom : public Top {
public:
   using Top::mypriv;
};

#include <stdio.h>

int main() {
   onehand::myvar = 3;
   Bottom b;
   b.mypriv = 2;
   printf("otherhand is %d\n",otherhand::myvar);
   printf("b.A::mypriv is %d\n",b.getPriv());
   printf("b.mypriv is %d\n",b.mypriv);
   return 0;
}
