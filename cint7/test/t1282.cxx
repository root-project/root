namespace onehand {
   int myvar;
}

namespace otherhand {
   using onehand::myvar;
}

class Top {
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
   return 0;
}
