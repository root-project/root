int nCopies = 0;

class A{
public:
 A(){};
 A(const A&) {std::cout << "A copy ctor\n";nCopies++;}
 ~A(){std::cout << "A dtor\n";} 

};

int checkTThreadedObjectCopies() {

   A a;
   ROOT::TThreadedObject<A> toa(a);
   if (nCopies != 1) {
     std::cerr << "ERROR: 1 copy expected, but " << nCopies << " copies took place." << std::endl;
     return 1;
     }
   return 0;
}
