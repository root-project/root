//#include <iostream>
//using namespace std;

class A
{
public:
    A &operator() ();
    A &operator() (double a);
    A &operator() (double a, double b);
};

A &A::operator() ()
{
   //    cerr<< " no arg " <<endl;
    return *this;
}

A &A::operator() (double a)
{
   //    cerr<<a<<endl;
    return *this;
}

A &A::operator() (double a, double b)
{
   //   cerr<<a << " and " << b <<endl;
    return *this;
}

int t01()
{
    A a;
    a(3.2)(5.4);
    a()(3.2)(5.4,3.0);
#ifdef ClingWorkAroundErracticValuePrinter
    printf("(int)0\n");
#endif
    return 0;
}


