class cont {};

class top {
public:
   top& operator=(const top&) { return *this; }
};

#ifndef HIDE
class withConstMember {
   // should not generate the operator= (but does in cint 5.15.130 with the operator= support enabled).
public:
   withConstMember(int val = 0) : a(val) {};
   const int a;
};

class bot : public withConstMember {
public:
   int b;
};
#endif

class privateOp {
private:
   privateOp& operator=(const privateOp&);
public:
   int c;
};

class privateOp2 {
private:
   privateOp2& operator=(const int&);
public:
   int c;
};

class test {
private:
   void somethingprivate(); 
public:
   cont c;
};

class test2 {
public:
   privateOp op;
};

void copying() {
//    test2 t1;
//    test2 t2;
//     t1 = t2;
}




