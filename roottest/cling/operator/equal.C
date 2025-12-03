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

class withRef : public top  {
   const int &ref;
   int &ref2;
public:
   withRef(int &p) : ref(p),ref2(p) {};
   int get() { return ref + ref2; }
};

class withHiddenRef : public top {
   withHiddenRef &operator=(const withHiddenRef&); // intentionally NOT implemented
   int a;
#ifndef __CINT__
   const int &ref;
   int &ref2;
#endif
public:
   withHiddenRef(int &p) : ref(p),ref2(p) {};
};

#ifdef __CINT__
#pragma link C++ class withHiddenRef-;
#endif


void copying() {
//    test2 t1;
//    test2 t2;
//     t1 = t2;
}




