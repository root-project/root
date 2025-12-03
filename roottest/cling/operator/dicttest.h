// make sure "private" is seen by the compiler - CINT's dict hides it
#ifdef private
# undef private
# define private private
#endif

class noctor {
private:
   noctor(const noctor&);
};

class nonconstctor {
public:
   nonconstctor(nonconstctor&) {}
private:
#ifndef _MSC_VER
   nonconstctor(const nonconstctor&) {}
#endif
};

class withconstctor {
public:
   withconstctor(const withconstctor&) {}
   //private: - must not be private due to dictionary:
   // p = new withconstctor(*(withconstctor*) libp->para[0].ref);
   //withconstctor(withconstctor&);
};


class dicttest{
public:
   operator const noctor&() {return *n;}
   operator const noctor*() {return n;}
   operator const noctor&() const {return *n;}
   operator const noctor*() const {return n;}

   operator const withconstctor&() {return *w;}
   operator const nonconstctor*() {return nc;}
   //operator const nonconstctor() {}
   operator const withconstctor&() const {return *w;}
   operator const nonconstctor*() const {return nc;}
   //operator const nonconstctor() const {}

   operator nonconstctor&() {return *nc;}
   operator nonconstctor*() {return nc;}
   //operator nonconstctor() {}
   operator nonconstctor&() const {return *nc;}
   operator nonconstctor*() const {return nc;}
   //operator nonconstctor() const {}

   noctor* n;
   withconstctor* w;
   nonconstctor* nc;
};
