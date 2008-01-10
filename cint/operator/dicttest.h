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
   nonconstctor(nonconstctor&);
 private:
   nonconstctor(const nonconstctor&);
};

class withconstctor {
 public:
   withconstctor(const withconstctor&);
   //private: - must not be private due to dictionary:
   // p = new withconstctor(*(withconstctor*) libp->para[0].ref);
   //withconstctor(withconstctor&);
};


class dicttest{
 public:
   operator const noctor&();
   operator const noctor*();
   operator const noctor&() const;
   operator const noctor*() const;

   operator const withconstctor&();
   operator const nonconstctor*();
   //operator const nonconstctor();
   operator const withconstctor&() const;
   operator const nonconstctor*() const;
   //operator const nonconstctor() const;

   operator nonconstctor&();
   operator nonconstctor*();
   //operator nonconstctor();
   operator nonconstctor&() const;
   operator nonconstctor*() const;
   //operator nonconstctor() const;
};
