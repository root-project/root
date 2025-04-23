#ifdef __CINT__
// GCC and clang disagree with CINT, i.e. CINT fails for those.
# define HIDEKNOWNFAILURES
#endif

class D_FRIEND;

class BASE {
public:
   BASE(): mPublic(-7), mProtected(-11), mPrivate(-13) {}
   int mPublic;
   static const int sPublic = 7;
protected:
   int mProtected;
   static const int sProtected = 11;
private:
   int mPrivate;
   static const int sPrivate = 13;

   friend class D_FRIEND;
};

class D: BASE {
public:
   int good0() { return mPublic; }
   int good1() { return mProtected; }
   int good2() { return 99; }
#ifndef HIDEKNOWNFAILURES
   static const int dPublic = sPublic;
   static const int dProtected = sProtected;
#endif
} d;

class D_PRIVATE: private BASE {
public:
   int good0() { return mPublic; }
   int good1() { return mProtected; }
   int good2() { return 99; }
#ifndef HIDEKNOWNFAILURES
   static const int dPublic = sPublic;
   static const int dProtected = sProtected;
#endif
} d_private;

class D_PROTECTED: protected BASE {
public:
   int good0() { return mPublic; }
   int good1() { return mProtected; }
   int good2() { return 99; }
#ifndef HIDEKNOWNFAILURES
   static const int dPublic = sPublic;
   static const int dProtected = sProtected;
#endif
} d_protected;

class D_PUBLIC: public BASE {
public:
   int good0() { return mPublic; }
   int good1() { return mProtected; }
   int good2() { return 99; }
   static const int dPublic = sPublic;
   static const int dProtected = sProtected;
} d_public;

class D_FRIEND {
public:
   int good0() { return b.mPublic; }
   int good1() { return b.mProtected; }
   int good2() { return b.mPrivate; }
   BASE b;
   static const int dPublic = BASE::sPublic;
#ifndef HIDEKNOWNFAILURES
   static const int dProtected = BASE::sProtected;
   static const int dPrivate = BASE::sPrivate;
#endif
} d_friend;

int assertBaseMembers() {
   int i = d_public.mPublic
      + d.good0() + d.good1() + d.good2()
      + d_private.good0() + d_private.good1() + d_private.good2()
      + d_protected.good0() + d_protected.good1() + d_protected.good2()
      + d_public.good0() + d_public.good1() + d_public.good2()
      + d_friend.good0() + d_friend.good1() + d_friend.good2();
   return 0;
}
