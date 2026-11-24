#ifndef SENTINEL
#define SENTINEL

class Sentinel {
   static unsigned int fCtorCalls;
   static unsigned int fDtorCalls;
   static unsigned int fCopyCtorCalls;
   int x;

public:
   Sentinel() : x(0) { fCtorCalls++; }

   Sentinel(const Sentinel &o) : x(o.x) { fCopyCtorCalls++; }
   Sentinel &operator=(const Sentinel &) = delete;

   Sentinel(Sentinel &&) = delete;
   Sentinel &operator=(Sentinel &&) = delete;

   ~Sentinel() { fDtorCalls++; }

   void set(int _x) { x = _x; }
   int get() const { return x; }

   static unsigned int GetCtorCalls() { return fCtorCalls; }
   static unsigned int GetCopyCtorCalls() { return fCopyCtorCalls; }
   static unsigned int GetDtorCalls() { return fDtorCalls; }
};

#endif // SENTINEL
