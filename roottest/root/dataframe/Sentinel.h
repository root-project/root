#ifndef SENTINEL
#define SENTINEL

class Sentinel {
   static unsigned int fCtorCalls;
   static unsigned int fDtorCalls;
   static unsigned int fCopyCtorCalls;

public:
   Sentinel() : x(0) { fCtorCalls++; }

   Sentinel(const Sentinel &o) : x(o.x) {}

   ~Sentinel() { fDtorCalls++; }

   void set(int _x) { x = _x; }
   int get() const { return x; }

   static unsigned int GetCtorCalls() { return fCtorCalls; }
   static unsigned int GetCopyCtorCalls() { return fCopyCtorCalls; }
   static unsigned int GetDtorCalls() { return fDtorCalls; }

private:
   int x;
};

#endif // SENTINEL
