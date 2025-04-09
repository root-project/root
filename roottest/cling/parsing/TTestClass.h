#ifndef TTestClass_h
#define TTestClass_h

class TTestClass {

   protected:
      unsigned   fInt;

   public:

   unsigned   GetI() { return fInt; }
   unsigned int GetII(void* ) { return fInt; }
   unsigned  GetIII(void* ) { return fInt; }


   unsigned * GetPI() { return &fInt; }
   unsigned int* GetPII(void* ) { return &fInt; }
   unsigned * GetPIII(void* ) { return &fInt; }


   TTestClass() {}
   virtual ~TTestClass() {}


};

#endif
