#ifndef TTestClass_h
#define TTestClass_h

class TTestClass {

   protected:  
      unsigned   fInt;

   public: 
   
   unsigned   GetI() { return fInt; }
   unsigned int GetII(void* arg) { return fInt; }
   unsigned  GetIII(void* arg) { return fInt; }


   unsigned * GetPI() { return &fInt; }
   unsigned int* GetPII(void* arg) { return &fInt; }
   unsigned * GetPIII(void* arg) { return &fInt; }

   
   TTestClass() {}
   virtual ~TTestClass() {}
     
   
};

#endif
