class TheClass {
 public:
   TheClass(){};
   int a(int /* b */, int /* c */ =9) {return 1;}
   int a(float /* b */, float /* c */=9.) {return 2;}
   int a(int* /* b */, int* /* c */=0) {return 4;}
   int a(float* /* b */, int /* c */=9) {return 5;}
   int a(const char * /* b */, int /* c */=9) {return 6;}
};

