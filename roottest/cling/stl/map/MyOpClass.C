class MyOpClass {
public:
   const char* operator[](const char *s) {
      fprintf(stderr,"calling operator[] with index %s\n",s);
      return s;
   }
   MyOpClass &value() { return *this; }
};

