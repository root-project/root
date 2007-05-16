   class BAR {
   public:
      int i;
      float f;
   };
   class FOO {
   public:
      int a;
      double b;
      BAR* bar;
      int A() const {return a;}
      int A() {return ++a;}
      double B() { if (bar) return bar->f; else return -1.;}
   };
