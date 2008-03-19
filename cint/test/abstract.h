class top {
public:
   virtual int getone() = 0;
   virtual int gettwo() = 0;
   virtual double getthree(int) const = 0;
};

class middle : public top {
public:
   int getone() { return 1; }
};

class bottom : public middle {
public:
   int gettwo() { return 2; }
   double getthree(int) const { return 3.0; }
};


