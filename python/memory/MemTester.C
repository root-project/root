class MemTester {
public:
   static int counter;

public:
   MemTester() {
      ++counter;
   }

   MemTester( const MemTester& ) {
      ++counter;
   }

   ~MemTester() {
      --counter;
   }

   void Dummy() {}
};

int MemTester::counter = 0;
