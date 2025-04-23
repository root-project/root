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

   virtual ~MemTester() {
      --counter;
   }

   void Dummy() {}

public:
   static void CallRef( MemTester& ) {}
   static void CallConstRef( const MemTester& ) {}
   static void CallPtr( MemTester* ) {}
   static void CallConstPtr( const MemTester* ) {}
};

int MemTester::counter = 0;
