class TheBase {
public:
   TheBase() {}
};


class SimpleClass : public TheBase {
public:
   SimpleClass()
   {
      fData = -42;
   }

   int GetData()
   {
      return fData;
   }

   void SetData( int data )
   {
      fData = data;
      return;
   }

   void ByValue( SimpleClass )
   {
   }

public:
   int fData;
};

typedef SimpleClass SimpleClass_t;
