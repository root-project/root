class TheBase {
public:
   TheBase() {}
};


class SimpleClass : public TheBase {
public:
   SimpleClass() {
      m_data = -42;
   }

   int GetData() {
      return m_data;
   }

   void SetData( int data ) {
      m_data = data;
   }

public:
   int m_data;
};