class MyId {
public:
   static MyId reg() { return MyId(); }
};

MyId id = MyId::reg();
