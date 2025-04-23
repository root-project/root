namespace ns {

  class MyClass {
  public:
    MyClass() {}
    
    class iterator {
    public:
      iterator() {}
    };
    
    iterator begin(){return iterator();}
    iterator end(){return iterator();}
    
  };
}
