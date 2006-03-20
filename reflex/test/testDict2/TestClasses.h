
namespace testclasses {
  
  class UnnamedTypes {
    
    struct {
      int i;
      double d;
    } m_struct;

    union {
      struct { int i; } m_str1;
      struct { int j; } m_str2;
    } m_union;

  private:
    
    struct Private_Inner_Struct {
      int i;
      double j;
    };
    
  };

  struct PrivateConstructors {
    int i;
    void foo() {}
  };

  namespace OverloadedOperators {

     struct NoOp { public: std::vector<int> makeItComplex; };

     struct OpNew {
        public:
        void * operator new ( size_t n ) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }
     };

     struct PlOpNew {
        public:
        void * operator new ( size_t n, void * v ) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }
     };

     struct PlOpOpNew {
        public:
        void * operator new ( size_t n ) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }
        void * operator new ( size_t n, void * v ) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }
     };

     struct OpANew {
        public:
        void * operator new[] ( size_t n ) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }
     };

     struct PlOpANew {
        public:
        void * operator new[] ( size_t n, void * v ) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }
     };

     struct PlOpAOpANew {
        public:
        void * operator new[] ( size_t n ) { std::cout << "overloaded new" << n << std::endl; return malloc(n); }
        void * operator new[] ( size_t n, void * v ) { std::cout << "overloaded placement new " << n << v << std::endl; return malloc(n); }
     };

  } // ns OverloadedOperators

  
#if 0
  
  struct ConversionOperator {

    typedef const int* (ConversionOperator::* ptr_to_mem_fun)() const;

    operator ptr_to_mem_fun() const { return &ConversionOperator::i; }
    
    typedef int* ConversionOperator::* ptr_to_mem_data;

    operator ptr_to_mem_data() const { return &ConversionOperator::m_ip; }

    const int* i() const { return &m_i; }

    int m_i;

    int * m_ip;

  };

  
  template < class T > struct ConversionOperatorT {

    typedef const T* (ConversionOperatorT<T>::* ptr_to_mem_fun)() const;

    operator ptr_to_mem_fun() const { return &ConversionOperatorT<T>::i; }
    
    typedef T* ConversionOperatorT<T>::* ptr_to_mem_data;

    operator ptr_to_mem_data() const { return &ConversionOperatorT<T>::m_ip; }

    const T* i() const { return &m_i; }

    T m_i;

    T * m_ip;

  };

#endif
  
} // namespace testclasses


// template instances
namespace {
  struct _testclasses_instances {
    //testclasses::ConversionOperatorT<int> m1;
  };
}
