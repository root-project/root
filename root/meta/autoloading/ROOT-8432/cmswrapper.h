namespace edm {
   template <class T> struct Wrapper { T* t; Wrapper(); };
   template <class T> Wrapper<T>::Wrapper(): t{} {}
}
