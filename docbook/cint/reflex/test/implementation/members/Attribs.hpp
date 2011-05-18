template <typename T>
class X {
private:
   void f_0() {}
   void fc_0() const {}
   virtual void vf_0() {}
   virtual void vfc_0() const {}
   static void sf_0() {}
   virtual void af_0() = 0;
   virtual void afc_0() const = 0;

   int i_0;
   const int ci_0;
   static int si_0;
   static const int sci_0;
   mutable int mi_0;
   int ti_0; //!

protected:
   void f_1() {}
   void fc_1() const {}
   virtual void vf_1() {}
   virtual void vfc_1() const {}
   static void sf_1() {}
   virtual void af_1() = 0;
   virtual void afc_1() const = 0;

   int i_1;
   const int ci_1;
   static int si_1;
   static const int sci_1;
   mutable int mi_1;
   int ti_1; //!

public:
   void f_2() {}
   void fc_2() const {}
   virtual void vf_2() {}
   virtual void vfc_2() const {}
   static void sf_2() {}
   virtual void af_2() = 0;
   virtual void afc_2() const = 0;

   int i_2;
   const int ci_2;
   static int si_2;
   static const int sci_2;
   mutable int mi_2;
   int ti_2; //!

   explicit X(int):
      ci_0(0), ci_1(0), ci_2(0)
   {}
   X():
      ci_0(0), ci_1(0), ci_2(0)
   {}

   virtual ~X(){}
};

template<typename T> int X<T>::si_0 = 12;
template<typename T> int X<T>::si_1 = 13;
template<typename T> int X<T>::si_2 = 14;
template<typename T> const int X<T>::sci_0 = 22;
template<typename T> const int X<T>::sci_1 = 23;
template<typename T> const int X<T>::sci_2 = 24;

template class X<float>;
