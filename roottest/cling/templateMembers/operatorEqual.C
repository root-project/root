template<class T = double>
class StThreeVector {
public:
    template<class X> StThreeVector<T>   func(X) { return *this; };
    template<class X> StThreeVector<T>   operator-= (const StThreeVector<X>&) { return *this; };
    template<class X> StThreeVector<T>&  operator+= (const StThreeVector<X>&) { return *this; };
};

#ifdef __MAKECLING__
#pragma link C++ class StThreeVector+;
#pragma link C++ function StThreeVector<double>::func(float);
#pragma link C++ function StThreeVector<double>::operator+=(const StThreeVector<float>&);
#endif
