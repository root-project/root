#include <complex>


class classWithComplex {
public:
classWithComplex(float re, float im, double red, double imd):fcf(re,im),fcd(red,imd){};
classWithComplex():fcf(0,0),fcd(0,0){};
  std::complex<float> GetF() const {return fcf;}
  std::complex<double> GetD() const {return fcd;}

private:
  std::complex<float> fcf;
  std::complex<double> fcd;

};



bool operator==(const classWithComplex& lhs, const classWithComplex& rhs) {
    return lhs.GetF() == rhs.GetF() && lhs.GetD() == rhs.GetD();
}

bool operator!=(const classWithComplex& lhs, const classWithComplex& rhs) {
    return !(lhs == rhs);
}
