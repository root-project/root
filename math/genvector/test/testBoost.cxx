#include "Math/BoostX.h"
#include "Math/Boost.h"
#include "Math/VectorUtil.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include <iterator>

using namespace ROOT::Math;

int main() {

  BoostX bx(0.8);
  std::cout << "BoostX - beta : " << bx.Beta() << "  gamma : " << bx.Gamma() << std::endl;

  XYZTVector v(1.,2.,3.,4.);

  XYZTVector vb1 = bx(v);
  XYZTVector vb2 = VectorUtil::boostX(v,0.8);

  if (vb1 != vb2) {
    std::cout << "BoostX test failed" << std::endl;
    int pr = std::cout.precision(18);
    std::cout << vb1 << std::endl;
    std::cout << vb2 << std::endl;
    std::cout.precision (pr);
  }

  //Polar3DVector bv(1.,2.,0.8);
  Polar3DVector bv(0.99999,1.,2);
  std::cout << "BoostVector " << XYZVector(bv) << " beta boost = " << XYZVector(bv).R() << std::endl;
  Boost b(bv);
  std::cout << "Boost Components : ";
  std::ostream_iterator<double> oi(std::cout,"\t");
  b.GetComponents(oi);
  std::cout << std::endl;




  vb1 = b(v);
  vb2 = VectorUtil::boost(v,bv);
  if (vb1 != vb2) {
    std::cout << "Boost test failed" << std::endl;
    int pr = std::cout.precision(18);
    std::cout << vb1 << std::endl;
    std::cout << vb2 << std::endl;
    std::cout.precision (pr);
  }

}
