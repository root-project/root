#ifndef ROOT_VECOP_H
#define ROOT_VECOP_H

// generic (2 dim)
template <class V, int Dim>
struct VecOp {

   template <class It>
   static V Create(It &x, It &y, It &, It &)
   {
      return V(*x++, *y++);
   }
   template <class It>
   static void Set(V &v, It &x, It &y, It &, It &)
   {
      v.SetXY(*x++, *y++);
   }

   static double Add(const V &v) { return v.X() + v.Y(); }
   static double Delta(const V &v1, const V &v2)
   {
      double d = ROOT::Math::VectorUtil::DeltaPhi(v1, v2);
      return d * d;
   } // is v2-v1
};
// specialized for 3D
template <class V>
struct VecOp<V, 3> {
   template <class It>
   static V Create(It &x, It &y, It &z, It &)
   {
      return V(*x++, *y++, *z++);
   }
   template <class It>
   static void Set(V &v, It &x, It &y, It &z, It &)
   {
      v.SetXYZ(*x++, *y++, *z++);
   }
   static V Create(double x, double y, double z, double) { return V(x, y, z); }
   static void Set(V &v, double x, double y, double z, double) { v.SetXYZ(x, y, z); }
   static double Add(const V &v) { return v.X() + v.Y() + v.Z(); }
   static double Delta(const V &v1, const V &v2) { return ROOT::Math::VectorUtil::DeltaR2(v1, v2); }
};

// specialized for 4D
template <class V>
struct VecOp<V, 4> {
   template <class It>
   static V Create(It &x, It &y, It &z, It &t)
   {
      return V(*x++, *y++, *z++, *t++);
   }
   template <class It>
   static void Set(V &v, It &x, It &y, It &z, It &t)
   {
      v.SetXYZT(*x++, *y++, *z++, *t++);
   }

   static double Add(const V &v) { return v.X() + v.Y() + v.Z() + v.E(); }
   static double Delta(const V &v1, const V &v2)
   {
      return ROOT::Math::VectorUtil::DeltaR2(v1, v2) + ROOT::Math::VectorUtil::InvariantMass(v1, v2);
   }
};
// specialized for SVector<3>
template <>
struct VecOp<ROOT::Math::SVector<double, 3>, 3> {
   typedef ROOT::Math::SVector<double, 3> V_t;
   template <class It>
   static V_t Create(It &x, It &y, It &z, It &)
   {
      return V_t(*x++, *y++, *z++);
   }

   static double Add(const V_t &v) { return v(0) + v(1) + v(2); }
};
// specialized for SVector<4>
template <>
struct VecOp<ROOT::Math::SVector<double, 4>, 4> {
   typedef ROOT::Math::SVector<double, 4> V_t;
   template <class It>
   static V_t Create(It &x, It &y, It &z, It &t)
   {
      return V_t(*x++, *y++, *z++, *t++);
   }

   static double Add(const V_t &v) { return v(0) + v(1) + v(2) + v(3); }
};

#endif // ROOT_VECOP_H
