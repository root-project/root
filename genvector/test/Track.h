// dummy track class for testing I/o of matric

#include "Math/Point3D.h"
#include "Math/Vector4D.h"

#include <vector>

#include "TRandom.h"

typedef double Double32_t;



typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >        Vector4D; 
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >    Vector4D32;
//typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >    Vector4D32;

typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >        Point3D; 
typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> >    Point3D32; 



// track class 
class  TrackD { 

public:

   typedef Vector4D VectorType;
   typedef Point3D  PointType;

   TrackD() {}

   TrackD(const Vector4D & q, const Point3D & p) : fVec(q), fPos(p) {}

   const Vector4D & Vec() const { return fVec; }
   const Point3D & Pos() const { return fPos; }

  double mag2() const { 
    return fVec.mag2() + fPos.mag2(); 
  }

  void Set( const Vector4D & q, const Point3D & p) { 
    fVec = q; fPos = p; 
  }

private:

  Vector4D fVec; 
  Point3D  fPos;
   
}; 

// track class based on  of Double32


class  TrackD32 { 

public:

   typedef Vector4D32 VectorType;
   typedef Point3D32  PointType;


   TrackD32() {}

   TrackD32(const Vector4D32 & q, const Point3D32 & p) : fVec(q), fPos(p) {}

   const Vector4D32 & Vec() const { return fVec; }
   const Point3D32 & Pos() const { return fPos; }

  double mag2() const { 
    return fVec.mag2() + fPos.mag2(); 
  }

  void Set( const Vector4D32 & q, const Point3D32 & p) { 
    fVec =  q; 
    fPos =  p; 
  }
private:

  Vector4D32 fVec; 
  Point3D32  fPos;
   
}; 


// class containning a vector of tracks
class VecTrackD {

public: 

   typedef std::vector<TrackD>::const_iterator It;
   typedef Vector4D VectorType;
   typedef Point3D  PointType;

 
  VecTrackD() {}

  It begin() const { return fTrks.begin(); } 
  It end() const  { return fTrks.end(); } 


  double mag2() const { 
    double s = 0; 
    for (unsigned int i = 0; i < fTrks.size() ; ++i) 
      s += fTrks[i].mag2(); 
    
    return s; 
  }

  void Set( const Vector4D & q, const Point3D & p) { 
    int n = (gRandom->Poisson(4) + 1);
    fTrks.clear();
    fTrks.reserve(n);
    for (int i = 0; i < n; ++i) {
      double x,y,z; 
      gRandom->Sphere(x,y,z, p.R() ); 
      fTrks.push_back( TrackD(q,Point3D( x,y,z ))  );
    }  
  }

private:

  std::vector<TrackD>  fTrks;

};

// cluster  class (containing a vector of points)

class  ClusterD { 

public:

   ClusterD() {}

   typedef Vector4D VectorType;
   typedef Point3D  PointType;


   Vector4D & Vec() { return fVec; }
   Point3D & Pos() { return fPos[0]; }

  double mag2() const { 
    double s = fVec.mag2(); 
    for (unsigned int i = 0; i < fPos.size() ; ++i) 
      s += fPos[i].mag2(); 
    return s; 
  }

  void Set( const Vector4D & q, const Point3D & p) { 
    fVec = q; 
    int n = (gRandom->Poisson(4) + 1);
    fPos.clear();
    fPos.reserve(n);
    for (int i = 0; i < n; ++i) {
      double x,y,z; 
      gRandom->Sphere(x,y,z,p.R() ); 
      fPos.push_back( Point3D( x,y,z ) );
    }  
  }

private:

  Vector4D fVec; 
  std::vector<Point3D>  fPos;
   
}; 
