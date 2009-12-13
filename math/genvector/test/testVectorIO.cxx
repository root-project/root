
//
// Cint macro to test I/O of mathcore Lorentz Vectors in a Tree and compare with a 
// TLorentzVector. A ROOT tree is written and read in both using either a XYZTVector or /// a TLorentzVector. 
// 
//  To execute the macro type in: 
//
// root[0]: .x  mathcoreVectorIO.C


#ifdef USE_REFLEX
#include "Cintex/Cintex.h"
#include "Reflex/Reflex.h"
#endif

#include "TRandom3.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"

#include <iostream>

#include "TLorentzVector.h"

#include "Math/Vector4D.h"
#include "Math/Vector3D.h"
#include "Math/Point3D.h"

#include "Track.h"


#define DEBUG


//#define READONLY

//#define USE_REFLEX

#define DEFVECTOR4D(TYPE) \
typedef TYPE AVector4D; \
const std::string vector4d_type = #TYPE ;

#define DEFVECTOR3D(TYPE) \
typedef TYPE AVector3D; \
const std::string vector3d_type = #TYPE ;

#define DEFPOINT3D(TYPE) \
typedef TYPE APoint3D; \
const std::string point3d_type = #TYPE ;



//const double tol = 1.0E-16;
const double tol = 1.0E-6; // or doublr 32 or float

DEFVECTOR4D(ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >);
//DEFVECTOR4D(ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >);

DEFVECTOR3D(ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >); 

DEFPOINT3D(ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> >); 


// DEFVECTOR4D(ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >) 

//using namespace ROOT::Math;

template<class Vector> 
inline double getMag2(const Vector & v) { 
  return v.mag2();
}

// inline double getMag2(const VecTrackD & v) { 
//    // print the read points
//    std::cout << "VecTRackD " << std::endl;
//    for (VecTrackD::It itr = v.begin() ; itr != v.end(); ++itr) 
//       std::cout << (*itr).Pos() << std::endl; 

//   return v.mag2();
// }


inline double getMag2(const TVector3 & v) { 
  return v.Mag2();
}

inline double getMag2(const TLorentzVector & v) { 
  return v.Mag2();
}

template<class U> 
inline void setValues(ROOT::Math::DisplacementVector3D<U> & v, const double x[] ) { 
    v.SetXYZ(x[0],x[1],x[2]); 
}

template<class U> 
inline void setValues(ROOT::Math::PositionVector3D<U> & v, const double x[]) { 
    v.SetXYZ(x[0],x[1],x[2]); 
}

template<class U> 
inline void setValues(ROOT::Math::LorentzVector<U> & v, const double x[]) { 
    v.SetXYZT(x[0],x[1],x[2],x[3]); 
}
// specialization for T -classes 
inline void setValues(TVector3 & v, const double x[]) { 
    v.SetXYZ(x[0],x[1],x[2]); 
}
inline void setValues(TLorentzVector & v, const double x[]) { 
    v.SetXYZT(x[0],x[1],x[2],x[3]); 
}


template <class Vector> 
double testDummy(int n) { 

  TRandom3 R(111);

  TStopwatch timer;

  Vector v1;
  double s = 0; 
  double p[4];

  timer.Start();
  for (int i = 0; i < n; ++i) { 
        p[0] = R.Gaus(0,10);
	p[1] = R.Gaus(0,10);
	p[2] = R.Gaus(0,10);
	p[3] = R.Gaus(100,10);
	setValues(v1,p);
	s += getMag2(v1); 
  }

  timer.Stop();

  double sav = std::sqrt(s/double(n));

  std::cout << " Time for Random gen " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  int pr = std::cout.precision(18);  std::cout << "Average : " << sav << std::endl;   std::cout.precision(pr);

  return sav; 
}

//----------------------------------------------------------------
/// writing
//----------------------------------------------------------------

template<class Vector> 
double write(int n, const std::string & file_name, const std::string & vector_type, int compress = 0) { 

  TStopwatch timer;

  TRandom3 R(111);

  std::cout << "writing a tree with " << vector_type << std::endl; 

  std::string fname = file_name + ".root";
  TFile f1(fname.c_str(),"RECREATE","",compress);

  // create tree
  std::string tree_name="Tree with" + vector_type; 
  TTree t1("t1",tree_name.c_str());

  Vector *v1 = new Vector();
  std::cout << "typeID written : " << typeid(*v1).name() << std::endl;

  t1.Branch("Vector branch",vector_type.c_str(),&v1);

  timer.Start();
  double p[4]; 
  double s = 0; 
  for (int i = 0; i < n; ++i) { 
        p[0] = R.Gaus(0,10);
	p[1] = R.Gaus(0,10);
	p[2] = R.Gaus(0,10);
	p[3] = R.Gaus(100,10);
	//CylindricalEta4D<double> & c = v1->Coordinates();
	//c.SetValues(Px,pY,pZ,E);
	setValues(*v1,p);
	t1.Fill();
	s += getMag2(*v1); 
  }

  f1.Write();
  timer.Stop();

  double sav = std::sqrt(s/double(n));


#ifdef DEBUG
  t1.Print();
  std::cout << " Time for Writing " << file_name << "\t: " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  int pr = std::cout.precision(18);  std::cout << "Average : " << sav << std::endl;   std::cout.precision(pr);
#endif
  
  return sav; 
}


//----------------------------------------------------------------
/// reading
//----------------------------------------------------------------

template<class Vector> 
double read(const std::string & file_name) { 

  TStopwatch timer;

  std::string fname = file_name + ".root";

  TFile f1(fname.c_str());
  if (f1.IsZombie() ) { 
    std::cout << " Error opening file " << file_name << std::endl; 
    return -1; 
  }

  //TFile f1("mathcoreVectorIO_D32.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  Vector *v1 = 0;

  std::cout << "reading typeID  : " << typeid(*v1).name() << std::endl;

  t1->SetBranchAddress("Vector branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double s=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    s += getMag2(*v1);
  }


  timer.Stop();

  double sav = std::sqrt(s/double(n));

#ifdef DEBUG
  std::cout << " Time for Reading " << file_name << "\t: " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  int pr = std::cout.precision(18);  std::cout << "Average : " << sav << std::endl;   std::cout.precision(pr);
#endif

  return sav; 
}

template<class TrackType>
double writeTrack(int n, const std::string & file_name, int compress = 0) { 

  std::cout << "\n";
  std::cout << "  Test writing .." << file_name << " .....\n";
  std::cout << "**************************************************\n";

  TRandom3 R(111);

  TStopwatch timer;

  //std::cout << "writing a tree with " << vector_type << std::endl; 

  std::string fname = file_name + ".root";
  TFile f1(fname.c_str(),"RECREATE","",compress);

  // create tree
  std::string tree_name="Tree with TrackD"; 
  TTree t1("t1",tree_name.c_str());

  TrackType *track = new TrackType();
  std::cout << "typeID written : " << typeid(*track).name() << std::endl;

  
  //t1.Branch("Vector branch",&track,16000,0);
  // default split model
  t1.Branch("Vector branch",&track);

  timer.Start();
  double s = 0; 
  ROOT::Math::XYZTVector q; 
  ROOT::Math::XYZPoint p; 
  typedef typename TrackType::VectorType V;
  typedef typename TrackType::PointType  P;

  for (int i = 0; i < n; ++i) { 
    q.SetXYZT( R.Gaus(0,10),
	       R.Gaus(0,10),
	       R.Gaus(0,10),
	       R.Gaus(100,10) ); 
    p.SetXYZ( q.X(), q.Y(), q.Z() ); 
	
    track->Set( V(q), P(p) ); 
    
    t1.Fill();
    s += getMag2( *track ); 
  }

  f1.Write();
  timer.Stop();

  double sav = std::sqrt(s/double(n));


#ifdef DEBUG
  t1.Print();
  std::cout << " Time for Writing " << file_name << "\t: " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  int pr = std::cout.precision(18);  std::cout << "Average : " << sav << std::endl;   std::cout.precision(pr);
#endif
  
  return sav; 

}



int testResult(double w1, double r1, const std::string & type) { 

  int iret = 0; 
  std::cout << w1 << "  r " << r1 << std::endl; 
  // do like this to avoid nan's
  if (!( fabs(w1-r1)  < tol)) { 
    std::cout << "\nERROR: Differeces found  when reading " << std::endl;
    int pr = std::cout.precision(18);  std::cout << w1 << "   !=    " << r1 << std::endl; std::cout.precision(pr);
    iret = -1;
  }

  std::cout << "\n*********************************************************************************************\n"; 
  std::cout << "Test :\t " << type << "\t\t";
  if (iret ==0) 
    std::cout << "OK" << std::endl; 
  else 
    std::cout << "FAILED" << std::endl; 
  std::cout << "********************************************************************************************\n\n"; 

  return iret; 
}



int testVectorIO(bool readOnly = false) { 

  int iret = 0; 

// #ifdef __CINT__
//   gSystem->Load("libMathCore");  
//   gSystem->Load("libPhysics");  
//   using namespace ROOT::Math;
// #endif

#ifdef USE_REFLEX

// #ifdef __CINT__
//   gSystem->Load("libReflex");  
//   gSystem->Load("libCintex");  
// #endif

  std::cout << "Use Reflex dictionary " << std::endl; 

  gSystem->Load("libReflex");  
  gSystem->Load("libCintex");  

  ROOT::Cintex::Cintex::SetDebug(1);
  ROOT::Cintex::Cintex::Enable();



  //iret |= gSystem->Load("libSmatrixRflx");  
  //iret |= gSystem->Load("libMathAddRflx");  
  //iret |= gSystem->Load("libMathRflx");  
  if (iret |= 0) { 
    std::cerr <<"Failing to Load Reflex dictionaries " << std::endl;
    return iret; 
  }

#endif   // endif on using reflex

  
  int nEvents = 100000;
  //int nEvents = 100;

  double w1, r1 = 0;
  std::string fname; 
  
  testDummy<AVector4D>(nEvents);
 
  fname = "lorentzvector";
  if (readOnly) {
     w1 = 98.995527276968474;
     fname += "_prev";
  }
  else
     w1 = write<AVector4D>(nEvents,fname,vector4d_type);

  r1 = read<AVector4D>(fname);
  iret |= testResult(w1,r1,vector4d_type); 

  fname = "displacementvector";
  if (readOnly) {
     w1 = 17.3281570633214095; 
     fname += "_prev";
  }
  else 
     w1 = write<AVector3D>(nEvents,fname,vector3d_type);

  r1 = read<AVector3D>(fname);
  iret |= testResult(w1,r1,vector3d_type); 

  fname = "positionvector";
  if (readOnly)
     fname += "_prev";
  else 
     w1 = write<APoint3D>(nEvents,fname,point3d_type);
  
  r1 = read<APoint3D>(fname);
  iret |= testResult(w1,r1,point3d_type); 


  // test TrackD
  fname = "track";
  // load track dictionary 
  gSystem->Load("libTrackDict");  

  //nEvents = 10000; 

  if (readOnly) {
     fname += "_prev";
  }
  else 
    w1 = writeTrack<TrackD>( nEvents,fname); 
  
  r1 = read<TrackD>(fname);
  iret |= testResult(w1,r1,"TrackD"); 

  // test TrackD32

  fname = "trackD32";
  // load track dictionary 
  //  gSystem->Load("libTrackDict");  

  if (readOnly) {
     fname += "_prev";
  }
  else 
    w1 = writeTrack<TrackD32>( nEvents,fname); 
  
  r1 = read<TrackD32>(fname);
  iret |= testResult(w1,r1,"TrackD32"); 


  // test vector of tracks
  fname = "vectrack";

  nEvents = 10000; 

  if (readOnly) {
     fname += "_prev";
  }
  else 
    w1 = writeTrack<VecTrackD>( nEvents,fname); 
  
  r1 = read<VecTrackD>(fname);
  iret |= testResult(w1,r1,"VecTrackD"); 

  // test ClusterD (cotaining a vector of points) 
  fname = "cluster";

  nEvents = 10000; 

  if (readOnly) {
     fname += "_prev";
  }
  else 
    w1 = writeTrack<ClusterD>( nEvents,fname); 
  
  r1 = read<ClusterD>(fname);
  iret |= testResult(w1,r1,"ClusterD"); 
  
  return iret; 
}

int main(int argc, char ** ) { 

   bool readOnly = false;
   if (argc > 1) readOnly = true; 

   int iret =  testVectorIO(readOnly);
   if (iret != 0) 
      std::cerr << "testVectorIO:\t FAILED ! " << std::endl;
   else 
      std::cerr << "testVectorIO:\t OK ! " << std::endl;
   
   return iret; 

}
  
