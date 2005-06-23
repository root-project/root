// macro to test mathcore GenVector classes 

int ntest = 0; 
int nfail = 0; 
int ok = 0;

void mathcoreGenVector() {


  gSystem->Load("libMathCore");
  using namespace ROOT::Math;
 
  testVector3D();
  testPoint3D();
  testLorentzVector();
  testVectorUtil();

  std::cout << "\n\nNumber of tests " << ntest << " failed = " << nfail << std::endl;
}


int compare( double v1, double v2, const char* name, double Scale = 1.0) {
  ntest = ntest + 1; 

  // numerical double limit for epsilon 
  double eps = Scale* 2.22044604925031308e-16; 
  int iret = 0; 
  double delta = v2 - v1;
  if (delta < 0 ) delta = - delta; 
  if (v1 == 0 || v2 == 0) { 
    if  (delta > eps ) { 
      iret = 1; 
    }
  }
  // skip case v1 or v2 is infinity
  else { 
     double d = v1; 

    if ( v1 < 0) d = -d; 
    // add also case when delta is small by default
    if ( delta/d  > eps && delta > eps ) 
      iret =  1; 
  }

  if (iret == 0) 
    std::cout << ".";
  else { 
    int pr = std::cout.precision (18);
    std::cout << "\nDiscrepancy in " << name << "() : " << v1 << " != " << v2 << " discr = " << int(delta/d/eps) 
              << "   (Allowed discrepancy is " << eps  << ")\n";
    std::cout.precision (pr);
    nfail = nfail + 1;
  }
  return iret; 
}
      

  

int testVector3D() { 


  std::cout << "\n************************************************************************\n " 
	    << " Vector 3D Test" 
	    << "\n************************************************************************\n";


  int ret = 0;

  XYZVector v1(0.01, 0.02, 16);
  //XYZVector v1(1.0, 2.0, 3.0);

//   XYZVector v1(1.0, 2.0, 30.0);

//   double R = sqrt (v1.X()*v1.X() + v1.Y()*v1.Y() + v1.Z()*v1.Z());
//   // this formula in not precise enough
//   //  double Theta = R>0 ? acos ( v1.Z()/r ) : 0;
//   double Rho = sqrt (v1.X()*v1.X() + v1.Y()*v1.Y());
//   double Theta = v1.Z() == 0 || Rho == 0 ? 0 : atan2( Rho, v1.Z() );  
//   double Phi = Rho>0 ? atan2 (v1.Y(), v1.X()) : 0;

  std::cout << "Test Cartesian-Polar :          " ;

  Polar3DVector v2(v1.R(), v1.Theta(), v1.Phi() );

  ok = 0;
  ok+= compare(v1.X(), v2.X(), "x"); 
  ok+= compare(v1.Y(), v2.Y(), "y"); 
  ok+= compare(v1.Z(), v2.Z(), "z"); 
  ok+= compare(v1.Phi(), v2.Phi(), "phi"); 
  ok+= compare(v1.Theta(), v2.Theta(), "theta"); 
  ok+= compare(v1.R(), v2.R(), "r"); 
  ok+= compare(v1.Eta(), v2.Eta(), "eta"); 
  ok+= ok+= compare(v1.Rho(), v2.Rho(), "rho"); 

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Cartesian-Cylindrical :    ";

  RhoEtaPhiVector v3( v1.Rho(), v1.Eta(), v1.Phi() ); 

  ok = 0;
  ok+= compare(v1.X(), v3.X(), "x"); 
  ok+= compare(v1.Y(), v3.Y(), "y"); 
  ok+= compare(v1.Z(), v3.Z(), "z"); 
  ok+= compare(v1.Phi(), v3.Phi(), "phi"); 
  ok+= compare(v1.Theta(), v3.Theta(), "theta"); 
  ok+= compare(v1.R(), v3.R(), "r"); 
  ok+= compare(v1.Eta(), v3.Eta(), "eta"); 
  ok+= compare(v1.Rho(), v3.Rho(), "rho"); 

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Operations :               " ;

  ok = 0;
  double Dot = v1.Dot(v2);
  ok+= compare( Dot, v1.Mag2(),"dot"  );
  XYZVector vcross = v1.Cross(v2);
  ok+= compare( vcross.R(), 0,"cross"  );

  //std::cout << "\nTest Unit & scaling : " ;

  XYZVector vscale1 = v1*10;
  XYZVector vscale2 = vscale1/10;
  ok+= compare( v1.R(), vscale2.R(), "scale");

  XYZVector vu = v1.Unit();
  ok+= compare(v2.Phi(),vu.Phi(),"unit Phi");
  ok+= compare(v2.Theta(),vu.Theta(),"unit Theta");
  ok+= compare(1.0,vu.R(),"unit ");

  XYZVector q1 = v1;
  // RhoEtaPhiVector q2 = v1;  ! copy constructor between different vector does not work yet)  
  RhoEtaPhiVector q2; 
  q2 = v1;
  q2.Coordinates().SetPhi(2.0);
  
  XYZVector q3 = q1 + q2; 
  XYZVector q4 = q3 - q2; 

  ok+= compare( q4.X(), q1.X(), "op X"  );
  ok+= compare( q4.Y(), q1.Y(), "op Y" );
  ok+= compare( q4.Z(), q1.Z(), "op Z" );

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Linear Algebra conversion: " ;

  XYZVector vxyz1(1.,2.,3.); 
  
  TVectorD vla1(3);
  vxyz1.Coordinates().GetCoordinates(vla1.GetMatrixArray() );

  TVectorD vla2(3); 
  vla2[0] = 1.; vla2[1] = -2.; vla2[2] = 1.;

  XYZVector vxyz2; 
  vxyz2.AssignFrom(vla2, 0);

  ok = 0; 
  double prod1 =  vxyz1.Dot(vxyz2); 
  double prod2 = vla1*vla2; 
  ok+= compare( prod1, prod2, "la test" );

  if (ok == 0) std::cout << "\t\t OK " << std::endl;
  

}



int testPoint3D() { 

  std::cout << "\n************************************************************************\n " 
	    << " Point 3D Test" 
	    << "\n************************************************************************\n";


  int ret = 0;

  XYZPoint p1(1.0, 2.0, 3.0);

  std::cout << "Test Cartesian-Polar :          ";

  Polar3DPoint p2(p1.R(), p1.Theta(), p1.Phi() );

  ok = 0;
  ok+= compare(p1.X(), p2.X(), "x"); 
  ok+= compare(p1.Y(), p2.Y(), "y"); 
  ok+= compare(p1.Z(), p2.Z(), "z"); 
  ok+= compare(p1.Phi(), p2.Phi(), "phi"); 
  ok+= compare(p1.Theta(), p2.Theta(), "theta"); 
  ok+= compare(p1.R(), p2.R(), "r"); 
  ok+= compare(p1.Eta(), p2.Eta(), "eta"); 
  ok+= compare(p1.Rho(), p2.Rho(), "rho"); 

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Polar-Cylindrical :        ";

  RhoEtaPhiPoint p3( p2.Rho(), p2.Eta(), p2.Phi() ); 

  ok = 0;
  ok+= compare(p2.X(), p3.X(), "x"); 
  ok+= compare(p2.Y(), p3.Y(), "y"); 
  ok+= compare(p2.Z(), p3.Z(), "z",3); 
  ok+= compare(p2.Phi(), p3.Phi(), "phi"); 
  ok+= compare(p2.Theta(), p3.Theta(), "theta"); 
  ok+= compare(p2.R(), p3.R(), "r"); 
  ok+= compare(p2.Eta(), p3.Eta(), "eta"); 
  ok+= compare(p2.Rho(), p3.Rho(), "rho"); 

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test operations :               ";
  //std::cout << "\nTest Dot and Cross products with Vectors : ";
  Polar3DVector vperp(1.,p1.Theta() + TMath::PiOver2(),p1.Phi() );
  double Dot = p1.Dot(vperp);
  ok+= compare( Dot, 0.0,"dot", 10  );

  XYZPoint vcross = p1.Cross(vperp);
  ok+= compare( vcross.R(), p1.R(),"cross mag"  );
  ok+= compare( vcross.Dot(vperp), 0.0,"cross dir"  );


  //RhoEtaPhiPoint q1 = p1;  ! constructor yet not working in CINT
  RhoEtaPhiPoint q1; q1 = p1; 
  q1.Coordinates().SetEta(2.0);
  Polar3DVector v2(p1.R(), p1.Theta(),p1.Phi()); 

  
  RhoEtaPhiPoint q3 = q1 + v2; 
  // point -point in vector does not work yet
  RhoEtaPhiPoint q4 = q3 - v2; 
  ok+= compare( q4.X(), q1.X(), "PV op X"  );
  ok+= compare( q4.Y(), q1.Y(), "PV op Y" );
  ok+= compare( q4.Z(), q1.Z(), "PV op Z" );

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


//   RhoEtaPhiVector v4 = q3 - q1; 
//   ok+= compare( v4.X(), v2.X(), "op X"  );
//   ok+= compare( v4.Y(), v2.Y(), "op Y" );
//   ok+= compare( v4.Z(), v2.Z(), "op Z" );

}




int testLorentzVector() { 

  std::cout << "\n************************************************************************\n " 
	    << " Loorentz Vector Test" 
	    << "\n************************************************************************\n";


  int ret = 0;

  //LorentzVector v1(0.00001, 0.00001, 30000000000.0);
  LorentzVector v1(1.0, 2.0, 3.0, 4.0);


  std::cout << "Test Cartesian-Cylindrical4D :  ";

  LorentzVectorPtEtaPhiE v2( v1.Rho(), v1.Eta(), v1.Phi(), v1.E() ); 

  ok = 0;
  ok+= compare(v1.X(), v2.X(), "x"); 
  ok+= compare(v1.Y(), v2.Y(), "y"); 
  ok+= compare(v1.Z(), v2.Z(), "z", 2); 
  ok+= compare(v1.E(), v2.E(), "e"); 
  ok+= compare(v1.Phi(), v2.Phi(), "phi"); 
  ok+= compare(v1.Theta(), v2.Theta(), "theta"); 
  ok+= compare(v1.Pt(), v2.Pt(), "pt"); 
  ok+= compare(v1.M(), v2.M(), "mass", 5); 
  ok+= compare(v1.et(), v2.et(), "et"); 
  ok+= compare(v1.Mt(), v2.Mt(), "mt", 3); 

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test operations :               ";
  //std::cout << "\nTest Dot product : " ;

  ok = 0;
  double Dot = v1.Dot(v2);
  ok+= compare( Dot, v1.M2(),"dot" , 10 );

  //std::cout << "\nTest scaling : " ;

  LorentzVector vscale1 = v1*10;
  LorentzVector vscale2 = vscale1/10;
  ok+= compare( v1.M(), vscale2.M(), "scale");


  LorentzVector q1 = v1;
  // RhoEtaPhiVector q2 = v1;  ! copy onstructor between different vector does not work yet)  
  LorentzVectorPtEtaPhiE  q2; 
  q2 = v1;
  q2.Coordinates().SetEta(2.0);
  
  LorentzVector q3 = q1 + q2; 
  LorentzVector q4 = q3 - q2; 

  ok+= compare( q4.X(), q1.X(), "op X"  );
  ok+= compare( q4.Y(), q1.Y(), "op Y" );
  ok+= compare( q4.Z(), q1.Z(), "op Z" );
  ok+= compare( q4.E(), q1.E(), "op E" );

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


}


int testVectorUtil() { 

 
  std::cout << "\n************************************************************************\n " 
	    << " Utility Function Test" 
	    << "\n************************************************************************\n";

  std::cout << "Test Vector utility functions : ";


  XYZVector v1(1.0, 2.0, 3.0); 
  Polar3DVector v2pol(v1.R(), v1.Theta()+TMath::PiOver2(), v1.Phi() + 1.0); 
  // mixedmethods not yet impl. 
  XYZVector v2; v2 = v2pol; 

  ok = 0; 
  ok += compare( VectorUtil::DeltaPhi(v1,v2), 1.0, "deltaPhi Vec");
  
  RhoEtaPhiVector v2cyl(v1.Rho(), v1.Eta() + 1.0, v1.Phi() + 1.0);
  v2 = v2cyl; 


  ok += compare( VectorUtil::DeltaR(v1,v2), sqrt(2.0), "DeltaR Vec");

  XYZVector vperp = v1.Cross(v2);
  ok += compare( VectorUtil::CosTheta(v1,vperp), 0.0, "costheta Vec");
  ok += compare( VectorUtil::Angle(v1,vperp), TMath::PiOver2(), "angle Vec");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


  std::cout << "Test Point utility functions :  ";


  XYZPoint p1(1.0, 2.0, 3.0); 
  Polar3DPoint p2pol(p1.R(), p1.Theta()+TMath::PiOver2(), p1.Phi() + 1.0); 
  // mixedmethods not yet impl. 
  XYZPoint p2; p2 = p2pol; 

  ok = 0; 
  ok += compare( VectorUtil::DeltaPhi(p1,p2), 1.0, "deltaPhi Point");
  
  RhoEtaPhiPoint p2cyl(p1.Rho(), p1.Eta() + 1.0, p1.Phi() + 1.0);
  p2 = p2cyl; 
  ok += compare( VectorUtil::DeltaR(p1,p2), sqrt(2.0), "DeltaR Point");

  XYZPoint pperp(vperp.X(), vperp.Y(), vperp.Z());
  ok += compare( VectorUtil::CosTheta(p1,pperp), 0.0, "costheta Point");
  ok += compare( VectorUtil::Angle(p1,pperp), TMath::PiOver2(), "angle Point");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


  std::cout << "LorentzVector utility funct.:   ";


  LorentzVector q1(1.0, 2.0, 3.0,4.0); 
  LorentzVectorPtEtaPhiE q2cyl(q1.Pt(), q1.Eta()+1.0, q1.Phi() + 1.0, q1.E() ); 
  // mixedmethods not yet impl. 
  LorentzVector q2; q2 = q2cyl; 

  ok = 0; 
  ok += compare( VectorUtil::DeltaPhi(q1,q2), 1.0, "deltaPhi LVec");
  ok += compare( VectorUtil::DeltaR(q1,q2), sqrt(2.0), "DeltaR LVec");
  
  LorentzVector qsum = q1+q2; 
  ok += compare( VectorUtil::InvariantMass(q1,q2), qsum.M(), "InvMass");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


}
