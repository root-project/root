## \file
## \ingroup tutorial_math
## \notebook -nodraw
## Example macro testing available methods and operation of the GenVector
## classes. The results are compared and check at the numerical precision
## levels. Some small discrepancy can appear when the macro is executed on
## different architectures where it has been calibrated (Power PC G5) The macro
## is divided in 4 parts:
##    - testVector3D          :  tests of the 3D Vector classes
##    - testPoint3D           :  tests of the 3D Point classes
##    - testLorentzVector     :  tests of the 4D LorentzVector classes
##    - testVectorUtil        :  tests of the utility functions of all the
##    vector classes
##
## To execute the macro type in:
##
## ~~~{.py}
## IP[0]: %run mathcoreGenVector.py
## ~~~
##
## \macro_output
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT
from ROOT import Math 
import ctypes
import sys


#Note: TMath is different than Math library. 
#      Math is more expanded and has been recent developed.
#      TMath is a legacy class.
#      
TMath = ROOT.TMath 
TMatrixD = ROOT.TMatrixD 
TVectorD = ROOT.TVectorD 

#Note: The following classes belong to GenVector but are
#      found in the Math parent module as in:
#      MathClasses = Math.GenVector.MathClasses
AxisAngle = Math.AxisAngle 
Boost = Math.Boost 
BoostX = Math.BoostX 
BoostY = Math.BoostY 
BoostZ = Math.BoostZ 
EulerAngles = Math.EulerAngles 
LorentzRotation = Math.LorentzRotation 
Plane3D = Math.Plane3D 
Quaternion = Math.Quaternion 
Rotation3D = Math.Rotation3D 
RotationX = Math.RotationX 
RotationY = Math.RotationY 
RotationZ = Math.RotationZ 
RotationZYX = Math.RotationZYX 
Transform3D = Math.Transform3D 
VectorUtil = Math.VectorUtil 
#Point3D = Math.Point3D 
#Vector3D = Math.Vector3D 
#Vector4D = Math.Vector4D 
XYZPoint = Math.XYZPoint
XYZVector = Math.XYZVector
XYZTVector = Math.XYZTVector
Polar3DVector = Math.Polar3DVector
Polar3DPoint = Math.Polar3DPoint
RhoEtaPhiVector = Math.RhoEtaPhiVector
RhoEtaPhiPoint = Math.RhoEtaPhiPoint
RhoZPhiVector = Math.RhoZPhiVector
RhoZPhiPoint = Math.RhoZPhiPoint
PtEtaPhiEVector = Math.PtEtaPhiEVector
PtEtaPhiMVector = Math.PtEtaPhiMVector
PxPyPzMVector = Math.PxPyPzMVector

# math operations
sqrt = ROOT.sqrt

# types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
Float_t = ROOT.Float_t
char = ROOT.char
c_double = ctypes.c_double

# std
std = ROOT.std
precision = std.cout.precision

# Seemleasly integration 
ProcessLine = ROOT.gROOT.ProcessLine
Declare = ROOT.gInterpreter.Declare
 
#Note:
# TMatrixTAutoLoadOps isn't properly load in pyroot.
# ROOT.TMatrixTAutoloadOps is just a namespace.
# Alternatively TMatrixTAutoloadOps::Add function template 
# is loaded into pyroot, but it returns always a TVectorT<double> nullptr.
# We are loading the Add function into a Add__1616 function. See line 1616
# of TVectorT.cxx documentation.

# Loading TMatrixTAutoLoadOps::Add(target, scalar, a, source)
ProcessLine("""
      TVectorD Add_1616(TVectorD &target, double scalar,
                        const TMatrixD &a, const TVectorD &source){{
      
         return TMatrixTAutoloadOps::Add(target, scalar, a, source); 
      
      }};
""")
Add_1616 = ROOT.Add_1616


# Loading operator* for two whichever Rotation{X,Y,Z}
# First.
# Defining functions of all possible permutations.
from itertools import permutations
# Linguistic Note: the plural of AXIS is AXIIS or AXES or AXISES
AXIIS = ["X", "Y", "Z"]
for (i,j) in permutations(AXIIS, 2):
   declaration = f""" 
      ROOT::Math::Rotation3D 
      _mul_{i}{j} (ROOT::Math::Rotation{i} &r{i}, ROOT::Math::Rotation{j} &r{j}){{
         return r{i} * r{j};}}
   """
   #Declare(declaration)
   ProcessLine(declaration)
del declaration 
del AXIIS
del permutations

# Second.
# Defining the multiplication operator for Rotations.
def _mul_Rotation(Rotation_i, Rotation_j):

   i_name = Rotation_i.__class__.__name__ 
   j_name = Rotation_j.__class__.__name__ 

   if i_name == j_name :
      return Rotation_i * Rotation_j 

   #TYPES = ["RotationX", "RotationY", "RotationZ"]
   #AXES = ["X", "Y", "Z"]
   i = i_name.strip("Rotation") 
   j = j_name.strip("Rotation") 
   _mul_ij = getattr(ROOT, "_mul_" + i + j )
   return _mul_ij(Rotation_i, Rotation_j)
# Third.
# We will use it in Test LorentzRotation.
      


# FLAGS 
ntest = 0
nfail = 0
ok = 0

# convertion to c_array
def to_c(ls):
   return (c_double * len(ls))(*ls)
def to_py(c_ls):
   return [_ for _ in c_ls]

# cout style print (no new-line unless specified)
def c_print(*x): print(*x, end="")

# fix multiplication * operator 
def mvv(self, other):
   if len(self) != len(other):
      raise TypeError("Both vectors should be of the same size")
   if len(self) == len(other):
      product = 0
      for i in range(len(self)):
         product += self[i] * other[i]
      return product
         
def mvs(self, scalar):
   for i in range(len(self)):
      self[i] = scalar * self[i]   
   return self

def fixmul(self, other):
   
   global mvv, mvs
   # mvv: multiplication of vector by another vector
   if isinstance(other, self.__class__):
      return mvv(self, other)
   # mvs: multiplication of vector by an scalar 
   else:
      return mvs(self,other)

def fixtv2mul():
   ROOT.TVectorD(1,1,1) * ROOT.TVectorD(2,2,2)
   mvv = ROOT.TVectorD.__mul__
   del ROOT.TVectorD.__mul__

   ROOT.TVectorD(1,1,1) * ROOT.TVectorD(2,2,2)
   mvs = ROOT.TVectorD.__mul__
   return mvv, mvs

ROOT.TVectorD.__mul__ = fixmul


# int
def compare(v1 : Double_t, v2 : Double_t, name : char, Scale : Double_t=1.0) :
   global ntest, nfail, ok
   ntest = ntest + 1
   
   # numerical double limit for epsilon
   eps = Scale * 2.22044604925031308e-16
   iret = 0
   delta = v2 - v1
   d = 0
   if (delta < 0):
      delta = -delta
   if (v1 == 0 or v2 == 0):
      if delta > eps:
         iret = 1
         
      
   # skip case v1 or v2 is infinity
   else:
      d = v1
      
      if (v1 < 0):
         d = -d
      # add also case when delta is small by default
      if (delta / d > eps and delta > eps):
         iret = 1
   
   if (iret == 0):
      c_print(".")
   else:
      pr = std.cout.precision(18)
      discr = int() 
      if (d != 0):
         discr = int(delta / d / eps)
      else:
         discr = int(delta / eps)
      
      c_print("\nDiscrepancy in " , name , "() : " , v1 , " != " , v2)
      c_print(" discr = " , discr , "   (Allowed discrepancy is " , eps)
      c_print(")\n" )
      std.cout.precision(pr)
      nfail = nfail + 1
      
   return iret
   

# int
def testVector3D() :
   c_print("*************************************************************\n")
   c_print(" Vector 3D Test\n")
   c_print("*************************************************************\n")

   v1 = XYZVector(0.01, 0.02, 16) 
   
   c_print("Test Cartesian-Polar :          ")
   
   v2 = Polar3DVector(v1.R(), v1.Theta(), v1.Phi())
   
   ok = 0
   ok += compare(v1.X(), v2.X(), "x")
   ok += compare(v1.Y(), v2.Y(), "y")
   ok += compare(v1.Z(), v2.Z(), "z")
   ok += compare(v1.Phi(), v2.Phi(), "phi")
   ok += compare(v1.Theta(), v2.Theta(), "theta")
   ok += compare(v1.R(), v2.R(), "r")
   ok += compare(v1.Eta(), v2.Eta(), "eta")
   ok += compare(v1.Rho(), v2.Rho(), "rho")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test Cartesian-CylindricalEta : ")
   
   v3 = RhoEtaPhiVector(v1.Rho(), v1.Eta(), v1.Phi())
   
   ok = 0
   ok += compare(v1.X(), v3.X(), "x")
   ok += compare(v1.Y(), v3.Y(), "y")
   ok += compare(v1.Z(), v3.Z(), "z")
   ok += compare(v1.Phi(), v3.Phi(), "phi")
   ok += compare(v1.Theta(), v3.Theta(), "theta")
   ok += compare(v1.R(), v3.R(), "r")
   ok += compare(v1.Eta(), v3.Eta(), "eta")
   ok += compare(v1.Rho(), v3.Rho(), "rho")
   
   if (ok == 0):
      c_print("\t OK ", "\n")
   
   c_print("Test Cartesian-Cylindrical :    ")
   
   v4 = RhoZPhiVector(v1.Rho(), v1.Z(), v1.Phi())
   
   ok = 0
   ok += compare(v1.X(), v4.X(), "x")
   ok += compare(v1.Y(), v4.Y(), "y")
   ok += compare(v1.Z(), v4.Z(), "z")
   ok += compare(v1.Phi(), v4.Phi(), "phi")
   ok += compare(v1.Theta(), v4.Theta(), "theta")
   ok += compare(v1.R(), v4.R(), "r")
   ok += compare(v1.Eta(), v4.Eta(), "eta")
   ok += compare(v1.Rho(), v4.Rho(), "rho")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test Operations :               ")
   
   ok = 0
   Dot = v1.Dot(v2)
   ok += compare(Dot, v1.Mag2(), "dot")
   vcross = v1.Cross(v2)
   ok += compare(vcross.R(), 0, "cross")
   
   vscale1 = v1 * 10
   vscale2 = vscale1 / 10
   ok += compare(v1.R(), vscale2.R(), "scale")
   
   vu = v1.Unit()
   ok += compare(v2.Phi(), vu.Phi(), "unit Phi")
   ok += compare(v2.Theta(), vu.Theta(), "unit Theta")
   ok += compare(1.0, vu.R(), "unit ")
   
   q1 = v1
   q2 = RhoEtaPhiVector(1.0, 1.0, 1.0)
   
   q3 = q1 + q2
   q4 = q3 - q2
   
   ok += compare(q4.X(), q1.X(), "op X")
   ok += compare(q4.Y(), q1.Y(), "op Y")
   ok += compare(q4.Z(), q1.Z(), "op Z")
   
   # test operator ==
   w1 = v1
   w2 = v2
   w3 = v3
   w4 = v4
   #BP: static_cast<double>(True)
   ok += compare(w1 == v1, Double_t(True), "== XYZ")
   ok += compare(w2 == v2, Double_t(True), "== Polar")
   ok += compare(w3 == v3, Double_t(True), "== RhoEtaPhi")
   ok += compare(w4 == v4, Double_t(True), "== RhoZPhi")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   # test setters
   c_print(f"Test Setters :                  ")
   
   q2.SetXYZ(q1.X(), q1.Y(), q1.Z())
   
   ok += compare(q2.X(), q1.X(), "setXYZ X")
   ok += compare(q2.Y(), q1.Y(), "setXYZ Y")
   ok += compare(q2.Z(), q1.Z(), "setXYZ Z")
   
   q2.SetCoordinates(2.0 * q1.Rho(), q1.Eta(), q1.Phi())

   #BP: q1s = 2.0 * q1
   # Note: Left-hand multiplication not implemented.
   #       ( ) * q1  
   #       Right-hand multiplication is implemented.
   #       q1 * ( )
   #       To avoid confusion use .__mul__ method.
   # Both expressions are equivalent.
   #q1s = q1.__mul__(2.0)
   q1s = q1 * 2.0

   ok += compare(q2.X(), q1s.X(), "set X")
   ok += compare(q2.Y(), q1s.Y(), "set Y")
   ok += compare(q2.Z(), q1s.Z(), "set Z")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   
   c_print(f"Test Linear Algebra conversion: ")
   
   vxyz1 = XYZVector(1., 2., 3.)
   
   vla1 = TVectorD(3)
   vxyz1.Coordinates().GetCoordinates(vla1.GetMatrixArray())
   
   vla2 = TVectorD(3)
   vla2[0] = 1.
   vla2[1] = -2.
   vla2[2] = 1.
   
   vxyz2 = XYZVector() 
   vxyz2.SetCoordinates(vla2[0], vla2[1], vla2[2])
   
   ok = 0
   prod1 = vxyz1.Dot(vxyz2)
   prod2 = vla1 * vla2
   ok += compare(prod1, prod2, "la test")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")

   return ok
   

# int
def testPoint3D() :
   c_print(f"\n*************************************************************")
   c_print("***********\n ")
   c_print(" Point 3D Tests")
   c_print("\n*************************************************************")
   c_print("***********\n")
   
   p1 = XYZPoint(1.0, 2.0, 3.0)
   
   c_print(f"Test Cartesian-Polar :          ")
   
   p2 = Polar3DPoint(p1.R(), p1.Theta(), p1.Phi())
   
   ok = 0
   ok += compare(p1.x(), p2.X(), "x")
   ok += compare(p1.y(), p2.Y(), "y")
   ok += compare(p1.z(), p2.Z(), "z")
   ok += compare(p1.phi(), p2.Phi(), "phi")
   ok += compare(p1.theta(), p2.Theta(), "theta")
   ok += compare(p1.r(), p2.R(), "r")
   ok += compare(p1.eta(), p2.Eta(), "eta")
   ok += compare(p1.rho(), p2.Rho(), "rho")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test Polar-CylindricalEta :     ")
   
   p3 = RhoEtaPhiPoint(p2.Rho(), p2.Eta(), p2.Phi())
   
   ok = 0
   ok += compare(p2.X(), p3.X(), "x")
   ok += compare(p2.Y(), p3.Y(), "y")
   ok += compare(p2.Z(), p3.Z(), "z", 3)
   ok += compare(p2.Phi(), p3.Phi(), "phi")
   ok += compare(p2.Theta(), p3.Theta(), "theta")
   ok += compare(p2.R(), p3.R(), "r")
   ok += compare(p2.Eta(), p3.Eta(), "eta")
   ok += compare(p2.Rho(), p3.Rho(), "rho")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test Operations :               ")
   
   # c_print( "\nTest Dot and Cross products with Vectors : ")
   vperp = Polar3DVector(1., p1.Theta() + TMath.PiOver2(), p1.Phi())
   Dot = p1.Dot(vperp)
   ok += compare(Dot, 0.0, "dot", 10)
   
   vcross = p1.Cross(vperp)
   ok += compare(vcross.R(), p1.R(), "cross mag")
   ok += compare(vcross.Dot(vperp), 0.0, "cross dir")
   
   #BP: pscale1 = 10 * p1
   # Note: Left-hand multiplication not implemented.
   #Not to use: pscale1 = 10 * p1 # 
   pscale1 = p1 * 10 
   pscale2 = pscale1 / 10
   ok += compare(p1.R(), pscale2.R(), "scale")
   
   # test operator ==
   #BP: static_cast<double> 
   ok += compare(p1 == pscale2, Double_t(True), "== Point")
   
   # RhoEtaPhiPoint q1 = p1;  ! constructor yet not working in CLING
   q1 = RhoEtaPhiPoint()
   q1 = p1
   q1.SetCoordinates(p1.Rho(), 2.0, p1.Phi())
   
   v2 = Polar3DVector(p1.R(), p1.Theta(), p1.Phi())
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   
   return ok
   

# int
def testLorentzVector() :
   c_print(f"\n*************************************************************")
   c_print("***********\n ")
   c_print(" Lorentz Vector Tests")
   c_print("\n*************************************************************")
   c_print("***********\n")
   
   v1 = XYZTVector(1.0, 2.0, 3.0, 4.0)
   
   c_print(f"Test XYZT - PtEtaPhiE Vectors:  ")
   
   v2 = PtEtaPhiEVector(v1.Rho(), v1.Eta(), v1.Phi(), v1.E())
   
   ok = 0
   ok += compare(v1.Px(), v2.X(), "x")
   ok += compare(v1.Py(), v2.Y(), "y")
   ok += compare(v1.Pz(), v2.Z(), "z", 2)
   ok += compare(v1.E(), v2.T(), "e")
   ok += compare(v1.Phi(), v2.Phi(), "phi")
   ok += compare(v1.Theta(), v2.Theta(), "theta")
   ok += compare(v1.Pt(), v2.Pt(), "pt")
   ok += compare(v1.M(), v2.M(), "mass", 5)
   ok += compare(v1.Et(), v2.Et(), "et")
   ok += compare(v1.Mt(), v2.Mt(), "mt", 3)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test XYZT - PtEtaPhiM Vectors:  ")
   
   v3 = PtEtaPhiMVector(v1.Rho(), v1.Eta(), v1.Phi(), v1.M())
   
   ok = 0
   ok += compare(v1.Px(), v3.X(), "x")
   ok += compare(v1.Py(), v3.Y(), "y")
   ok += compare(v1.Pz(), v3.Z(), "z", 2)
   ok += compare(v1.E(), v3.T(), "e")
   ok += compare(v1.Phi(), v3.Phi(), "phi")
   ok += compare(v1.Theta(), v3.Theta(), "theta")
   ok += compare(v1.Pt(), v3.Pt(), "pt")
   ok += compare(v1.M(), v3.M(), "mass", 5)
   ok += compare(v1.Et(), v3.Et(), "et")
   ok += compare(v1.Mt(), v3.Mt(), "mt", 3)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test PtEtaPhiE - PxPyPzM Vect.: ")
   
   v4 = PxPyPzMVector(v3.X(), v3.Y(), v3.Z(), v3.M())
   
   ok = 0
   ok += compare(v4.Px(), v3.X(), "x")
   ok += compare(v4.Py(), v3.Y(), "y")
   ok += compare(v4.Pz(), v3.Z(), "z", 2)
   ok += compare(v4.E(), v3.T(), "e")
   ok += compare(v4.Phi(), v3.Phi(), "phi")
   ok += compare(v4.Theta(), v3.Theta(), "theta")
   ok += compare(v4.Pt(), v3.Pt(), "pt")
   ok += compare(v4.M(), v3.M(), "mass", 5)
   ok += compare(v4.Et(), v3.Et(), "et")
   ok += compare(v4.Mt(), v3.Mt(), "mt", 3)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   c_print(f"Test Operations :               ")
   
   ok = 0
   Dot = v1.Dot(v2)
   ok += compare(Dot, v1.M2(), "dot", 10)
   
   vscale1 = v1 * 10
   vscale2 = vscale1 / 10
   ok += compare(v1.M(), vscale2.M(), "scale")
   
   q1 = v1
   q2 = PtEtaPhiEVector(1.0, 1.0, 1.0, 5.0)
   
   q3 = q1 + q2
   q4 = q3 - q2
   
   ok += compare(q4.x(), q1.X(), "op X")
   ok += compare(q4.y(), q1.Y(), "op Y")
   ok += compare(q4.z(), q1.Z(), "op Z")
   ok += compare(q4.t(), q1.E(), "op E")
   
   # test operator ==
   w1 = v1
   w2 = v2
   w3 = v3
   w4 = v4
   #BP : static_cast<double>
   ok += compare(w1 == v1, Double_t(True), "== PxPyPzE")
   ok += compare(w2 == v2, Double_t(True), "== PtEtaPhiE")
   ok += compare(w3 == v3, Double_t(True), "== PtEtaPhiM")
   ok += compare(w4 == v4, Double_t(True), "== PxPyPzM")
   
   # test gamma beta and boost
   b = q1.BoostToCM()
   beta = q1.Beta()
   gamma = q1.Gamma()
   
   ok += compare(b.R(), beta, "beta")
   ok += compare(gamma, 1. / sqrt(1 - beta * beta), "gamma")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   # test setters
   c_print(f"Test Setters :                  ")
   
   q2.SetXYZT(q1.Px(), q1.Py(), q1.Pz(), q1.E())
   
   ok += compare(q2.X(), q1.X(), "setXYZT X")
   ok += compare(q2.Y(), q1.Y(), "setXYZT Y")
   ok += compare(q2.Z(), q1.Z(), "setXYZT Z", 2)
   ok += compare(q2.T(), q1.E(), "setXYZT E")
   
   q2.SetCoordinates(2.0 * q1.Rho(), q1.Eta(), q1.Phi(), 2.0 * q1.E())
   q1s = q1 * 2.0
   ok += compare(q2.X(), q1s.X(), "set X")
   ok += compare(q2.Y(), q1s.Y(), "set Y")
   ok += compare(q2.Z(), q1s.Z(), "set Z", 2)
   ok += compare(q2.T(), q1s.T(), "set E")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   
   return ok
   

# int
def testVectorUtil() :
   c_print(f"\n*************************************************************")
   c_print("***********\n ")
   c_print(" Utility Function Tests")
   c_print("\n*************************************************************")
   c_print("***********\n")
   
   c_print(f"Test Vector utility functions : ")
   
   v1 = XYZVector(1.0, 2.0, 3.0)
   v2pol = Polar3DVector(v1.R(), v1.Theta() + TMath.PiOver2(), v1.Phi() + 1.0)
   # mixedmethods not implemented yet.
   v2 = XYZVector()
   v2 = v2pol
   
   ok = 0
   ok += compare(VectorUtil.DeltaPhi(v1, v2), 1.0, "deltaPhi Vec")
   
   v2cyl = RhoEtaPhiVector(v1.Rho(), v1.Eta() + 1.0, v1.Phi() + 1.0)
   v2 = v2cyl
   
   ok += compare(VectorUtil.DeltaR(v1, v2), sqrt(2.0), "DeltaR Vec")
   
   vperp = v1.Cross(v2)
   ok += compare(VectorUtil.CosTheta(v1, vperp), 0.0, "costheta Vec")
   ok += compare(VectorUtil.Angle(v1, vperp), TMath.PiOver2(), "angle Vec")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   
   c_print(f"Test Point utility functions :  ")
   
   p1 = XYZPoint(1.0, 2.0, 3.0)
   p2pol = Polar3DPoint(p1.R(), p1.Theta() + TMath.PiOver2(), p1.Phi() + 1.0)
   # mixedmethods not implemented yet.
   p2 = XYZPoint()
   p2 = p2pol
   
   ok = 0
   ok += compare(VectorUtil.DeltaPhi(p1, p2), 1.0, "deltaPhi Point")
   
   p2cyl = RhoEtaPhiPoint(p1.Rho(), p1.Eta() + 1.0, p1.Phi() + 1.0)
   p2 = p2cyl
   ok += compare(VectorUtil.DeltaR(p1, p2), sqrt(2.0), "DeltaR Point")
   
   pperp = XYZPoint(vperp.X(), vperp.Y(), vperp.Z())
   ok += compare(VectorUtil.CosTheta(p1, pperp), 0.0, "costheta Point")
   ok += compare(VectorUtil.Angle(p1, pperp), TMath.PiOver2(), "angle Point")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   
   c_print(f"LorentzVector utility funct.:   ")
   
   q1 = XYZTVector(1.0, 2.0, 3.0, 4.0)
   q2cyl = PtEtaPhiEVector(q1.Pt(), q1.Eta() + 1.0, q1.Phi() + 1.0, q1.E())
   q2 = XYZTVector() 
   q2 = q2cyl
   
   ok = 0
   ok += compare(VectorUtil.DeltaPhi(q1, q2), 1.0, "deltaPhi LVec")
   ok += compare(VectorUtil.DeltaR(q1, q2), sqrt(2.0), "DeltaR LVec")
   
   qsum = q1 + q2
   ok += compare(VectorUtil.InvariantMass(q1, q2), qsum.M(), "InvMass")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   
   return ok
   
# fixing multiplication in Rotation3D, ...
#class EulerAngles_Py(EulerAngles):
class EulerAngles_Py:
   def __init__(self):
      super(EulerAngles, self).__init__()
   def __mul__(self, other):
      if isinstance(other, XYZPoint):
         tmp = other.__class__(other)
         tmp *= self
      else:
         tmp = self.__class__(self)
         tmp *= other
      return tmp
         
#class Rotation3D
#class Quaternion
#class AxisAngle
#class RotationZYX

#sys.exit()
# int
def testRotation() :
   c_print(f"\n*************************************************************")
   c_print("***********\n ")
   c_print(" Rotation and Transformation Tests")
   c_print("\n*************************************************************")
   c_print("***********\n")
   
   c_print(f"Test Vector Rotations :         ")
   ok = 0
   
   v = XYZPoint(1., 2, 3.)
   
   pi = TMath.Pi()
   # initiate rotation with some non-trivial angles to test all matrix
   r1 = EulerAngles(pi / 2., pi / 4., pi / 3)
   r2 = Rotation3D(r1)
   # only operator= is in CLING for the other rotations
   r3 = Quaternion(r2)
   r4 = AxisAngle(r3)
   r5 = RotationZYX(r2)
   global gv
   gv = v
   global gv1, gr1
   
   gr1=r1
   global gv2, gr2
   
   gr2=r2
   global gv3, gr3
   
   gr3=r3
   global gv4, gr4
   
   gr4=r4
   global gv5, gr5
   
   gr5=r5
     
   ##sys.exit()
   # Not to use:
   #try :
   #   v1 = r1 * v # XYZPoint
   #   v2 = r2 * v # XYZPoint
   #   v3 = r3 * v # XYZPoint
   #   v4 = r4 * v # XYZPoint
   #   v5 = r5 * v # XYZPoint
   #except:
   #   print("C++ and python did not integrate well.")
   #
   # A rotation is applied to a vector using 'operator()' 
   v1 = r1 ( v ) # XYZPoint
   v2 = r2 ( v ) # XYZPoint
   v3 = r3 ( v ) # XYZPoint
   v4 = r4 ( v ) # XYZPoint
   v5 = r5 ( v ) # XYZPoint
   
   ok += compare(v1.X(), v2.X(), "x", 2)
   ok += compare(v1.Y(), v2.Y(), "y", 2)
   ok += compare(v1.Z(), v2.Z(), "z", 2)
   
   ok += compare(v1.X(), v3.X(), "x", 2)
   ok += compare(v1.Y(), v3.Y(), "y", 2)
   ok += compare(v1.Z(), v3.Z(), "z", 2)
   
   ok += compare(v1.X(), v4.X(), "x", 5)
   ok += compare(v1.Y(), v4.Y(), "y", 5)
   ok += compare(v1.Z(), v4.Z(), "z", 5)
   
   ok += compare(v1.X(), v5.X(), "x", 2)
   ok += compare(v1.Y(), v5.Y(), "y", 2)
   ok += compare(v1.Z(), v5.Z(), "z", 2)
    
   # test with matrix
   #DOING
   rdata = [ Double_t() for _ in range(9)]
   c_rdata = (c_double*len(rdata))(*rdata)
   #BP
   r2.GetComponents(c_rdata)
   # BP GetComponents.
   m = TMatrixD(3, 3, c_rdata)
   vdata = [ Double_t() for _ in range(3)]
   c_vdata = (c_double*len(vdata))(*vdata)
   v.GetCoordinates(c_vdata)
   q = TVectorD(3, c_vdata)
   # BP * 
   global gm
   gm = m
   global gq
   gq = q

   # _mul_TMatrixD_TVectorD
   def _mul_(matrix, vector_source):
      """
      https://root.cern/doc/master/TVectorT_8cxx_source.html#l01434
      """
      target = TVectorD(matrix.GetRowLwb(), matrix.GetRowUpb()) 

      #if you apply once, it gives: return matrix*vector.
      #if you apply n-times, it adds the product matrix*vector to the target.
      #if you change the value 1.(scalar), it scales the result.
      return Add_1616(target, 1., matrix, vector_source) 
      
      # Not to use:
      #the_target = ROOT.Add(target, Double_t(1.0), matrix, vector_source) 
      #return the_target # It returns a nullptr to TVectorD 
       
   #Not to use:
   #q2 = m * q
   q2 = _mul_(m, q) # instead 

   v6 = XYZPoint() 
   v6.SetCoordinates(q2.GetMatrixArray())
   
   #BP: simple fix
   #v1 = v6 = v   
   ok += compare(v1.X(), v6.X(), "x")
   ok += compare(v1.Y(), v6.Y(), "y")
   ok += compare(v1.Z(), v6.Z(), "z")
   

   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")
   
   c_print(f"Test Axial Rotations :          ")
   ok = 0
   
   rx = RotationX(pi / 3)
   ry = RotationY(pi / 4)
   rz = RotationZ(4 * pi / 5)
   
   r3x = Rotation3D(rx)
   r3y = Rotation3D(ry)
   r3z = Rotation3D(rz)
   
   qx = Quaternion(rx)
   qy = Quaternion(ry)
   qz = Quaternion(rz)
   
   rzyx = RotationZYX(rz.Angle(), ry.Angle(), rx.Angle())
   
   global grx, gry, grz
   grx = rx
   gry = ry
   grz = rz
   
   #BP:
   # Not to use:
   #vrot1 = rx * ry * rz * v     # XYZPoint
   #vrot2 = r3x * r3y * r3z * v  # XYZPoint
   # Not to use:
   #vrot1 = rx *( ry *( rz *( v ) ) )     # XYZPoint
   #vrot2 = r3x *( r3y *( r3z *( v ) ) )  # XYZPoint
   # instead
   vrot1 = rx ( ry ( rz ( v ) ) )     # XYZPoint
   vrot2 = r3x ( r3y ( r3z ( v ) ) )  # XYZPoint


   ok += compare(vrot1.X(), vrot2.X(), "x")
   ok += compare(vrot1.Y(), vrot2.Y(), "y")
   ok += compare(vrot1.Z(), vrot2.Z(), "z")
   
   #BP:
   # Not to use: 
   #vrot2 = qx * qy * qz * v # XYZPoint
   # instead
   vrot2 = qx ( qy ( qz ( v ) ) )# XYZPoint
   
   ok += compare(vrot1.X(), vrot2.X(), "x", 2)
   ok += compare(vrot1.Y(), vrot2.Y(), "y", 2)
   ok += compare(vrot1.Z(), vrot2.Z(), "z", 2)

   #BP:
   #Not to use:
   #vrot2 = rzyx * v # XYZPoint
   #instead:
   vrot2 = rzyx ( v ) # XYZPoint
   
   ok += compare(vrot1.X(), vrot2.X(), "x")
   ok += compare(vrot1.Y(), vrot2.Y(), "y")
   ok += compare(vrot1.Z(), vrot2.Z(), "z")
   
   # now inverse (first x then y then z)
   #BP:
   #Not to use:
   # vrot1 = rz * ry * rx * v
   # vrot2 = r3z * r3y * r3x * v
   vrot1 = rz ( ry ( rx ( v ) ) )
   vrot2 = r3z ( r3y ( r3x ( v ) ) )
   
   ok += compare(vrot1.X(), vrot2.X(), "x")
   ok += compare(vrot1.Y(), vrot2.Y(), "y")
   ok += compare(vrot1.Z(), vrot2.Z(), "z")
   
   #BP:
   #Not to use:
   # vinv1 = rx.Inverse() * ry.Inverse() * rz.Inverse() * vrot1
   # instead:
   vinv1 = rx.Inverse() *( ry.Inverse() *( rz.Inverse() *( vrot1 ) ) ) # XYZPoint
   # Note: 
   # vrot1 is XYZPoint
   global gvrot1
   gvrot1 = vrot1 
   
   ok += compare(vinv1.X(), v.X(), "x", 2)
   ok += compare(vinv1.Y(), v.Y(), "y", 2) # C++-version omits 2
   ok += compare(vinv1.Z(), v.Z(), "z", 2) # C++-version omits 2
   #Note:
   #     About the omision of paramater "2" above.
   #     If you omit "2", you get error: 
   #     Discrepancy in  z () :  3.000000000000001  !=  3.0 discr =  1    (Allowed discrepancy is  2.220446049250313e-16)
   #     Remeber that 2 is the scale in delta. 
 
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")
   
   c_print(f"Test Rotations by a PI angle :  ")
   ok = 0
   
   # Note: To initialize an AxisAngle-object we have to:
   #       -use the default constructor
   #            AxisAngle()
   #       -define our c_array using ctypes
   #          import ctypes
   #          our_array = [6, 8, 10, 3.14]
   #          c_our_array = (c_double * 4) (*our_array)
   #       -set components using the SetComponents template-proxy 
   #          .SetComponents[" double * "]
   #       - ...while calling it the method-function 
   #          .SetComponents[" double * "](c_our_array, c_our_array)
   #       Why?
   #       Check out the C++ documentation at line-78: 
   #       https://root.cern.ch/doc/master/GenVector_2AxisAngle_8h_source.html
   #       Inside, we need to define `begin` and `end` of a `double *` array.
   #       Thus, in python, we have to reapeat the begin and end with the same variable.
   #       If you are interested in check-out its initialization, you can print :
   #       .Angle() 
   #       .Axis().X() 
   #       .Axis().Y() 
   #       .Axis().Z() 
   #
   
   b = [6, 8, 10, 3.14159265358979323]
   #BP arPi = AxisAngle(b, b + 4)
   #Not to use: arPi = AxisAngle(b, b + 4)
   c_b = to_c(b)      # step1
   arPi = AxisAngle() # step3
   arPi.SetComponents["double*"](c_b, c_b) 
   # arPi.Angle()  
   # arPi_Axis = arPi.Axis()  
   # arPi_Axis.X() 
   # print(arPi_Axis.X()) 
   # print(arPi_Axis.Y()) 
   # print(arPi_Axis.Z()) 

   rPi = Rotation3D(arPi)
   a1 = AxisAngle(rPi)
   ok += compare(arPi.Axis().X(), a1.Axis().X(), "x")
   ok += compare(arPi.Axis().Y(), a1.Axis().Y(), "y")
   ok += compare(arPi.Axis().Z(), a1.Axis().Z(), "z")
   ok += compare(arPi.Angle(), a1.Angle(), "angle")
   
   #ePi = EulerAngles()
   ePi = EulerAngles(rPi)
   #e1 = EulerAngles()
   e1 = EulerAngles(Rotation3D(a1))
   global gePi, ge1
   gePi = ePi
   ge1 = e1
   ok += compare(ePi.Phi(), e1.Phi(), "phi")
   ok += compare(ePi.Theta(), e1.Theta(), "theta")
   ok += compare(ePi.Psi(), e1.Psi(), "ps1")
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   else:
      c_print(f"")
   c_print(f"Test Inversions :               ")
   ok = 0
   
   s1 = r1.Inverse()
   s2 = r2.Inverse()
   s3 = r3.Inverse()
   s4 = r4.Inverse()
   s5 = r5.Inverse()
   global gs2
   gs2 = s2
  
   # Euler angles not impl. yet
   #BP:
   #Not to use:
   #p = s2 * r2 * v
   #p = s2 *( r2 *( v ) )
   #Note:
   #      r2*(v) doesn't function properly the first time
   #      error: Cannot compile this scalar expression yet
   #      Math/GenVector/Rotation3D.h:134:51: 
   #      explicit Rotation3D(const ForeignMatrix & m) { SetComponents(m); }
   #instead:
   p = s2 ( r2 ( v ) )
   
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   #BP: p = s3 * r3 * v
   #Not to use:
   #p = s3 * r3 * v
   #p = s3 *( r3 *( v ) )
   #instead:
   p = s3 ( r3 ( v ) )
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   #BP: p = s4 * r4 * v
   #BP: p = s4 *( r4 *( v ) )
   p = s4 ( r4 ( v ) )
   # axis angle inversion not very precise
   ok += compare(p.X(), v.X(), "x", 1E9)
   ok += compare(p.Y(), v.Y(), "y", 1E9)
   ok += compare(p.Z(), v.Z(), "z", 1E9)
   
   #BP: p = s5 * r5 * v
   #BP: p = s5 *( r5 *( v ) )
   p = s5 ( r5 ( v ) )
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   r6 = Rotation3D(r5)
   s6 = r6.Inverse()
   
   #BP
   #p = s6 * r6 * v
   #BP: p = s6 *( r6 *( v ) )
   p = s5 ( r5 ( v ) )
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")
   
   # test Rectify
   c_print(f"Test rectify :                  ")
   ok = 0
   
   u1 = XYZVector(0.999498, -0.00118212, -0.0316611)
   u2 = XYZVector(0, 0.999304, -0.0373108)
   u3 = XYZVector(0.0316832, 0.0372921, 0.998802)
   rr = Rotation3D(u1, u2, u3)
   # check orto-normality
   #BP:
   #vrr = rr * v
   vrr = rr ( v )
   ok += compare(v.R(), vrr.R(), "R", 1.E9)
   
   if (ok == 0):
      c_print(f"\t\t OK ", "\n")
   else:
      c_print(f"")
   
   c_print(f"Test Transform3D :              ")
   ok = 0
   
   d = XYZVector(1., -2., 3.)
   global gd 
   gd = d
   t = Transform3D(r2, d)
   
   #BP: pd = t * v
   pd = t ( v )
   # apply directly rotation
   #vd = r2 * v + d
   vd = r2 ( v ) + d
   
   ok += compare(pd.X(), vd.X(), "x")
   ok += compare(pd.Y(), vd.Y(), "y")
   ok += compare(pd.Z(), vd.Z(), "z")
   
   # test with matrix
   tdata = [Double_t() for _ in range(12)]
   c_tdata = to_c(tdata)
   t.GetComponents(c_tdata)
   mt = TMatrixD(3, 4, c_tdata)
   ## needs a vector of dim 4
   vData = [Double_t() for _ in range(4)]
   c_vData = to_c(vData)
   v.GetCoordinates(c_vData)
   c_vData[3] = 1.
   q0 = TVectorD(4, c_vData)
   
   global gmt
   gmt = mt
   global gq0 
   gq0 = q0
   
   #BP: 
   #Not to use:
   # qt = mt * q0
   #instead:
   qt = _mul_(mt,q0)
   global gqt
   gqt = qt

   
   ok += compare(pd.X(), qt(0), "x")
   ok += compare(pd.Y(), qt(1), "y")
   ok += compare(pd.Z(), qt(2), "z")
   
   # test inverse
   
   tinv = t.Inverse()
   
   #BP: p = tinv * t * v
   p = tinv ( t ( v ) )
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   # test construct inverse from translation first
   #BP:
   #Not to use:
   #r2.Inverse() * (-d))
   #instead:
   #r2.Inverse() (-d)
   # in:
   tinv2 = Transform3D(r2.Inverse(), r2.Inverse() (-d))
   #BP: p = tinv2 * t * v
   p = tinv2 ( t ( v ) )
   
   ok += compare(p.X(), v.X(), "x", 10)
   ok += compare(p.Y(), v.Y(), "y", 10)
   ok += compare(p.Z(), v.Z(), "z", 10)
   
   # test from only rotation and only translation
   ta = Transform3D(EulerAngles(1., 2., 3.))
   tb = Transform3D(XYZVector(1, 2, 3))
   tc = Transform3D(Rotation3D(EulerAngles(1., 2., 3.)), XYZVector(1, 2, 3))
   #BP: td = Transform3D(ta.Rotation(), ta.Rotation() * XYZVector(1, 2, 3))
   #Not to use:  
   # ta.Rotation() * XYZVector(1, 2, 3))
   #Instead:
   # ta.Rotation().__call__ ( XYZVector(1, 2, 3)) # .__call__ operator
   td = Transform3D(ta.Rotation(), ta.Rotation().__call__( XYZVector(1,2,3) ) )
   
   #BP:DOING
   ok += compare(tc == tb * ta, Double_t(True), "== Rot*Tra")
   ok += compare(td == ta * tb, Double_t(True), "== Rot*Tra")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")
   
   c_print(f"Test Plane3D :                  ")
   ok = 0
   # test transfrom a 3D plane
   
   p1 = XYZPoint(1, 2, 3)
   p2 = XYZPoint(-2, -1, 4)
   p3 = XYZPoint(-1, 3, 2)
   plane = Plane3D(p1, p2, p3)
   
   """
Vector operator() (const Vector & v) const {
      return Vector( fM[kXX]*v.X() + fM[kXY]*v.Y() + fM[kXZ]*v.Z() ,
                     fM[kYX]*v.X() + fM[kYY]*v.Y() + fM[kYZ]*v.Z() ,
                     fM[kZX]*v.X() + fM[kZY]*v.Y() + fM[kZZ]*v.Z()  );
   }
   """ 
   n = plane.Normal()
   # normal is perpendicular to vectors on the planes obtained from subtracting
   # the points
   ok += compare(n.Dot(p2 - p1), 0.0, "n.v12", 10)
   ok += compare(n.Dot(p3 - p1), 0.0, "n.v13", 10)
   ok += compare(n.Dot(p3 - p2), 0.0, "n.v23", 10)
   
   #Note: 
   #      In Python:     
   #                     t -> Transform3D
   #                     t(plane) -> # Math.PositionVector
   #      In C++ version: 
   #                     t -> Transform3D
   #                     Plane3D plane1 = t(plane);
   #      
   #BP:DOING
   
   #Note: operator()(Plane3D) is not well defined in pyroot.
   #      Here we redefine it using the web-page documentation:
   #      https://root.cern/doc/master/GenVector_2Transform3D_8h_source.html#l00760
   """
   /**
      Transformation on a 3D plane
   */
   template <typename TYPE>
   Plane3D<TYPE> operator()(const Plane3D<TYPE> &plane) const
   {
      // transformations on a 3D plane
      const auto n = plane.Normal();
      // take a point on the plane. Use origin projection on the plane
      // ( -ad, -bd, -cd) if (a**2 + b**2 + c**2 ) = 1
      const auto d = plane.HesseDistance();
      Point p(-d * n.X(), -d * n.Y(), -d * n.Z());
      return Plane3D<TYPE>(operator()(n), operator()(p));
   }
   """
   def Transform3D_Py(transform, plane):
      n = plane.Normal()
      d = plane.HesseDistance()
      p = XYZPoint( -d * n.X(), -d * n.Y(), -d * n.Z() )
      n_t = transform(n) 
      p_t = transform(p)
      return Plane3D(n_t, p_t)
   
   #Plane3D plane1 = t(plane);
   #Not to use: plane1 = Plane3D(plane, t) 
   plane1 = Transform3D_Py(t, plane) 
   global gplane1
   gplane1 = plane1
   #plane1 = t(plane)
   #plane1 = t(plane)
   global gt 
   gt = t 
   global gplane
   gplane = plane
   
   # transform the points
   pt1 = t(p1)
   pt2 = t(p2)
   pt3 = t(p3)
   global gp1, gp2, gp3
   gp1 = p1
   gp2 = p2
   gp3 = p3
   global gpt1, gpt2, gpt3
   gpt1 = pt1
   gpt2 = pt2
   gpt3 = pt3
   plane2 = Plane3D(pt1, pt2, pt3)
   global gplane2
   gplane2 = plane2
   
   n1 = plane1.Normal()
   n2 = plane2.Normal()
   
   ok += compare(n1.X(), n2.X(), "a", 10)
   ok += compare(n1.Y(), n2.Y(), "b", 10)
   ok += compare(n1.Z(), n2.Z(), "c", 10)
   ok += compare(plane1.HesseDistance(), plane2.HesseDistance(), "d", 10)
   
   # check distances
   ok += compare(plane1.Distance(pt1), 0.0, "distance", 10)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")
   
   c_print(f"Test LorentzRotation :          ")
   ok = 0
   
   lv = XYZTVector(1., 2., 3., 4.)
   
   # test from rotx (using boosts and 3D rotations not impl. yet)
   # rx, ry and rz already defined
   #Not to use:
   #BP: r3d = rx * ry * rz
   #BP: r3d = rx ( ry ( rz ) )
   #BP: r3d = rx *( ry *( rz ) )
   #
   #Instead: we will use the pre-loaded rotation operator defined earlier.
   #         Thus we will be able to multiply rotation.
   #         The operator* is into _mul_Rotation function.
   #         TODO:
   #         Need to fix Rotation-multiplication between two diferent Rotations.
   #         Note:
   #               _mul_Rotation(rx, ry) is a Rotation3D-object.
   #Meanwhile: 
   r3d =  _mul_Rotation(rx, ry) * rz # Rotation3D
   
   rlx = LorentzRotation(rx)
   rly = LorentzRotation(ry)
   rlz = LorentzRotation(rz)
   global grlx, grly, grlz
   grlx = rlx
   grly = rly
   grlz = rlz
   
   #BP: rl0 = rlx * rly * rlz
   #BP: rl0 = rlx ( rly ( rlz ) ) 
   rl0 = rlx *( rly *( rlz ) ) 
   rl1 = LorentzRotation(r3d)
   
   #BP: lv0 = rl0 * lv
   lv0 = rl0 ( lv )
   
   #BP: lv1 = rl1 * lv
   lv1 = rl1 ( lv )
   
   #BP: lv2 = r3d * lv
   lv2 = r3d ( lv )
   
   ok += compare(lv1 == lv2, True, "V0==V2")
   ok += compare(lv1 == lv2, True, "V1==V2")
   
   rlData = [Double_t() for _ in range(16)]
   c_rlData = to_c(rlData)
   rl0.GetComponents(c_rlData)
   rlData = list(c_rlData)
   ml = TMatrixD(4, 4, c_rlData)
   lvData = [Double_t() for _ in range(4)]
   c_lvData = to_c(lvData)
   lv.GetCoordinates(c_lvData)
   lvData = list(c_lvData)
   ql = TVectorD(4, c_lvData)

   #BP: qlr = ml * ql # TVectorD
   #Not to use: ml * ql 
   # Note: ql is TVectorD
   #       ml is TMatrixD
   #instead:  
   qlr = _mul_(ml, ql) # TVectorD
   global gql, gml
   gql = ql
   gml = ml
   
   ok += compare(lv1.X(), qlr(0), "x")
   ok += compare(lv1.Y(), qlr(1), "y")
   ok += compare(lv1.Z(), qlr(2), "z")
   ok += compare(lv1.E(), qlr(3), "t")
   
   # test inverse
   global glv, grl0, glv0
   glv = lv
   grl0 = rl0
   glv0 = lv0 

   #BP: lv0 = rl0 * rl0.Inverse() * lv
   #Note: lv0 has to be a TLorentzVector
   #      But this operations gives another LorentzRotation: 
   #         lv0 = rl0 * rl0.Inverse() * lv
   #      Instead we will use operator()
   #      rl0(lv) -> another LorentzVector
  
   lv0 = (rl0 * rl0.Inverse()).__call__(lv)
   
   ok += compare(lv0.X(), lv.X(), "x")
   ok += compare(lv0.Y(), lv.Y(), "y")
   ok += compare(lv0.Z(), lv.Z(), "z")
   ok += compare(lv0.E(), lv.E(), "t")
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")

   # test Boosts
   c_print(f"Test Boost :                    ")
   ok = 0
   
   bst = Boost(0.3, 0.4, 0.5) #  boost (must be <= 1)
   
   lvb = bst(lv) # XYZTVector
   
   rl2 = LorentzRotation(bst)
   
   lvb2 = rl2(lv) # XYZTVector
   
   # test with lorentz rotation
   ok += compare(lvb.X(), lvb2.X(), "x")
   ok += compare(lvb.Y(), lvb2.Y(), "y")
   ok += compare(lvb.Z(), lvb2.Z(), "z")
   ok += compare(lvb.E(), lvb2.E(), "t")
   ok += compare(lvb.M(), lv.M(), "m", 50); # m must stay constant
  
   
   # test inverse
   # lv0 = (bst.Inverse()).__call__(lvb) # or 
   lv0 = bst.Inverse() * lvb
   
   ok += compare(lv0.X(), lv.X(), "x", 5)
   ok += compare(lv0.Y(), lv.Y(), "y", 5)
   ok += compare(lv0.Z(), lv.Z(), "z", 3)
   ok += compare(lv0.E(), lv.E(), "t", 3)
   
   brest = lv.BoostToCM()
   bst.SetComponents(brest.X(), brest.Y(), brest.Z())
   
   lvr = bst * lv
   
   ok += compare(lvr.X(), 0.0, "x", 10)
   ok += compare(lvr.Y(), 0.0, "y", 10)
   ok += compare(lvr.Z(), 0.0, "z", 10)
   ok += compare(lvr.M(), lv.M(), "m", 10)
   
   if (ok == 0):
      c_print(f"\t OK ", "\n")
   else:
      c_print(f"")

   return ok
   

# void
def mathcoreGenVector() :
   
   testVector3D()
   testPoint3D()
   testLorentzVector()
   testVectorUtil()
   testRotation()
   
   c_print(f"\n\nNumber of tests " , ntest , " failed = " , nfail)
  
   


if __name__ == "__main__":
   mathcoreGenVector()
