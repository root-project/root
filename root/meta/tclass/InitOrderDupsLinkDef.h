
// In some cases (eg with module enabled) this lead to a duplicate dictionary
// being generated for the LorentzVector
#pragma link C++ typedef  Vector4D;

#pragma link C++ class Vector3D+; // This already has a dictionary without a pcm from libMathCore.
