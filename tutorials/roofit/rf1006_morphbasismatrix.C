/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Morphing basis matrix
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 04/2016 - Carsten Burgard

#include "RooLagrangianMorphing.h"
#include "RooStringVar.h"
using namespace RooFit;

void rf1006_morphbasismatrix()
{
  // ---------------------------------------------------------
  // E f f e c t i v e   L a g r a n g i a n   M o r p h i n g
  // =========================================================

  // Define identifier for infilename
  std::string infilename = "input/vbfhwwlvlv_3d.root";

  // Define identifier for input sample foldernames,
  // require 15 samples to describe three paramter morphing function
  std::vector<std::string> samplelist = {"kAwwkHwwkSM0","kAwwkHwwkSM1","kAwwkHwwkSM10","","kAwwkHwwkSM11","kAwwkHwwkSM12",
                                         "kAwwkHwwkSM13","kAwwkHwwkSM2","kAwwkHwwkSM3","kAwwkHwwkSM4","kAwwkHwwkSM5",
                                          "kAwwkHwwkSM6","kAwwkHwwkSM7","kAwwkHwwkSM8","kAwwkHwwkSM9","kSM0"};

  // Construct list of input samples
  RooArgList inputs;
  for(auto const& sample: samplelist)
  {
     RooStringVar* v = new RooStringVar(sample.c_str(), sample.c_str(), sample.c_str());
     inputs.add(*v);
  }

  // C r e a t e   m o r p h i n g   f u n c t i o n
  // ------------------------------------------------

  // Construct three parameter morphing functions for opening angle
  //  of the final-state jets in the process VBF Higgs decaying to
  //   W+ W- in the Higgs Characterisation Model 
  RooHCvbfWWMorphFunc morphfunc("morphfunc_dphijj", "morphfunc_dphijj", infilename.c_str(), "twoSelJets/dphijj", inputs);

  // Write out the coefficient matrix
  if(!morphfunc.writeCoefficients("matrix.txt"))
    std::cout << "failed to save matrix" << std::endl;
}
