// Author: Federico Sossai
// Last modified: 2021/07/30
// Description:
//    SOFIE command line compiler.
//    This program is automatically run when the target 'TestCustomModels' is built.
//    Usage example: $./sofiec indir/mymodel.onnx outdir/myname.hxx

#include <iostream>

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

int main(int argc, char *argv[])
{
   if (argc < 2) {
      std::cerr << "ERROR: missing input file\n";
      return -1;
   }

   std::string outname;
   if (argc == 3) {
      outname = argv[2];
   }

   RModelParser_ONNX parser;
   RModel model = parser.Parse(argv[1]);
   model.Generate();
   model.OutputGenerated(outname);

   return 0;
}