#include "TMVA/RModelParser_ONNX.hxx"
#include <string>

#include "gtest/gtest.h"

TEST(ONNXParser, ConvAddFusedReluParsesSuccessfully)
{
   std::string const model{"ConvAddReluFuseBug.onnx"};
   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   EXPECT_NO_THROW((void)parser.Parse(model, false));
}
