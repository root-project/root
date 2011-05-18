#include <vector>;

void StandaloneClassExample()
{
   // A simple example of how the "standalone classes" can be used
      
   // Load the stand alone "trained LD class" 
   // if the example were not a ROOT macro but a stand alone program, you
   // would simplye "include" this file"

   gROOT->LoadMacro("weights/TMVAClassification_LD.class.C++");

   std::vector<string> inputVariableNames;
   // you need to use the same names as during traiing. Meant as a "consistency" check, that
   // makes you aware of what you are doing. You can find the names in the "xxx.class.C" file
   // just look for "training input variables". Obviously, you want to "apply" it using the
   // same variables.

   inputVariableNames.push_back("var1+var2");
   inputVariableNames.push_back("var1-var2");
   inputVariableNames.push_back("var3");
   inputVariableNames.push_back("var4");
   // instanticat the LD class and tell it about the variable names
   // to allow it to check internally that it has actually been trained with
   // THESE variables
   IClassifierReader* classReader = new ReadLD(inputVariableNames);
   
   // put your input variables into a std::vector 
   // (this would typically be inside an "event loop" of course..
   std::vector<double> inputVariableValues;
   inputVariableValues.push_back(1.);
   inputVariableValues.push_back(1.6);
   inputVariableValues.push_back(3.4);
   inputVariableValues.push_back(2.4);

   cout << "For input values: " ;
   for (int i=0; i<4; i++) cout << inputVariableValues[i] << "  ";
   cout << endl;

   // get the MVA output value for this particular event variables.
   cout << "The LD MVA value is: " << classReader->GetMvaValue(inputVariableValues) << endl;


}
