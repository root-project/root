{
   // --------- S t y l e ---------------------------
   const Bool_t UsePaperStyle = 0;
   // -----------------------------------------------
   
   gSystem->Load("libMLP");
   
   // load TMVA shared library created in local release
   TString libTMVA( "../lib/libTMVA.1" );
   gSystem->Load( libTMVA );
   
   // welcome the user
   TMVA::Tools::TMVAWelcomeMessage();
   cout << "TMVAlogon: loaded TMVA library: \"" << libTMVA << "\"" << endl;
   
#include "tmvaglob.C"
   
   TMVAGlob::SetTMVAStyle(); 
   cout << "TMVAlogon: use style " << gStyle->GetName() << " [" << gStyle->GetTitle() << "]" << endl;
   cout << endl;
}
