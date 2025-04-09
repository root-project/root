{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L WrapSimple.h+");
   if (gROOT->GetClass("AddSpace::Simple")==0) {
      cerr << "Could not retrive the class named AddSpace::Simple\n";
   }
   if (gROOT->GetClass("AddSpace")) {
      // The test is about being able to find the class even-though
      // the user did not have an explicit pragma include of for
      // the namespace.  With cling, we no longer filter which 
      // symbol are known to the interpreter dictionary (AST nodes),
      // it is non longer possible to have the situation where
      // it is forgotten.  In addition, we can always create a
      // TClass for namespace from its NamespaceDecl AST node
      // so this test is moot.
      // cerr << "We found the namespace AddSpace.  The test is not complete!\n";
   }
}
