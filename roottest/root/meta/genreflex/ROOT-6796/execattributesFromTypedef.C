void execattributesFromTypedef(){

   int res = gInterpreter->AutoLoad("RootType");
   if (res!=0){
      cerr << "ERROR: Class RootType ended up in the rootmap!\n";
   } else {
      cout << "Class RootType did not end in the rootmap!\n";
   }

}
