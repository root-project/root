{
   gSystem->Load("inst2lib");
   gSystem->Load("instlib");
   gInterpreter->AutoParse("Outer"); // To get the prototype of writeFile
   return writeFile();
}
