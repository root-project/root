

void readFromFile(){
 TXMLFile ifile ("example.xml");  
 Container* container =  dynamic_cast<Container*>(ifile.Get("Container"));
 
 // If successful, dump some info
 if (!container) {
    std::cerr << "Error reading the Container from the file!\n";
 }
 else{
   std::cout << "Supercluster energy: " << container->GetSc().GetEnergy() << std::endl;
   const Vertex& vertex = container->GetParticle().GetVertex();
   std::cout << "Particle vertex position: " << vertex.X() << " "<< vertex.Y() << " " << vertex.Z() << "\n";  
 }   
}

void inspectDicts(){
 // Supercluster energy
 TClass* sc = TClass::GetClass("SuperCluster");   
 TDataMember* dm=sc->GetDataMember("energy2");   
 if (dm) std::cout << "SuperCluster::energy2 type is " << dm->GetTypeName() << std::endl;
 dm=sc->GetDataMember("energy"); // should not 
 if (dm) std::cout << "SuperCluster::energy type is " << dm->GetTypeName() << std::endl;

 // Particle vertex 
 TClass* par = TClass::GetClass("Particle"); 
 dm = par->GetDataMember("vertex");
 if (dm) std::cout << "Particle::vertex type is " << dm->GetTypeName() << std::endl;
 dm = par->GetDataMember("vertex2"); // should not print
 if (dm) std::cout << "Particle::vertex2 type is " << dm->GetTypeName() << std::endl; 
}

void dumpOnFile(){
       
   TXMLFile ofile("newFile.xml","RECREATE");
   
   Container cont;       
   cont.SetScEnergy(123.);
   cont.Write();

   ofile.Close();
}

void execionameiotype(){

 // Some gymnastic with dictionaries
 inspectDicts();
 
 // Now read from file
 readFromFile();
 
 // Now Dump on a file
 dumpOnFile();
 
}