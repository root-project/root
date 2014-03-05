{
 TFile ifile("example.root"); 
 
 Container* container =  dynamic_cast<Container*>(ifile.Get("Container"));
 
 if (!container) {
    std::cerr << "Error reading the Container from the file!\n";
 }
 else{
   std::cout << "Supercluster energy: " << container->GetSc().GetEnergy() << std::endl;
   const Vertex& vertex = container->GetParticle().GetVertex();
   std::cout << "Particle vertex position: " << vertex.X() << " "<< vertex.Y() << " " << vertex.Z() << "\n";  
 }
 
 return 0;
}
