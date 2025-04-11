{

    bool result = true;

    TChain *chain = new TChain("ntuple");
    chain->Add("hsimple1.root");
    chain->Add("hsimple2.root");
    
    TTree *tupleClone1 = (TTree*)chain->CopyTree("");
    TTree *tupleClone2 = (TTree*)chain->CopyTree("");
    
    
    TLeaf *origleaf  = chain->GetLeaf("pz");
    TLeaf *clone1leaf = tupleClone1->GetLeaf("pz");
    
    if (origleaf->GetValuePointer() != clone1leaf->GetValuePointer()) {
       cerr << "We have a problem since the address of the value is different in the original and in the first copy!" << endl;
       result = false;
    }
    
    TLeaf *clone2leaf = tupleClone2->GetLeaf("pz");
    
    if (origleaf->GetValuePointer() != clone2leaf->GetValuePointer()) {
       cerr << "We have a problem since the address of the value is different in the original and in the second copy!" << endl;
       result = false;
    }

    // We know all 3 are TNuple and pz is the 3rd 'arguments'
    
    TNtuple *ntuple = dynamic_cast<TNtuple*>(chain->GetTree());
    if ( (&(ntuple->GetArgs()[2])) != origleaf->GetValuePointer()) {
       cerr << "Error: the original should own the memory\n";
       result = false;
    }

  
    ntuple = dynamic_cast<TNtuple*>(tupleClone1);
    if ( (&(ntuple->GetArgs()[2])) == clone1leaf->GetValuePointer() ) {
       cerr << "Error: the 1st clone should NOT own the memory\n";
       result = false;
    }

    ntuple = dynamic_cast<TNtuple*>(tupleClone2);
    if ( (&(ntuple->GetArgs()[2])) == clone2leaf->GetValuePointer() ) {
       cerr << "Error: the 2nd clone should NOT own the memory\n";
       result = false;
    }
    
    TTree *tupleClone3 = (TTree*)chain->GetTree()->CloneTree(0);
    double a;
    tupleClone3->Branch("MT",&a,"MT/D");

    chain->LoadTree(0);
    origleaf  = chain->GetLeaf("pz");
    TLeaf *clone3leaf = tupleClone3->GetLeaf("pz");
    
    if (origleaf->GetValuePointer() != clone3leaf->GetValuePointer()) {
       cerr << "We have a problem since the address of the value is different in the original and in the third copy!" << endl;
       result = false;
    }
    if (tupleClone3->GetBranch("MT")->GetAddress() != (char*)&a) {
       cerr << "We have a problem since the address of the branch MT (" << (void*)(tupleClone3->GetBranch("MT")->GetAddress())
            << " is not the address of the variable (" << (void*)&a  << ")" << endl;
    }
    
    float py;
    chain->SetBranchAddress("py",&py);
    origleaf  = chain->GetLeaf("py");
    clone1leaf = tupleClone1->GetLeaf("py");
    clone2leaf = tupleClone2->GetLeaf("py");
    clone3leaf = tupleClone3->GetLeaf("py");

    if (   origleaf->GetValuePointer() != clone1leaf->GetValuePointer()
        || origleaf->GetValuePointer() != clone2leaf->GetValuePointer()
           || origleaf->GetValuePointer() != clone3leaf->GetValuePointer() ) {

       cerr << "We have a problem since the address of the value is different in the original and in one of the three copy!" << endl;
       result = false;
    }  


    delete chain;
    
    clone1leaf = tupleClone1->GetLeaf("pz");
    
    clone2leaf = tupleClone2->GetLeaf("pz");
    
    ntuple = dynamic_cast<TNtuple*>(tupleClone1);
    if ( (&(ntuple->GetArgs()[2])) != clone1leaf->GetValuePointer() ) {
       cerr << "Error: the 1st clone should own the memory\n";
       result = false;
    }
    
    ntuple = dynamic_cast<TNtuple*>(tupleClone2);
    if ( (&(ntuple->GetArgs()[2])) != clone2leaf->GetValuePointer() ) {
       cerr << "Error: the 2nd clone should own the memory\n";
       result = false;
    }
    
    if (!result) gApplication->Terminate(1);
}
