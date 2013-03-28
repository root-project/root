// Fill an n-tuple and write it to a file simulating measurement of 
// conductivity of a material in different conditions of pressure
// and temperature.

void write_ntuple_to_file(){

    // Initialise the TNtuple
    TNtuple cond_data("cond_data",
                      "Example N-Tuple",
                      "Potential:Current:Temperature:Pressure");

    // Fill it randomly to fake the acquired data
    float pot,cur,temp,pres;
    for (int i=0;i<10000;++i){
        pot=gRandom->Uniform(0.,10.);      // get voltage
        temp=gRandom->Uniform(250.,350.);  // get temperature
        pres=gRandom->Uniform(0.5,1.5);    // get pressure
        cur=pot/(10.+0.05*(temp-300.)-0.2*(pres-1.)); // current
// add some random smearing (measurement errors)  
        pot*=gRandom->Gaus(1.,0.01); // 1% error on voltage
        temp+=gRandom->Gaus(0.,0.3); // 0.3 abs. error on temp.
        pres*=gRandom->Gaus(1.,0.02);// 1% error on pressure
        cur*=gRandom->Gaus(1.,0.01); // 1% error on current
// write to ntuple
        cond_data.Fill(pot,cur,temp,pres);
        }

    // Open a file, save the ntuple and close the file
    TFile ofile("conductivity_experiment.root","RECREATE");
    cond_data.Write();
    ofile.Close();
}
