// Fill an n-tuple and write it to a file simulating measurement of
// conductivity of a material in different conditions of pressure
// and temperature.

void write_ntuple_to_file(){

    TFile ofile("conductivity_experiment.root","RECREATE");

    // Initialise the TNtuple
    TNtuple cond_data("cond_data",
                      "Example N-Tuple",
                      "Potential:Current:Temperature:Pressure");

    // Fill it randomly to fake the acquired data
    TRandom3 rndm;
    float pot,cur,temp,pres;
    for (int i=0;i<10000;++i){
        pot=rndm.Uniform(0.,10.);      // get voltage
        temp=rndm.Uniform(250.,350.);  // get temperature
        pres=rndm.Uniform(0.5,1.5);    // get pressure
        cur=pot/(10.+0.05*(temp-300.)-0.2*(pres-1.)); // current
// add some random smearing (measurement errors)
        pot*=rndm.Gaus(1.,0.01); // 1% error on voltage
        temp+=rndm.Gaus(0.,0.3); // 0.3 abs. error on temp.
        pres*=rndm.Gaus(1.,0.02);// 1% error on pressure
        cur*=rndm.Gaus(1.,0.01); // 1% error on current
// write to ntuple
        cond_data.Fill(pot,cur,temp,pres);
        }

    // Save the ntuple and close the file
    cond_data.Write();
    ofile.Close();
}
