 {
   // Copyright(c) 2001 [BNL] Brookhaven National Laboratory, Valeri Fine  (fine@bnl.gov). All right reserved",
   // TGenericTable and TIndexTable test macro
    int problemCounter = 0;
    gSystem->Load("libTable");
    struct hit {
     float  energy;     /* energy */
     int    detectorId; /* geometry id */
    };
 
    TGenericTable *allHits = new TGenericTable("hit","hits",1000);
    allHits->Print();
    hit  a;
    memset(&a,0,sizeof(a));
    int i = 0;
    for (i=0; i<5; i++) {
	   a.energy = sin(i*0.1);
	   a.detectorId = i;
	   allHits->AddAt(&a);
    }
    allHits->Print();
    // Draw the histogram for the selected column
    allHits->Draw("energy");
 }
 
