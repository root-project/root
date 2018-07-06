void tester(int MAXN, int nthread=1, int seed=123) {
    // this is a test to tell the difference between old build of KDTree and the parallel one.
    // Since clock() doesn't work for multi-threading, time() is used, which means the time can only be s instead of ms
    // So you need a bigger MAXN(the number of elements that is used to build KDTree) such like 4000000 or higher.
    srand(seed);
    struct threeDcoord_t {
        Double_t x;
        Double_t y;
        Double_t z;
    };
    threeDcoord_t point;

    TTree *tree = new TTree("3DPoints", "3DP");
    tree->Branch("p", &point.x, "x/D:y/D:z/D");

    for (int i = 0; i < MAXN; i++) {
        point.x = rand() / Double_t(RAND_MAX);
        point.y = rand() / Double_t(RAND_MAX);
        point.z = rand() / Double_t(RAND_MAX);
        //printf("%lf",point.z);
        tree->Fill();
    }

    time_t timer_a, timer_b;
    //tree->Scan("z");
    //tree->Print();

    tree->Draw("x:y:z", "", "goff");

    //now make a kd-tree on the drawn variables
    TKDTreeID *kdtree = new TKDTreeID(MAXN, 3, 1);
    kdtree->SetData(0, tree->GetV1());
    kdtree->SetData(1, tree->GetV2());
    kdtree->SetData(2, tree->GetV3());

    timer_a = time(NULL);
    kdtree->Build();
    timer_b = time(NULL);
    cout << "Origin Time : " <<(timer_b - timer_a)<< "s" << endl;

    TKDTreeID *pkdtree = new TKDTreeID(MAXN, 3, 1);
    pkdtree->SetData(0, tree->GetV1());
    pkdtree->SetData(1, tree->GetV2());
    pkdtree->SetData(2, tree->GetV3());
	
    timer_a = time(NULL);
    pkdtree->Build(nthread);
    timer_b = time(NULL);
    cout << "Parallel Time : " <<(timer_b - timer_a)<< "s" << endl;

    double a,b;
    for(int i = 0; i<MAXN;i++)
    {
        a = kdtree->GetNodeValue(i);
        b = pkdtree->GetNodeValue(i);
        if(a!=b)
            cout << "differ occured at : " << i << "with" << a <<" and "<< b << endl;
    }
}

