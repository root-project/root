{
   TTree *t = new TTree("T","T");
   int arr[2];
   t->Branch("arr",&arr[0],"arr[2]/I");
   arr[0] = 1;
   arr[1] = 2;
   t->Fill();
   t->Scan("arr");
   TSQLResult *r;
   r = t->Query("arr");
   TSQLRow *row;
   while( ( row = r->Next() ) ) { cout << row->GetField(0) << endl; delete row ; row = 0; }
}
