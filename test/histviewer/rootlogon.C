{
   printf("\n");
   if (gSystem->Load("./libHistViewer.so")==0) {
      printf("Shared library libHistViewer.so loaded\n");
      gui = new HistAction(gClient->GetRoot(),1,1);
   } else {
      printf("Unable to load libHistViewer.so\n");
   } 
}
