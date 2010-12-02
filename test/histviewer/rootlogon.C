{
   printf("\n");
   if (gSystem->Load("./libHistViewer")==0) {
      printf("Shared library libHistViewer loaded\n");
      gui = new HistAction(gClient->GetRoot(),1,1);
   } else {
      printf("Unable to load libHistViewer\n");
   } 
}
