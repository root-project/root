

double dude() {
   for(unsigned int i=0; i< myvec->size(); ++i) {
      double dval = myvec[i];
      fprintf(stdout,"myvec[%d]==%g\n",i,dval);
   }
   for(int j=0; j< myvecvec.GetEntries(); ++j) {
      fprintf(stdout,"myvecvec size: %d\n",myvecvec.GetEntries());
      for(unsigned int i=0; i< myvecvec[j].size(); ++i) {
         double dval = myvecvec[j][i];
         fprintf(stdout,"ptrmyvec[%d][%d]==%g\n",j,i,dval);
      }
   }
   return 3;
}
