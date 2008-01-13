double dude() {
   for(int i=0; i< myvec->size(); ++i) { 
      double dval = myvec->at(i);
      fprintf(stdout,"myvec[%d]==%g\n",i,dval);
   }
   return 3;
}
