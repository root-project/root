void myfunc(std::vector<double> &vec) {
   for(unsigned int i=0; i< vec.size(); ++i) { 
      double dval = vec[i];
      fprintf(stdout,"myfunc: vec[%d]==%g\n",i,dval);
   }
}

void myfunc2(std::vector<vector<double> > &vec) {
   for(unsigned int j=0; j<vec.size(); ++j) {
      for(unsigned int i=0; i< vec[j].size(); ++i) { 
         double dval = vec[j][i];
         fprintf(stdout,"myfunc2: vec[%d][%d]==%g\n",j,i,dval);
      }
   }
}

double dude() {
   myfunc(*myvec);
   for(unsigned i=0; i< myvec->size(); ++i) { 
      double dval = myvec[i];
      fprintf(stdout,"myvec[%d]==%g\n",i,dval);
   }
   myfunc2(*myvecvec);
   int x = 0; x = myvecvec[x].size();
   for(unsigned int j=0; j< myvecvec->size(); ++j) {
      fprintf(stdout,"myvecvec size: %lu\n",(long)myvecvec->size());
      for(unsigned int i=0; i< myvecvec[j].size(); ++i) {
         double dval = myvecvec[j][i]; 
         fprintf(stdout,"ptrmyvec[%d][%d]==%g\n",j,i,dval);
      }
   }
   return 3;
}
