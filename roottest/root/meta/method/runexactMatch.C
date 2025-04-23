int runexactMatch() {
   TClass *cl = TObject::Class();
   if (!cl) {
      fprintf(stderr,"Could not find the TClass for TObject\n");
      return 1;
   }

   TMethod * m = cl->GetMethodWithPrototype("Print","char*",true);
   if (!m) {
      fprintf(stderr, "Could not find the Print method with a ConversionMatch.\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Print","char*",true,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Found the Print method with an ExactMatch but using 'char*'.\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Print","const char*",true,ROOT::kExactMatch);
   if (!m) {
      fprintf(stderr, "Could not find the Print method with a ExactMatch to const char*.\n");
      return 1;
   }
   
   m = cl->GetMethodWithPrototype("Print","const char*",false,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Found the Print method with an ExactMatch but non-const request.\n");
      return 1;
   }
  
   cl = TNamed::Class();
   if (!cl) {
      fprintf(stderr,"Could not find the TClass for TNamed\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Print","char*",true);
   if (!m) {
      fprintf(stderr, "Could not find the Print method in TNamed with a ConversionMatch.\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Print","char*",true,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Found the Print method in TNamed with an ExactMatch but using 'char*'.\n");
      return 1;
   }

   m = cl->GetMethodWithPrototype("Print","const char*",true,ROOT::kExactMatch);
   if (!m) {
      fprintf(stderr, "Could not find the Print method in TNamed with a ExactMatch to const char*.\n");
      return 1;
   }
   
   m = cl->GetMethodWithPrototype("Print","const char*",false,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Found the Print method in TNamed with an ExactMatch but non-const request.\n");
      return 1;
   }
 

#if 0
   m = cl->GetClassMethodWithPrototype("Print","const char*",true,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Found the Print method in TNamed with a ExactMatch to const char* but request fo r local search.\n");
      return 1;
   }

   m = cl->GetClassMethodWithPrototype("Print","const char*",true,ROOT::kExactMatch);
   if (m) {
      fprintf(stderr, "Could not find the Print method in TNamed with a ExactMatch to const char* but request fo r local search.\n");
      return 1;
   }
#endif
   return 0;
}
