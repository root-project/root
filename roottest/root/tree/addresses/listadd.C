void printadd(TBranch* br) {
  fprintf(stdout, "%s : %p\n",br->GetName(), br->GetAddress());
}

void listadd(TBranch* top) {
  if (top) {
    TIter next(top->GetListOfBranches());
    TBranch *br = 0;
    while ( br=(TBranch*)next()) {
      printadd(br);
      listadd(br);
    }
  }
}

void listadd(TTree* tree) {
  if (tree) {
    TIter next(tree->GetListOfBranches());
    TBranch *br = 0;
    while ( br=(TBranch*)next()) {
      printadd(br);
      listadd(br);
    }
  }
}
