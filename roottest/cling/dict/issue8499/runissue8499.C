int runissue8499() {
   gSystem->Load("issue8499_dict");
   return TClass::GetClass("o2::vertexing::DCAFitter2")->GetClassSize() > 0 ? 0 : 1;
}
