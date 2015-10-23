int runCheckSplitTestClass(){
   return TClass::GetClass("CheckSplitTestClass")->CanSplit() ? 0 : 1;
}