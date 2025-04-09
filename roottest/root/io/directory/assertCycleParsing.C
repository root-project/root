int TestParsing(const char *input, short expected)
{
   short cycle;
   char  name[1000];
   TDirectory::DecodeNameCycle(input,name,cycle,1000);
   if (cycle != expected) {
      Error("TestParsing","With input=\"%s\", the cycle was found to be: %d rather than %d\n",input,cycle,expected);
      return 1;
   } 
   return 0;
}

int assertCycleParsing() {
  int result = TestParsing("obj;32000",32000);
  result += TestParsing("obj;33000",0);
  return result;
}
 
