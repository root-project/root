void execmangledTest(){

  auto className="mangledTest";
  if (!TClass::GetClass(className)) cout << "ERROR: Cannot find " << className << " class\n";
  else cout << "SUCCESS: " << className << " class found\n";

}
