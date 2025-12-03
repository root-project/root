{
int old = gInterpreter->SetClassAutoloading(kFALSE);
gInterpreter->LoadLibraryMap("lib/libbtag.rootmap");
gInterpreter->LoadLibraryMap("lib/libjet.rootmap");
gInterpreter->LoadLibraryMap("lib/libsjet.rootmap");
gInterpreter->SetClassAutoloading(old);
}

