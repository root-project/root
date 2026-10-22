{
int old = gInterpreter->SetClassAutoloading(kFALSE);
gInterpreter->LoadLibraryMap("libbtag.rootmap");
gInterpreter->LoadLibraryMap("libjet.rootmap");
gInterpreter->LoadLibraryMap("libsjet.rootmap");
gInterpreter->SetClassAutoloading(old);
}

