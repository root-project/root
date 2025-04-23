
// Ambiguous with something in std
#pragma link C++ namespace next;
#pragma link C++ class next::Inside_next+;


// Ambiguous with something we put in std
#pragma link C++ namespace Next;
#pragma link C++ class Next::Inside_Next+;

// Ambiguous with something we put in the Functions namespace
#pragma link C++ namespace OtherNext;
#pragma link C++ class OtherNext::Inside_OtherNext+;

// No ambiguity (reference)
#pragma link C++ namespace YetAnotherNext;
#pragma link C++ class YetAnotherNext::Inside_YetAnotherNext+;
