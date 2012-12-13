{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   gROOT->ProcessLine(".L header.C+");
   TClass *cl = gROOT->GetClass("Hard2Stream");
// cl->SetStreamer(new TStreamer(hard2StreamStreamer));
//setStreamer();

   cout << "Hard2Stream version #" << cl->GetClassVersion() << endl;
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   " if (1) {"
#endif
   "TClass *cl = gROOT->GetClass(\"Hard2Stream\");"
   "TBufferFile buf(TBuffer::kWrite);"
   "Hard2Stream myobj(33.33);"
   "myobj.print();"
   "buf.WriteObjectAny(&myobj,cl);"
   "buf.SetReadMode();"
   "buf.Reset(); "
   "Hard2Stream* readobj = (Hard2Stream*) buf.ReadObjectAny(0);"
   "readobj->print();"
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   " } "
#endif
);
#else
   TBufferFile buf(TBuffer::kWrite);
   Hard2Stream myobj(33.33);
   myobj.print();
   buf.WriteObjectAny(&myobj,cl);
   buf.SetReadMode();
   buf.Reset(); 
   Hard2Stream* readobj = (Hard2Stream*) buf.ReadObjectAny(0);
   readobj->print();
#endif

#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
}
#endif
}

   
