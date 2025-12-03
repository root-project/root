{
   gROOT->ProcessLine(".L header.C+");
   TClass *cl = gROOT->GetClass("Hard2Stream");
   // cl->SetStreamer(new TStreamer(hard2StreamStreamer));
   // setStreamer();

   cout << "Hard2Stream version #" << cl->GetClassVersion() << endl;
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
   "TBufferFile buf(TBuffer::kWrite);"
   "Hard2Stream myobj(33.33);"
   "myobj.print();"
   "buf.WriteObjectAny(&myobj,cl);"
   "buf.SetReadMode();"
   "buf.Reset(); "
   "Hard2Stream* readobj = (Hard2Stream*) buf.ReadObjectAny(0);"
   "readobj->print();"
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

}


