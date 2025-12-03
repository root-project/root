{
   gSystem->Load("libcustomStreamer_rflx_dictrflx");
   FooCustomStreamer foo;
   TBufferFile read_buffer(TBuffer::EMode::kRead);
   foo.Streamer(read_buffer);
   TBufferFile write_buffer(TBuffer::EMode::kWrite);
   foo.Streamer(write_buffer);
}
