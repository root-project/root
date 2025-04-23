class AliTRDrawStream {

};

struct AliTagFrame {

   void (AliTagFrame::*fTagCutMethods [3]) (void); //tag fields
   void (AliTRDrawStream::*fStoreError)();         //! function pointer to method used for storing the error
   virtual ~AliTagFrame() {}
   ClassDef(AliTagFrame,2);

};

int assertFuncArray()
{
   return 0;
}
