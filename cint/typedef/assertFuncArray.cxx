class AliTRDrawStream {

};

class AliTagFrame {

   void (AliTagFrame::*fTagCutMethods [3]) (void); //tag fields
   void (AliTRDrawStream::*fStoreError)();         //! function pointer to method used for storing the error
   ClassDef(AliTagFrame,2);

};

int assertFuncArray()
{
   return 0;
}
