#ifndef ROOT_TRESPONSETABLEITERATOR
#define ROOT_TRESPONSETABLEITERATOR

class TResponseTable;
class TVolumeView;

class TResponseIterator {
  protected:
     TResponseTable *fResponse;
     TVolumeView  *fVolumeView;
     
  public:
    TResponseIterator(TResponseTable *response, TVolumeView *detector);

  ClassDef(TResponseIterator,0)   
};

#endif

