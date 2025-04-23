class TestObj : public TObject {
public:
  enum {
    kAnswerGood,
    kAnswerBad
  } testAnswers;
  TestObj();
  virtual ~TestObj() {};
  void func(Double_t aValue);
  ClassDef(TestObj,0);
};

void execROOT6625() {
  TestObj::Class()->GetMethod("func", "TestObj::kAnswerGood^TestObj::kAnswerBad");
}
