void run(int i=9) {
   TFile *file;
   switch (i) {
      case 9: file = new TFile("Event.new.split9.root"); break;
      case 2: file = new TFile("Event.new.split2.root"); break;
      case 1: file = new TFile("Event.new.split1.root"); break;
      case 0: file = new TFile("Event.new.split0.root"); break;
      default: return;
   }

   TTree * tree = (TTree*)file->Get("T");

   if (i==0) draw(tree,"script0.C");
   else draw(tree,"script.C");
}
