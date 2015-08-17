// inspired by ROOT-7007
int execNonExistingFriend() {
   TTree* a = new TTree("a", "a");
   auto fe = a->AddFriend("asfasf", "asfasgfags.root");
   if (!fe->IsZombie()){
      std::cerr << "The TFriendElement instance should be a zombie but it's not!\n";
   } else {
      std::cout << "The TFriendElement instance is a zombie.\n";
   }

   return 0;
}
