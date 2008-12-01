TEveWindowSlot *s = 0;

void test_windows()
{
   TEveManager::Create();

   TEveWindowSlot *slot = 0;

   // ----------------------------------------------------------------

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());

   TEveWindowPack* pack1 = slot->MakePack();
   s = pack1->NewSlot();
   slot = pack1->NewSlot();
   
   TEveWindowPack* pack2 = slot->MakePack();
   pack2->FlipOrientation();

   pack2->NewSlot();
   pack2->NewSlot();

   // ----------------------------------------------------------------

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());

   TEveWindowTab* tab1 = slot->MakeTab();
   tab1->NewSlot();
   slot = tab1->NewSlot();

   TEveWindowTab* tab2 = slot->MakeTab();
   tab2->NewSlot();
   tab2->NewSlot();

   // ----------------------------------------------------------------

   gEve->GetBrowser()->GetTabRight()->SetTab(1);
}
