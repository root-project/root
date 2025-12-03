void testing()
{
   test t;
   auto cl = TClass::GetClass(typeid(t));
   cl->GetStreamerInfo()->ls("noaddr");
   auto el = cl->GetStreamerInfo()->GetElement(0);
   auto collcl = el->GetClass();
   if (!collcl)
      std::cout << "Error: missing TClass for nested map\n";
   else if (!collcl->GetCollectionProxy())
      std::cout << "Error: missing collection proxy for nested map\n";
   else if (collcl->GetCollectionProxy()->GetProperties() & TVirtualCollectionProxy::kIsEmulated)
      std::cout << "Error: collection proxy for nested map is emulated\n";
}
