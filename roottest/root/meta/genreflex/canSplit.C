int canSplit()
{
   auto c = TClass::GetClass("canSplit");
   return c->CanSplit() ? 0 : 1;
}
