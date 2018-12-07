#include <ROOT/REveTableInfo.hxx>

#include "json.hpp"

using namespace ROOT::Experimental;

void REveTableViewInfo::SetDisplayedCollection( ElementId_t collectionId)
{
    fDisplayedCollection = collectionId;
    for (std::vector<Delegate_t>::iterator it = fDelegates.begin(); it != fDelegates.end(); ++it) {
       (*it)(collectionId);
    }
}

Int_t REveTableViewInfo::WriteCoreJson(nlohmann::json& j, Int_t rnr_offset)
{
   int off = REveElement::WriteCoreJson(j, rnr_offset);
   j["fDisplayedCollection"] = fDisplayedCollection;
   //j["_typename"]  = IsA()->GetName();
   return off;
}
