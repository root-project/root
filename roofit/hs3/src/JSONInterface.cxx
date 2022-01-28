#include <RooFitHS3/JSONInterface.h>

namespace {
template <class Nd>
class childItImpl : public RooFit::Experimental::JSONNode::child_iterator_t<Nd>::Impl {
public:
   using child_iterator = RooFit::Experimental::JSONNode::child_iterator_t<Nd>;
   childItImpl(Nd &n, size_t p) : node(n), pos(p) {}
   childItImpl(const childItImpl &other) : node(other.node), pos(other.pos) {}
   virtual std::unique_ptr<typename child_iterator::Impl> clone() const override
   {
      return std::make_unique<childItImpl>(node, pos);
   }
   virtual void forward() override { ++pos; }
   virtual void backward() override { --pos; }
   virtual Nd &current() override { return node.child(pos); }
   virtual bool equal(const typename child_iterator::Impl &other) const override
   {
      auto it = dynamic_cast<const childItImpl<Nd> *>(&other);
      return it && &(it->node) == &(this->node) && (it->pos) == this->pos;
   }

private:
   Nd &node;
   size_t pos;
};
} // namespace

namespace RooFit {
namespace Experimental {

template class JSONNode::child_iterator_t<JSONNode>;
template class JSONNode::child_iterator_t<const JSONNode>;

JSONNode::children_view JSONNode::children()
{
   return {child_iterator(std::make_unique<::childItImpl<JSONNode>>(*this, 0)),
           child_iterator(std::make_unique<::childItImpl<JSONNode>>(*this, this->num_children()))};
}
JSONNode::const_children_view JSONNode::children() const
{
   return {const_child_iterator(std::make_unique<::childItImpl<const JSONNode>>(*this, 0)),
           const_child_iterator(std::make_unique<::childItImpl<const JSONNode>>(*this, this->num_children()))};
}

std::ostream &operator<<(std::ostream &os, JSONNode const &s)
{
   s.writeJSON(os);
   return os;
}

template <>
int JSONNode::val_t<int>() const
{
   return val_int();
}
template <>
float JSONNode::val_t<float>() const
{
   return val_float();
}
template <>
double JSONNode::val_t<double>() const
{
   return val_float();
}
template <>
bool JSONNode::val_t<bool>() const
{
   return val_bool();
}
template <>
std::string JSONNode::val_t<std::string>() const
{
   return val();
}

} // namespace Experimental
} // namespace RooFit
