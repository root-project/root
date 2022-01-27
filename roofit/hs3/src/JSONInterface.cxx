#include <RooFitHS3/JSONInterface.h>

namespace RooFit {
namespace Experimental {

// RooFit::Experimental::JSONNode::child_iterator_t implementation

template <class Nd>
JSONNode::child_iterator_t<Nd>::child_iterator_t(std::unique_ptr<Impl> impl) : it(std::move(impl))
{
}

template <class Nd>
JSONNode::child_iterator_t<Nd>::child_iterator_t(const child_iterator_t &other) : it(std::move(other.it->mkptr()))
{
}

template <class Nd>
JSONNode::child_iterator_t<Nd>::~child_iterator_t()
{
}

template <class Nd>
JSONNode::child_iterator_t<Nd> &JSONNode::child_iterator_t<Nd>::operator++()
{
   it->forward();
   return *this;
}
template <class Nd>
JSONNode::child_iterator_t<Nd> &JSONNode::child_iterator_t<Nd>::operator--()
{
   it->backward();
   return *this;
}

template <class Nd>
Nd &JSONNode::child_iterator_t<Nd>::operator*() const
{
   return it->current();
}
template <class Nd>
Nd &JSONNode::child_iterator_t<Nd>::operator->() const
{
   return it->current();
}
template <class Nd>
bool JSONNode::child_iterator_t<Nd>::operator!=(const child_iterator_t &that) const
{
   return !this->it->equal(*that.it);
}
template <class Nd>
bool JSONNode::child_iterator_t<Nd>::operator==(const child_iterator_t &that) const
{
   return this->it->equal(*that.it);
}
} // namespace Experimental
} // namespace RooFit

namespace {
template <class Nd>
class childItImpl : public RooFit::Experimental::JSONNode::child_iterator_t<Nd>::Impl {
public:
   enum POS { BEGIN, END };
   Nd &node;
   size_t pos;
   using child_iterator = RooFit::Experimental::JSONNode::child_iterator_t<Nd>;
   childItImpl(Nd &n, POS p) : node(n), pos(p == BEGIN ? 0 : n.num_children()) {}
   childItImpl(Nd &n, size_t p) : node(n), pos(p) {}
   childItImpl(const childItImpl &other) : node(other.node), pos(other.pos) {}
   virtual std::unique_ptr<typename child_iterator::Impl> mkptr() const override
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
};
} // namespace

namespace RooFit {
namespace Experimental {

JSONNode::child_iterator JSONNode::childIteratorBegin()
{
   return child_iterator(std::make_unique<::childItImpl<JSONNode>>(*this, ::childItImpl<JSONNode>::BEGIN));
}
JSONNode::child_iterator JSONNode::childIteratorEnd()
{
   return child_iterator(std::make_unique<::childItImpl<JSONNode>>(*this, ::childItImpl<JSONNode>::END));
}
JSONNode::const_child_iterator JSONNode::childConstIteratorBegin() const
{
   return const_child_iterator(
      std::make_unique<::childItImpl<const JSONNode>>(*this, ::childItImpl<const JSONNode>::BEGIN));
}
JSONNode::const_child_iterator JSONNode::childConstIteratorEnd() const
{
   return const_child_iterator(
      std::make_unique<::childItImpl<const JSONNode>>(*this, ::childItImpl<const JSONNode>::END));
}

template class JSONNode::child_iterator_t<JSONNode>;
template class JSONNode::child_iterator_t<const JSONNode>;

JSONNode::children_view JSONNode::children()
{
   return children_view(this->childIteratorBegin(), this->childIteratorEnd());
}
JSONNode::const_children_view JSONNode::children() const
{
   return const_children_view(this->childConstIteratorBegin(), this->childConstIteratorEnd());
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
