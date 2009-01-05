// Check that the shadows implement a final overrider for virtual functions.
// See https://savannah.cern.ch/bugs/?32874

class IA
{
public:
virtual ~IA();
virtual void ia() = 0;
};

class Obj : public virtual IA
{
public:
virtual ~Obj();
virtual void ia();
virtual void obj();
};

class ICh : public virtual IA
{
public:
virtual ~ICh();
virtual void ia();
virtual void iCh();
};

class Comp : public Obj, public virtual ICh
{
public:
virtual ~Comp();
virtual void ia();
virtual void comp();
};
