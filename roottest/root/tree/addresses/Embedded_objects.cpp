#include <iostream>
#include "Embedded_objects.h"

Normal_objects::Normal_objects() {}
Normal_objects::~Normal_objects() {}
void Normal_objects::initData(int ind) {
   i = 0;
   j = -1;
   k = -2;
   l = -3;
   m = -4;
   emb.initData(ind);
}
void Normal_objects::dump() const { emb.dump(); }

Embedded_objects::Embedded_objects() {}
Embedded_objects::~Embedded_objects() {}

Embedded_objects::EmbeddedClasses::EmbeddedClasses() 
{
  m_pemb1 = 0;
  m_pemb2 = 0;
  m_pemb3 = 0;
}
Embedded_objects::EmbeddedClasses::~EmbeddedClasses() 
{
  if ( m_pemb1 ) delete m_pemb1;
  if ( m_pemb2 ) delete m_pemb2;
  if ( m_pemb3 ) delete m_pemb3;
}

void Embedded_objects::initData(int i) {
  Embedded_objects& o = *this;
  o.m_emb1.i                     = i;
  o.m_emb2.m_embed1.i            = 10*i;
  o.m_emb2.d                     = double(100*i);
  o.m_emb3.m_embed2.m_embed1.i   = 1000*i;
  o.m_emb3.m_embed2.d            = double(10000*i);
//   fprintf(stderr,"location of o.m_emb3 and o.m_emb3.f %p %p\n",
//           &o.m_emb3,&o.m_emb3.f);
  o.m_emb3.f                     = float( 100000*i);

  m_embedded.m_pemb1 = new EmbeddedClasses::Embedded1;
  m_embedded.m_pemb2 = new EmbeddedClasses::Embedded2;
  m_embedded.m_pemb3 = new EmbeddedClasses::Embedded3;
  m_embedded.m_pemb1->i                   = 2*i;
  m_embedded.m_pemb2->m_embed1.i          = 20*i;
  m_embedded.m_pemb2->d                   = double(200*i);
  m_embedded.m_pemb3->m_embed2.m_embed1.i = 2000*i;
  m_embedded.m_pemb3->m_embed2.d          = double(20000*i);
  m_embedded.m_pemb3->f                   = float( 200000*i);
}

void Embedded_objects::dump() const {
  const Embedded_objects& o = *this;
  std::cout << "       embedded: ";
  std::cout << "  1.i:"      << o.m_emb1.i;
  std::cout << "  2.1.i:"    << o.m_emb2.m_embed1.i;
  std::cout << "  2.d:"      << o.m_emb2.d;
  std::cout << "  3.2.1.i:"  << o.m_emb3.m_embed2.m_embed1.i;
  std::cout << "  3.2.d:"    << o.m_emb3.m_embed2.d;
  std::cout << "  3.f:"      << o.m_emb3.f;
  std::cout << std::endl;
  std::cout << "       embedded*:";
  std::cout << " *1.i:"     << m_embedded.m_pemb1->i;
  std::cout << " *2.1.i:"   << m_embedded.m_pemb2->m_embed1.i;
  std::cout << " *2.d:"     << m_embedded.m_pemb2->d;
  std::cout << " *3.2.1.i:" << m_embedded.m_pemb3->m_embed2.m_embed1.i;
  std::cout << " *3.2.d:"   << m_embedded.m_pemb3->m_embed2.d;
  std::cout << " *3.f:"     << m_embedded.m_pemb3->f;
  std::cout << std::endl;
}

Embedded_objects::EmbeddedClasses::Embedded1::Embedded1() 
{
  i = 0;
}
Embedded_objects::EmbeddedClasses::Embedded1::~Embedded1() 
{
}

Embedded_objects::EmbeddedClasses::Embedded2::Embedded2() 
{
  d = 0.0;
}
Embedded_objects::EmbeddedClasses::Embedded2::~Embedded2() 
{
}

Embedded_objects::EmbeddedClasses::Embedded3::Embedded3() 
{
  f = 0.0;
}
Embedded_objects::EmbeddedClasses::Embedded3::~Embedded3() 
{
}

