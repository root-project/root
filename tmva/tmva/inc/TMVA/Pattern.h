#pragma once

#include <cstddef> // for size_t
#include <vector>


class Pattern
{
 public:

   typedef typename std::vector<double>::iterator iterator;
   typedef typename std::vector<double>::const_iterator const_iterator;

   Pattern ()
      : m_weight (0)
      {
      }

   ~Pattern ()
      {
      }

   Pattern (const Pattern& other)
      {
         m_input.assign (std::begin (other.m_input), std::end (other.m_input));
         m_output.assign (std::begin (other.m_output), std::end (other.m_output));
         m_weight = other.m_weight;
      }

   Pattern (Pattern&& other)
      {
         m_input = std::move (other.m_input);
         m_output = std::move (other.m_output);
         m_weight = other.m_weight;
      }

   Pattern& operator= (const Pattern& other)
      {
         m_input.assign (std::begin (other.input ()), std::end (other.input ()));
         m_output.assign (std::begin (other.output ()), std::end (other.output ()));
         m_weight = other.m_weight;
         return *this;
      }

   template <typename ItValue>
      Pattern (ItValue inputBegin, ItValue inputEnd, ItValue outputBegin, ItValue outputEnd, double _weight = 1.0)
      : m_input (inputBegin, inputEnd)
      , m_output (outputBegin, outputEnd)
      , m_weight (_weight)
   {
   }

   template <typename ItValue>
      Pattern (ItValue inputBegin, ItValue inputEnd, double outputValue, double _weight = 1.0)
      : m_input (inputBegin, inputEnd)
      , m_weight (_weight)
   {
      m_output.push_back (outputValue);
   }

   template <typename InputContainer, typename OutputContainer>
      Pattern (InputContainer& _input, OutputContainer& _output, double _weight = 1.0)
      : m_input (std::begin (_input), std::end (_input))
      , m_output (std::begin (_output), std::end (_output))
      , m_weight (_weight)
   {
   }

   const_iterator beginInput () const { return m_input.begin (); }
   const_iterator endInput   () const  { return m_input.end (); }
   const_iterator beginOutput () const  { return m_output.begin (); }
   const_iterator endOutput   () const  { return m_output.end (); }

   double weight () const { return m_weight; }
   void weight (double w) { m_weight = w; }

   size_t inputSize () const { return m_input.size (); }
   size_t outputSize () const { return m_output.size (); }

   void addInput (double value) { m_input.push_back (value); }
   void addOutput (double value) { m_output.push_back (value); }

   std::vector<double>& input  () { return m_input; }
   std::vector<double>& output () { return m_output; }
   const std::vector<double>& input  () const { return m_input; }
   const std::vector<double>& output () const { return m_output; }

 private:
   std::vector<double> m_input;
   std::vector<double> m_output;
   double m_weight;
};
