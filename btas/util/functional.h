/*
 * functional.h
 *
 *  Created on: Dec 28, 2013
 *      Author: evaleev
 */

#ifndef FUNCTIONAL_H_
#define FUNCTIONAL_H_

namespace btas {

  /// Computes T -> T
  template <typename T>
  struct identity {
      typedef T type;
      T& operator()(T& x) const { return x; }
      const T& operator()(const T& x) const { return x; }
  };

  template <typename Pair> struct first_of_pair;
  template <typename Pair> struct second_of_pair;
  /// Computes pair<T1,T2> -> T1
  template <typename T1, typename T2>
  struct first_of_pair< std::pair<T1,T2> > {
      typedef T1 type;
      type& operator()(std::pair<T1,T2>& x) const { return x.first; }
      const type& operator()(const std::pair<T1,T2>& x) const { return x.first; }
  };

  /// Computes pair<T1,T2> -> T2
  template <typename T1, typename T2>
  struct second_of_pair< std::pair<T1,T2> > {
      typedef T2 type;
      type& operator()(std::pair<T1,T2>& x) const { return x.second; }
      const type& operator()(const std::pair<T1,T2>& x) const { return x.second; }
  };

  /// returns the first element of a tuple
  template <typename ...Types>
  auto first(const std::tuple<Types...>& x) -> decltype(std::get<0>(x)) {
    return std::get<0>(x);
  }
  /// returns the second element of a tuple
  template <typename ...Types>
  auto second(const std::tuple<Types...>& x) -> decltype(std::get<1>(x)) {
    return std::get<1>(x);
  }
  /// returns the third element of a tuple
  template <typename ...Types>
  auto third(const std::tuple<Types...>& x) -> decltype(std::get<2>(x)) {
    return std::get<2>(x);
  }
  /// returns the fourth element of a tuple
  template <typename ...Types>
  auto fourth(const std::tuple<Types...>& x) -> decltype(std::get<3>(x)) {
    return std::get<3>(x);
  }

}


#endif /* FUNCTIONAL_H_ */
