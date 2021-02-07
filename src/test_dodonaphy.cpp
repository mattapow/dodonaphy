#include <dodonaphy_cpp_utils.hpp>
#include <vector>
#include <iostream>

using namespace dodonaphy_model_namespace;
using namespace dodonaphy_leaves_model_namespace;

void test_make_peel() {
  std::vector<double> leaf_r = {0.1745793,0.1805378,0.2308089,0.1782356,0.1911079,0.1821260};
  std::vector<std::vector<double> > leaf_locs(6);
  leaf_locs[0] = {0.2158479,-0.8665832,0.4499369};
  leaf_locs[1] = {-0.6753461,-0.7025528,0.2243372};
  leaf_locs[2] = {-0.2025646,0.2051602,-0.9575369};
  leaf_locs[3] = {0.4863753,-0.3517850,-0.7998040};
  leaf_locs[4] = {-0.4961180,0.5966320,0.6307910};
  leaf_locs[5] = {0.7157135,0.3218363,0.6198190};

  std::vector<double> int_r = {0.09,0.08,0.11,0.03};
  std::vector<std::vector<double> > int_locs(4);
  int_locs[0] = {-0.6853461,-0.7525528,0.2143372};
  int_locs[1] = {0.4563753,-0.3717850,-0.7198040};
  int_locs[2] = {-0.4461180,0.5866320,0.6507910};
  int_locs[3] = {0.7257135,0.3118363,0.6098190};
  std::vector<std::vector<int> > peel(5,std::vector<int>(3));
  std::vector<int> location_map(int_locs.size()+leaf_locs.size()+1);

  make_peel(leaf_r, leaf_locs, int_r, int_locs, peel, location_map, &std::cout);
}

void test_make_peel_geodesics() {
  std::vector<std::vector<double> > leaf_locs(6);

  leaf_locs[0] = {0.06644704,0.18495312,-0.03839615};
  leaf_locs[1] = {-0.12724270,-0.02395994,0.20075329};
  auto lca = hyp_lca(leaf_locs[0],leaf_locs[1]);
  double d0_1 = poincare_distance(leaf_locs[0], leaf_locs[1]);
  double d0_lca = poincare_distance(leaf_locs[0], lca);
  double d1_lca = poincare_distance(leaf_locs[1], lca);
  double d0_lca_d1 = d0_lca + d1_lca;
  // d0_1 should be the same as d0_lca_d1


  leaf_locs[0] = {0.05909264,0.16842421,-0.03628194};
  leaf_locs[1] = {0.08532969,-0.07187002,0.17884444};
  leaf_locs[2] = {-0.11422830,0.01955054,0.14127290};
  leaf_locs[3] = {-0.06550432,0.07029946,-0.14566249};
  leaf_locs[4] = {-0.07060744,-0.12278600,-0.17569585};
  leaf_locs[5] = {0.11386343,-0.03121063,-0.18112418};

  double d01 = log(cosh(poincare_distance(leaf_locs[0], leaf_locs[1])));
  double d02 = log(cosh(poincare_distance(leaf_locs[0], leaf_locs[2])));
  double d03 = log(cosh(poincare_distance(leaf_locs[0], leaf_locs[3])));

  std::vector<std::vector<double> > int_locs(5, std::vector<double>(1));
  std::vector<std::vector<int> > peel(5,std::vector<int>(3));

  make_peel_geodesics(leaf_locs, int_locs, peel, &std::cout);

  std::cerr << "test passed\n";
}

int main(int argc, char* argv[]) {
  test_make_peel_geodesics();
  test_make_peel();
}