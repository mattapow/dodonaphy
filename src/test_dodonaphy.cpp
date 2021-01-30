#include <dodonaphy_cpp_utils.hpp>
#include <vector>
#include <iostream>

using namespace dodonaphy_model_namespace;

int main(int argc, char* argv[]) {
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