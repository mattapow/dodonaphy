#ifndef __hyperphy_hacks__
#define __hyperphy_hacks__

#include <stan/model/model_header.hpp>
#include <queue>
#include <stack>

namespace dodonaphy_model_namespace {

double get_val(double v){ return v; }
double get_val(const stan::math::var& v){ return v.val(); }

// computes euclidean distance
template<typename array2D_1, typename array2D_2>
double euclidean_dist( size_t node_1, size_t node_2, const array2D_1& locs_1, const array2D_2& locs_2) {
    double dist = 0;
    for(size_t i=0; i < locs_1.size(); i++) {
	for(size_t j=0; j < locs_1.size(); j++) {
        	dist += pow(get_val(locs_1[node_1][i]) - get_val(locs_2[node_2][i]), 2);
	}
    }
    return sqrt(dist);
}

/**
 * hyperbolic distance function, translated to C++ from the R hydra package
 */
template <typename T0__, typename T1__>
double hyperbolic_distance(T0__ loc_r1, const std::vector<T0__>& loc1, T1__ loc_r2, const std::vector<T1__>& loc2, double curvature) {
	// compute the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
	double prodsum = 0;
	for(size_t i=0; i < loc1.size(); i++){
		prodsum += get_val(loc1[i]) * get_val(loc2[i]);
	}
	// force between numerical -1.0 and 1.0 to eliminate rounding errors
	double iprod = prodsum < -1.0 ? -1.0 : prodsum;
	iprod = iprod > 1.0 ? 1.0 : iprod;

	// hyperbolic 'angle'; force numerical >= 1.0
  double r1 = get_val(loc_r1);
  double r2 = get_val(loc_r2);
	double hyper_angle = 2.0*(pow(r1,2) + pow(r2,2) - 2.0*r1*r2*iprod)/((1 - pow(r1,2))*(1 - pow(r2,2)));
	hyper_angle = 1.0 + (hyper_angle < 0 ? 0 : hyper_angle);

	// final hyperbolic distance between points i and j
	double distance = 1/sqrt(curvature) * acosh(hyper_angle);
	return distance;
}

class u_edge {
 public:
 	u_edge(double distance, size_t node_1, size_t node_2) : distance(distance), from(node_1), to(node_2) {}
	double distance;
	size_t from;
	size_t to;
};

constexpr bool operator<(const u_edge& e1, const u_edge& e2) {
	return e1.distance < e2.distance;
}


/**
 * using a set of points embedded into a continuous space, identify a minimum spanning tree that
 * connects all leaf nodes, make it a binary tree, and store a post-order traversal in "peel".
 * internal nodes that are unused in the MST can have their coordinates reassigned. Any reassignments
 * are stored in "location_map"
 */
template <typename T0__, typename T1__, typename T2__, typename T3__>
void
make_peel(const std::vector<T0__>& leaf_r, const std::vector<std::vector<T1__> >& leaf_dir,
            const std::vector<T2__>& int_r, const std::vector<std::vector<T3__> >& int_dir,
            std::vector<std::vector<int> >& peel,
            std::vector<int>& location_map, std::ostream* pstream__) {

	size_t leaf_node_count = leaf_r.size();
	size_t node_count = leaf_r.size() + int_r.size();
	std::vector< std::vector<u_edge> > edge_list(node_count);
/*
  std::cerr << "leaf nodes: " << leaf_locs.size() << std::endl;
  std::cerr << "internal nodes: " << int_locs.size() << std::endl;
  for(size_t i=0; i < int_locs.size(); i++){
    for(size_t j=0; j < int_locs[i].size(); j++){
      std::cerr << '\t' << int_locs[i][j];
    }
    std::cerr << '\n';
  }
*/
	for(size_t i=0; i < node_count; i++){
		for(size_t j=std::max(i+1,leaf_node_count); j < node_count; j++){
      double dist_ij = 0;

      if(i < leaf_node_count){
        // leaf to internal
        dist_ij = hyperbolic_distance(leaf_r[i], leaf_dir[i], int_r[j-leaf_node_count], int_dir[j-leaf_node_count], 1.0);
      }else{
        // internal to internal
        size_t i_node = i - leaf_node_count;
        dist_ij = hyperbolic_distance(int_r[i_node], int_dir[i_node], int_r[j-leaf_node_count], int_dir[j-leaf_node_count], 1.0);
      }

      // apply the inverse transform from Matsumoto et al 2020
      dist_ij = log(cosh(dist_ij));
      // use negative of distance so that least dist has largest value in the priority queue
      edge_list[i].emplace_back(-dist_ij, i, j);
      edge_list[j].emplace_back(-dist_ij, j, i);
//      std::cerr << "dist " << i << "," << j << ":" << dist_ij << std::endl;
		}
	}

		
	// construct a minimum spanning tree among the internal nodes
	std::priority_queue<u_edge, std::vector<u_edge>> queue;
	std::vector<bool> visited(node_count);
	u_edge start_edge(0,0,0);
	queue.push(start_edge);
	std::vector< std::vector<int> > mst_adjacencies(node_count);
  size_t visited_count = 0;
  size_t open_slots = 0;
	while (!queue.empty() && visited_count < node_count) {
		u_edge e = queue.top();
		queue.pop();
    // ensure the destination node has not been visited yet
    // internal nodes can have up to 3 adjacencies, of which at least one must be internal
    // leaf nodes can only have a single edge in the MST
    bool is_valid = true;
    if(visited[e.to]) is_valid = false;
    if(e.from < leaf_node_count && mst_adjacencies[e.from].size() > 0) is_valid = false;
    if(e.to < leaf_node_count && mst_adjacencies[e.to].size() > 0) is_valid = false;
    if(e.from >= leaf_node_count){
      if(mst_adjacencies[e.from].size()==2){
        bool found_internal = e.to >= leaf_node_count;
        if(mst_adjacencies[e.from][0] >= leaf_node_count) found_internal = true;
        if(mst_adjacencies[e.from][1] >= leaf_node_count) found_internal = true;
        if(!found_internal) is_valid = false;
      }else if(mst_adjacencies[e.from].size()==3){
        is_valid = false; // this node is already full
      }
    }
    // don't use the last open slot unless this is the last node
    if(open_slots==1 && e.to < leaf_node_count && visited_count < node_count - 1) is_valid = false;
		if(is_valid) {
      if(e.to != e.from){
			  mst_adjacencies[e.from].push_back(e.to);
			  mst_adjacencies[e.to].push_back(e.from);
      }

      // a new internal node has room for 2 more adjacencies
      if(e.to >= leaf_node_count) open_slots += 2;
      if(e.from >= leaf_node_count) open_slots--; // one slot consumed
			
			visited[e.to] = true;
      visited_count++;
			for(auto& new_e : edge_list[e.to]){
				if(visited[new_e.to]) continue;
				queue.push(new_e);
			}
		}
	}
	
	// prune internal nodes that don't create a bifurcation
	std::stack<size_t> to_check;
	for(size_t n=leaf_node_count; n < mst_adjacencies.size(); n++){
		if(mst_adjacencies[n].size() < 3){
			to_check.push(n);			
		}
	}
	std::vector<size_t> unused;
	while(to_check.size()>0){
		size_t n = to_check.top();
		to_check.pop();
		if(mst_adjacencies[n].size()==1){
			size_t neighbour = mst_adjacencies[n][0];
			mst_adjacencies[n].clear();
			for(size_t i=0; i < mst_adjacencies[neighbour].size(); i++){
				if(mst_adjacencies[neighbour][i] == n){
					mst_adjacencies[neighbour].erase(mst_adjacencies[neighbour].begin() + i);
				}				
			}
			unused.push_back(n);
			to_check.push(neighbour);
		}else if(mst_adjacencies[n].size()==2){
			size_t n1 = mst_adjacencies[n][0];
			size_t n2 = mst_adjacencies[n][1];
			mst_adjacencies[n].clear();
			for(size_t i=0; i < mst_adjacencies[n1].size(); i++){
				if(mst_adjacencies[n1][i] == n){
					mst_adjacencies[n1][i] = n2;
				}				
			}
			for(size_t i=0; i < mst_adjacencies[n2].size(); i++){
				if(mst_adjacencies[n2][i] == n){
					mst_adjacencies[n2][i] = n1;
				}				
			}
			unused.push_back(n);
		}
	}
//  std::cerr << "unused: " << unused.size() << std::endl;

	// initialize location_map with every node pointing to itself
	for(size_t i=0; i < location_map.size(); i++){
		location_map[i]=i;
	}

	// transform the MST into a binary tree.
	// find any nodes with more than three adjacencies and introduce
	// intermediate nodes to reduce the number of adjacencies
	if(unused.size()>0){
		for(size_t n=0; n < mst_adjacencies.size(); n++){
			while(mst_adjacencies[n].size() > 3){
				size_t new_node = unused.back();
				unused.erase(unused.end()-1);
				size_t move_1 = mst_adjacencies[n].back();
				size_t move_2 = mst_adjacencies[n].front();
				mst_adjacencies[n].erase(mst_adjacencies[n].end()-1);
				mst_adjacencies[n][0] = new_node;
				// link up new node
				mst_adjacencies[new_node].push_back(move_1);
				mst_adjacencies[new_node].push_back(move_2);
        mst_adjacencies[new_node].push_back(n);
				for(auto& move : {move_1,move_2}){
					for(size_t i=0; i < mst_adjacencies[move].size(); i++){
						if(mst_adjacencies[move][i] == n){
							mst_adjacencies[move][i] = new_node;
						}
					}
				}
				// map the location for the new node to the original node
				location_map[new_node] = n;
			}
		}
	}
//  std::cerr << "tree pruned of unused nodes.\n";

  // update the location map - handles multiple reassignments
  for(size_t i=0; i<location_map.size(); i++){
    size_t parent = location_map[i];
    while(parent != location_map[parent]){
      location_map[parent] = location_map[location_map[parent]];
      parent = location_map[parent];
    }
    location_map[i] = parent;
  }
//  std::cerr << "loc map updated.\n";

  // add a fake root above node 0
  int zero_parent = mst_adjacencies[0][0];
  mst_adjacencies.push_back({0,zero_parent});
  mst_adjacencies[0][0] = mst_adjacencies.size()-1;
  for(size_t i=0; i < mst_adjacencies[zero_parent].size(); i++){
    if(mst_adjacencies[zero_parent][i]==0){
      mst_adjacencies[zero_parent][i] = mst_adjacencies.size()-1;
    }
  }
  location_map[mst_adjacencies.size()-1]=zero_parent;
//  std::cerr << "fake root added.\n";
	
	// make peel via pre-order traversal
	visited = std::vector<bool>(node_count);
	std::stack<int> node_stack;
	node_stack.push((int)mst_adjacencies[0][0]); // start with the fake root
	size_t peelI = peel.size();
	while(!node_stack.empty()) {
		int cur_node = node_stack.top();
		node_stack.pop(); 
//    std::cerr << "peeling for " << cur_node << "\n";
		if(mst_adjacencies[cur_node].size() < 2) {
			continue;  // leaf node, nothing to do
		}
		// remove already-visited nodes from adjacencies, leaving just two children 
		for(auto iter = mst_adjacencies[cur_node].begin(); iter != mst_adjacencies[cur_node].end(); iter++){
			if(visited[*iter]){
				mst_adjacencies[cur_node].erase(iter);
			}
		}
		// peel entries are child, child, parent
		// cur_node should always have two adjacencies
		peelI--;
		peel[peelI] = {(int)mst_adjacencies[cur_node][0],(int)mst_adjacencies[cur_node][1],(int)cur_node};
		node_stack.push(peel[peelI][0]);
		node_stack.push(peel[peelI][1]);
    visited[cur_node] = true;
	}
  for(size_t i=0; i < peel.size(); i++){
    for(size_t j=0; j < peel[i].size(); j++) peel[i][j]++;
//    std::cerr << "peel " << i << ": " << peel[i][0] << ", " << peel[i][1] << ", " << peel[i][2] << "\n";
  }
  for(size_t i=0; i < location_map.size(); i++){
    location_map[i]++;
  }
}

}

#endif  // __hyperphy_hacks__

