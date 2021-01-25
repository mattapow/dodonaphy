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
template<typename array2D_1, typename array2D_2>
double hyperbolic_dist(double r1, double r2, const array2D_1& directional1, const array2D_1& directional2, double curvature) {
	// compute the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
	double prodsum = 0;
	for(size_t i=0; i < directional1.size(); i++){
		prodsum += directional1[i] * directional2[i];
	}
	// force between numerical -1.0 and 1.0 to eliminate rounding errors
	double iprod = prodsum < -1.0 ? -1.0 : prodsum;
	iprod = iprod > 1.0 ? 1.0 : iprod;

	// hyperbolic 'angle'; force numerical >= 1.0
	double hyper_angle = 2.0*(pow(r1,2) + pow(r2,2) - 2.0*r1*r2*iprod)/((1 - pow(r1,2))*(1 - pow(r2,2)));
	hyper_angle = 1.0 + hyper_angle < 0 ? 0 : hyper_angle;

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
template <typename T0__, typename T1__>
void
make_peel(const std::vector<std::vector<T0__> >& leaf_locs,
              const std::vector<std::vector<T1__> >& int_locs,
              std::vector<std::vector<int> >& peel,
              std::vector<int>& location_map, std::ostream* pstream__) {

	size_t leaf_node_count = leaf_locs.size();
	size_t node_count = leaf_locs.size() + int_locs.size();
	std::vector< std::vector<u_edge> > edge_list(node_count);
	
	for(size_t i=0; i < node_count; i++){
		for(size_t j=i+1; j < node_count; j++){
			double dist_ij = 0;
			if(i < leaf_node_count){
				if(j < leaf_node_count){
					dist_ij = std::numeric_limits<double>::max();
				}else{
					dist_ij = euclidean_dist(i,j-leaf_node_count,leaf_locs,int_locs);
				}
			} else {
				dist_ij = euclidean_dist(i-leaf_node_count,j-leaf_node_count,int_locs,int_locs);
			}
			if(j >= leaf_node_count){
				edge_list[i].emplace_back(dist_ij, i, j);
				edge_list[j].emplace_back(dist_ij, j, i);
			}
		}
	}

		
	// construct a minimum spanning tree among the internal nodes
	std::priority_queue<u_edge, std::vector<u_edge>> queue;
	std::vector<bool> visited(node_count);
	u_edge start_edge(0,0,0);
	queue.push(start_edge);
	std::vector< std::vector<int> > mst_adjacencies(node_count);
	while (!queue.empty()) {
		u_edge e = queue.top();
		queue.pop();
		if(!visited[e.to]){
			mst_adjacencies[e.from].push_back(e.to);
			mst_adjacencies[e.to].push_back(e.from);
			
			visited[e.to] = true;
			for(auto& new_e : edge_list[e.to]){
				if(visited[new_e.to]) continue;
				queue.push(new_e);
			}
		}
	}
	
	// prune internal nodes that don't lead to leaf nodes
	std::stack<size_t> to_check;
	for(size_t n=0; n < mst_adjacencies.size(); n++){
		if(mst_adjacencies[n].size()==1 && mst_adjacencies[n][0] >= leaf_node_count){
			to_check.push(n);			
		}
	}
	std::vector<size_t> unused;
	while(to_check.size()>0){
		size_t n = to_check.top();
		to_check.pop();
		if(mst_adjacencies[n].size()==1 && mst_adjacencies[n][0] >= leaf_node_count){
			size_t neighbour = mst_adjacencies[n][0];
			mst_adjacencies[n].clear();
			for(size_t i=0; i < mst_adjacencies[neighbour].size(); i++){
				if(mst_adjacencies[neighbour][i] == n){
					mst_adjacencies[neighbour].erase(mst_adjacencies[neighbour].begin() + i);
				}				
			}
			unused.push_back(n);
			to_check.push(neighbour);
		}
	}

	// initialize location_map with every node pointing to itself
	for(size_t i=0; i < node_count; i++){
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
				mst_adjacencies[n].erase(mst_adjacencies[n].begin());
				// link up new node
				mst_adjacencies[new_node].push_back(move_1);
				mst_adjacencies[new_node].push_back(move_2);
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
	
	// make peel via pre-order traversal
	visited = std::vector<bool>(node_count);
	std::stack<int> node_stack;
	node_stack.push((int)mst_adjacencies[0][0]); // start with an internal node adjoining one of the leaves
	size_t peelI = peel.size();
	while(!node_stack.empty()) {
		int cur_node = node_stack.top();
		node_stack.pop(); 
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
	}
}

}

#endif  // __hyperphy_hacks__

