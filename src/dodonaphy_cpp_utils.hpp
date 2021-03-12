#ifndef __dodonaphy_cpp_utils_hpp__
#define __dodonaphy_cpp_utils_hpp__

#include <stan/model/model_header.hpp>

// #include <>
#include <queue>
#include <stack>

double get_val(double v){ return v; }
double get_val(const stan::math::var& v){ return v.val(); }

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

namespace dodonaphy_model_namespace {

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
	

  // add a fake root above node 0
  int zero_parent = mst_adjacencies[0][0];
  mst_adjacencies.push_back({0,zero_parent});
  mst_adjacencies[0][0] = mst_adjacencies.size()-1;
  for(size_t i=0; i < mst_adjacencies[zero_parent].size(); i++){
    if(mst_adjacencies[zero_parent][i]==0){
      mst_adjacencies[zero_parent][i] = mst_adjacencies.size()-1;
    }
  }
  // location_map[mst_adjacencies.size()-1]=zero_parent;
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

} // dodonaphy_model_namespace

namespace dodonaphy_all_model_namespace {
template <typename T0__, typename T1__, typename T2__, typename T3__>
void
make_peel(const std::vector<T0__>& leaf_r, const std::vector<std::vector<T1__> >& leaf_dir,
            const std::vector<T2__>& int_r, const std::vector<std::vector<T3__> >& int_dir,
            std::vector<std::vector<int> >& peel,
            std::vector<int>& location_map, std::ostream* pstream__) {
  dodonaphy_model_namespace::make_peel(leaf_r, leaf_dir, int_r, int_dir, peel, location_map, pstream__);
}
} // dodonaphy_all_model_namespace

namespace dodonaphy_leaves_model_namespace {

template <typename T0__>
double sum_squared(const std::vector<T0__>& a){
  double ss = 0;
  for(size_t i=0; i < a.size(); i++){
    ss += pow(get_val(a[i]),2);
  }
  return ss;
}

template <typename T0__>
double poincare_distance(const std::vector<T0__>& a, const std::vector<T0__>& b) {
  double ss_a = sum_squared(a);
  double ss_b = sum_squared(b);
  // blob = sum((u-v)**2)) / (1-sum(u**2)) / (1-sum(v**2))
  // acosh( 1 + 2 * blob )
  if(ss_a >= 1.0 || ss_b >= 1.0){
      return std::numeric_limits<double>::max();
    }else{
      double ssq_diff = 0;
      for(size_t i=0; i < a.size(); i++){
        ssq_diff += pow(get_val(a[i])-get_val(b[i]), 2);
      }
    return acosh(1.0 + 2.0 * (ssq_diff/(1.0-ss_a)/(1.0-ss_b)));
    }
}

// Reflection (circle inversion of x through orthogonal circle centered at a).
template <typename T0__>
std::vector<double> isometric_transform(const std::vector<double>& a, const std::vector<T0__>& x) {
  double r2 = sum_squared(a) - 1;
  double denom = 0;
  for(size_t i=0; i < a.size(); i++){
    denom += pow(get_val(x[i]) - a[i], 2);
  }
  std::vector<double> rval(x.size(), 0);
  for(size_t i=0; i < a.size(); i++){
    rval[i] = (r2 / denom) * (get_val(x[i]) - a[i]) + a[i];
  }
  return rval;
}

// Center of inversion circle
template <typename T0__>
std::vector<double> reflection_center(const std::vector<T0__>& mu) {
  std::vector<double> rval(mu.size(), 0);
  double ss = sum_squared(mu);
  for(size_t i=0; i < mu.size(); i++) {
    rval[i] = get_val(mu[i]) / ss;
  }
  return rval;
}

// Euclidean reflection (also hyperbolic) of x Along the geodesic that goes through a and the origin (straight line)
template <typename T0__>
std::vector<double> euc_reflection(const std::vector<T0__>& x, const std::vector<double>& a) {
  double xTa = 0;
  for(size_t i=0; i < x.size(); i++) {
    xTa += get_val(x[i]) * a[i];
  }
  double norm_a_sq = sum_squared(a);
  std::vector<double> proj(x.size(), 0);
  for(size_t i=0; i < x.size(); i++) {
    proj[i] = xTa * a[i] / norm_a_sq;
    proj[i] = 2.0 * proj[i] - get_val(x[i]);
  }
  return proj;
}

// computes the point on the geodesic segment from o to x at half the distance
template <typename T0__, typename T1__>
std::vector<T1__> _halve(const std::vector<T0__>& x) {
  double ss = sum_squared(x);
  std::vector<T1__> rval(x.size(), 0);
  for(size_t i=0; i < x.size(); i++) {
    rval[i] = get_val(x[i]) / (1.0 + sqrt(1.0-ss));
  }
  return rval;
}

// Computes the projection of the origin on the geodesic between a and b
template <typename T0__>
std::vector<T0__> hyp_lca(const std::vector<T0__>& a, const std::vector<T0__>&b) {
  std::vector<double> r = reflection_center(a);
  std::vector<double> b_inv = isometric_transform(r, b);
  std::vector<double> o_inv_ref = euc_reflection(a,b_inv);
  std::vector<double> o_ref = isometric_transform(r, o_inv_ref);
  std::vector<T0__> proj = _halve<double, T0__>(o_ref);
  return proj;
}

/**
 * use geodesic arcs to make a binary tree from a set of leaf node points embedded
 * in hyperbolic space.
 * Output: int_r, int_dir, peel
 */
template <typename T0__, typename T1__>
void
make_peel_geodesics(const std::vector<std::vector<T0__> >& leaf_locs, std::vector<std::vector<T1__> >& int_locs,
            std::vector<std::vector<int> >& peel, std::ostream* pstream__) {

	size_t leaf_node_count = leaf_locs.size();
	size_t node_count = leaf_locs.size() + int_locs.size();
	std::vector< std::vector<u_edge> > edge_list(node_count);
	std::priority_queue<u_edge, std::vector<u_edge>> queue;

  // compute all pairwise leaf distances
	for(size_t i=0; i < leaf_node_count; i++){
		for(size_t j=i+1; j < leaf_node_count; j++){
      double dist_ij = 0;
      dist_ij = poincare_distance(leaf_locs[i], leaf_locs[j]);

      // apply the inverse transform from Matsumoto et al 2020
      dist_ij = log(cosh(dist_ij));
      // use negative of distance so that least dist has largest value in the priority queue
      queue.push({-dist_ij, i, j});
//      std::cerr << "dist " << i << "," << j << ":" << dist_ij << std::endl;
		}
	}

	std::vector<bool> visited(node_count);
  size_t cur_interal = leaf_node_count;
  size_t cur_peel = 0;
  while(cur_interal < node_count){
		u_edge e = queue.top();
		queue.pop();
    if(visited[e.from] || visited[e.to]) continue;

    // create a new internal node to link these
    // use Lemma 4.1 from Chami et al 2020 to get the location for the new node
    size_t int_i = cur_interal-leaf_node_count;
/*
    double theta = leaf_dir[e.from][0] - leaf_dir[e.to][0];
    double xnorm = leaf_r[e.from];
    double ynorm = leaf_r[e.to];
    double alpha = atan((1.0 / sin(theta)) * (xnorm * (pow(ynorm,2) + 1) / (ynorm * (pow(xnorm,2) + 1)) - cos(theta)));
    double R = sqrt(pow((pow(xnorm,2) + 1) / (2.0 * xnorm * cos(alpha)), 2) - 1.0);
    double d_o = 2.0 * atanh(sqrt(pow(R,2)+1)-R);
    int_r[int_i] = d_o;
    int_dir[int_i][0] = leaf_dir[e.from][0] + alpha;
*/
    std::vector<T0__> from_point, to_point;
    from_point = e.from < leaf_node_count ? leaf_locs[e.from] : int_locs[e.from-leaf_node_count];
    to_point = e.to < leaf_node_count ? leaf_locs[e.to] : int_locs[e.to-leaf_node_count];
    int_locs[int_i] = hyp_lca(from_point, to_point);
    peel[cur_peel][0] = e.from+1;
    peel[cur_peel][1] = e.to+1;
    peel[cur_peel][2] = cur_interal+1;
    visited[e.from] = true;
    visited[e.to] = true;
//    double dist1 = poincare_distance

    // add all pairwise distances between the new node and other active nodes
    for(size_t i=0; i < cur_interal; i++){
      if(visited[i]) continue;
      double dist_ij = 0;
      if(i < leaf_node_count){
        dist_ij = poincare_distance(leaf_locs[i], int_locs[int_i]);
      } else {
        dist_ij = poincare_distance(int_locs[i-leaf_node_count], int_locs[int_i]);
      }
      // apply the inverse transform from Matsumoto et al 2020
//      dist_ij = log(cosh(dist_ij));
      // use negative of distance so that least dist has largest value in the priority queue
      queue.push({-dist_ij, i, cur_interal});
    }
//    std::cerr << "peel: " << peel[cur_peel][0] << ", " << peel[cur_peel][1] << " to " << peel[cur_peel][2] << std::endl;

    cur_peel++;
    cur_interal++;
  }
/*
  std::cerr << "leaf positions:\n";
  for(size_t i=0; i < leaf_locs.size(); i++){
    for(size_t j=0; j < leaf_locs[i].size(); j++){
      std::cerr << '\t' << get_val(leaf_locs[i][j]);
    }
    std::cerr << '\n';
  }
  std::cerr << "int positions:\n";
  for(size_t i=0; i < int_locs.size(); i++){
    for(size_t j=0; j < int_locs[i].size(); j++){
      std::cerr << '\t' << get_val(int_locs[i][j]);
    }
    std::cerr << '\n';
  }
*/
}

} // dodonaphy_leaves_model_namespace

#endif // dodonaphy_cpp_utils_hpp

