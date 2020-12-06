/** Barnes-hut application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Martin Burtscher <burtscher@txstate.edu>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include <limits>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <stdio.h>
#include <strings.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

namespace {
const char* name = "Barneshut N-Body Simulator";
const char* desc =
  "Simulation of the gravitational forces in a galactic cluster using the "
  "Barnes-Hut n-body algorithm\n";
const char* url = "barneshut";

int nbodies=100;
int ntimesteps=1;
int seed=0;

// To support 3-dimensions
struct Point {
  double x, y, z;
  Point() : x(0.0), y(0.0), z(0.0) { }
  Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) { }
  explicit Point(double v) : x(v), y(v), z(v) { }
//-----------------------------------DEFINING C++ OPERATORS--------------------------------------------------------
  double operator[](const int index) const {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  double& operator[](const int index) {
    switch (index) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  bool operator==(const Point& other) {
    if (x == other.x && y == other.y && z == other.z)
      return true;
    return false;
  }

  bool operator!=(const Point& other) {
    return !operator==(other);
  }

  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Point& operator*=(double value) {
    x *= value;
    y *= value;
    z *= value;
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, const Point& p) {
  os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
  return os;
}
//----------------------------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------DEFINING OCTREE STRUCTURE-----------------------------------------------------    
/**
 * A node in an octree is either an internal node or a body (leaf/external).
 */
struct Octree {
  bool isleaf;
  virtual ~Octree() { }
//  virtual bool isLeaf() const = 0;
  bool isLeaf() const{ return isleaf; }
};

// INTERNAL WILL HAVE CHILDREN
struct OctreeInternal : Octree {
  Octree* child[8];
  Point pos;
  double mass;
  OctreeInternal(Point _pos) : pos(_pos), mass(0.0) {
	isleaf = false;
    bzero(child, sizeof(*child) * 8);
  }
  OctreeInternal(){
  }
  void assign(Point _pos){
	pos = _pos;
	mass = 0;
	isleaf = false;
    bzero(child, sizeof(*child) * 8);
  }
  bool isLeaf() const {
    return false;
  }
  virtual ~OctreeInternal() {
    for (int i = 0; i < 8; i++) {
      if (child[i] != NULL && !child[i]->isLeaf()) {
        delete child[i];
      }
    }
  }
};
    
//EXTERNAL NODE STRUCTURE    
// Each body is assigned the properties we need for the simulation
struct Body : Octree {
  Point pos;
  Point vel;
  Point acc;
  double mass;
  Body() { isleaf = true; }
  bool isLeaf() const { return true; }
  ~Body(){}
};

//This operator (<<) applied to an output stream is known as insertion operator, and performs formatted output:
std::ostream& operator<<(std::ostream& os, const Body& b) {
  os << "(pos:" << b.pos
     << " vel:" << b.vel
     << " acc:" << b.acc
     << " mass:" << b.mass << ")";
  return os;
}

//----------------------------------------------------CREATING THE STRUCTURE FOR BOUNDING BOXES-------------------------------------------    
// Creating our own structure for bounding squares.
struct BoundingBox {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) { }
  BoundingBox() :
    min(std::numeric_limits<double>::max()),
    max(std::numeric_limits<double>::min()) { }

  void merge(const BoundingBox& other) {
    for (int i = 0; i < 3; i++) {
      if (other.min[i] < min[i])
        min[i] = other.min[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other.max[i] > max[i])
        max[i] = other.max[i];
    }
  }

  void merge(const Point& other) {
    for (int i = 0; i < 3; i++) {
      if (other[i] < min[i])
        min[i] = other[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other[i] > max[i])
        max[i] = other[i];
    }
  }

  double diameter() const {
    double diameter = max.x - min.x;
    for (int i = 1; i < 3; i++) {
      double t = max[i] - min[i];
      if (diameter < t)
        diameter = t;
    }
    return diameter;
  }

  double radius() const {
    return diameter() / 2;
  }

  Point center() const {
    return Point(
        (max.x + min.x) * 0.5,
        (max.y + min.y) * 0.5,
        (max.z + min.z) * 0.5);
  }
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& b) {
  os << "(min:" << b.min << " max:" << b.max << ")";
  return os;
}
//-------------------------------------------------SIMULATION CONFIGURATION SETTINGS-------------------------------------------------------
struct Config {
  const double dtime; // length of one time step
  const double eps; // potential softening parameter
  const double tol; // tolerance for stopping recursion, <0.57 to bound error
  const double dthf, epssq, itolsq;
  Config() :
    dtime(0.5),
    eps(0.05),
    tol(0.025),
    dthf(dtime * 0.5),
    epssq(eps * eps),
    itolsq(1.0 / (tol * tol))  { }
};

Config config;

    
inline int getIndex(const Point& a, const Point& b) {
  int index = 0;
  if (a.x < b.x)
    index += 1;
  if (a.y < b.y)
    index += 2;
  if (a.z < b.z)
    index += 4;
  return index;
}

inline void updateCenter(Point& p, int index, double radius) {
  for (int i = 0; i < 3; i++) {
    double v = (index & (1 << i)) > 0 ? radius : -radius;
    p[i] += v;
  }
}

//--------------------------------------DEFINIING A STACK TEMPLATE FOR SIMULATION STORAGE--------------------------------------------
template<typename T, int N>
class Stack{
	T elem[N+1];
	int sz;
public:
	typedef T* iterator;
public:
	Stack() : sz(0){
	}
	iterator begin(){
		return elem;
	}
	iterator end(){
		return elem+sz;
	}
	bool empty()const{
		return sz == 0;
	}
	int size()const{
		return sz;
	}
	T& operator[](int i){
		return elem[i];
	}
	void push_back(const T& e){
		if(sz >= N) throw std::runtime_error("too many elements");
		elem[sz++] = e;
	}
	void pop_back(){
		sz--;
	}
	const T& back()const{
		return elem[sz-1];
	}
};

typedef std::vector<Body> Bodies;

//-------------------------------------------BUILDING THE STRUCTURE TO ALLOW OCTREE TO BE BUILT------------------------------------------
    
struct BuildOctree {
  // NB: only correct when run sequentially
  typedef int tt_does_not_need_stats;

  OctreeInternal* root;
  double root_radius;

  BuildOctree(OctreeInternal* _root, double radius) :
    root(_root),
    root_radius(radius) { }

  template<typename Context>
  void operator()(Body* b, Context&) {
    insert(b, root, root_radius);
  }

  void insert(Body* b, OctreeInternal* node, double radius) {
    int index = getIndex(node->pos, b->pos);
    assert(!node->isLeaf());

    Octree *child = node->child[index];
    
    if (child == NULL) {
      node->child[index] = b;
      return;
    }
    
    radius *= 0.5;
    if (child->isLeaf()) {
      // Expand leaf
      Body* n = static_cast<Body*>(child);
      Point new_pos(node->pos);
      updateCenter(new_pos, index, radius);
      OctreeInternal* new_node = new OctreeInternal(new_pos);

      assert(n->pos != b->pos);
      
      insert(b, new_node, radius);
      insert(n, new_node, radius);
      node->child[index] = new_node;
    } else {
      OctreeInternal* n = static_cast<OctreeInternal*>(child);
      insert(b, n, radius);
    }
  }
};

//----------------------------------------CENTER OF GRAVITY/ CENTER OF MASS FUNCTION--------------------------------------------------------
    
struct ComputeCenterOfMass {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  OctreeInternal* root;

  ComputeCenterOfMass(OctreeInternal* _root) : root(_root) { }

  void operator()() {
    root->mass = recurse(root);
  }

private:
  double recurse(OctreeInternal* node) {
    double mass = 0.0;
    int index = 0;
    Point accum;
    
    for (int i = 0; i < 8; i++) {
      Octree* child = node->child[i];
      if (child == NULL)
        continue;

      // Reorganize leaves to be denser up front 
      if (index != i) {
        node->child[index] = child;
        node->child[i] = NULL;
      }
      index++;
      
      double m;
      const Point* p;
      if (child->isLeaf()) {
        Body* n = static_cast<Body*>(child);
        m = n->mass;
        p = &n->pos;
      } else {
        OctreeInternal* n = static_cast<OctreeInternal*>(child);
        m = recurse(n);
        p = &n->pos;
      }

      mass += m;
      for (int j = 0; j < 3; j++){
        accum[j] += (*p)[j] * m;
      }
    }

    node->mass = mass;
    
    if (mass > 0.0) {
      double inv_mass = 1.0 / mass;
      for (int j = 0; j < 3; j++){
        node->pos[j] = accum[j] * inv_mass;
      }
    }

    return mass;
  }
};
//--------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------COMPUTING FORCES------------------------------------------------------------------------
    
struct ComputeForces {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_context;

  OctreeInternal* top;
  double diameter;
  double root_dsq;

  ComputeForces(OctreeInternal* _top, double _diameter) :
    top(_top),
    diameter(_diameter) {
    root_dsq = diameter * diameter * config.itolsq;
  }
  
  template<typename Context>
  void operator()(Body* bb, Context&) {
    Body& b = *bb;
    Point p = b.acc;
    for (int i = 0; i < 3; i++){
      b.acc[i] = 0;
    }

    //recurse(b, top, root_dsq);
    iterate(b, root_dsq);
    for (int i = 0; i < 3; i++){
      b.vel[i] += (b.acc[i] - p[i]) * config.dthf;
    }
  }

  void recurse(Body& b, Body* node, double dsq) {
    Point p;
    for (int i = 0; i < 3; i++){
      p[i] = node->pos[i] - b.pos[i];
    }

    double psq = p.x * p.x + p.y * p.y + p.z * p.z;
    psq += config.epssq;
    double idr = 1 / sqrt(psq);
    // b.mass is fine because every body has the same mass
    double nphi = b.mass * idr;
    double scale = nphi * idr * idr;
    for (int i = 0; i < 3; i++){
      b.acc[i] += p[i] * scale;
    }
  }

  struct Frame {
    double dsq;
    OctreeInternal* node;
	Frame(){}
    Frame(OctreeInternal* _node, double _dsq) : dsq(_dsq), node(_node) { }
  };

  void iterate(Body& b, double root_dsq) {
//    std::vector<Frame> stack;
    Stack<Frame, 100> stack;
    stack.push_back(Frame(top, root_dsq));

    Point p;
    while (!stack.empty()) {
      Frame f = stack.back();
      stack.pop_back();

      for (int i = 0; i < 3; i++){
        p[i] = f.node->pos[i] - b.pos[i];
      }

      double psq = p.x * p.x + p.y * p.y + p.z * p.z;
      if (psq >= f.dsq) {
        // Node is far enough away, summarize contribution
        psq += config.epssq;
        double idr = 1 / sqrt(psq);
        double nphi = f.node->mass * idr;
        double scale = nphi * idr * idr;
        for (int i = 0; i < 3; i++){
#pragma tls forkpoint id 1 cannot
          b.acc[i] += p[i] * scale;
#pragma tls joinpoint id 1
        }
#pragma tls barrierpoint id 1
        continue;
      }

      double dsq = f.dsq * 0.25;

      for (int i = 0; i < 8; i++) {
        Octree *next = f.node->child[i];
        if (next == NULL)
          break;
        if (next->isLeaf()) {
          // Check if it is me
          if (&b != next) {
            recurse(b, static_cast<Body*>(next), dsq);
          }
        } else {
          stack.push_back(Frame(static_cast<OctreeInternal*>(next), dsq));
        }
      }
    }
  }
};
    
//----------------------------------------------------------UPDATE BODIES/ADVANCE BODIES-------------------------------------------------    
    
// Update bodies' position and velocity
struct AdvanceBodies {
  // Optimize runtime for no conflict case
  typedef int tt_does_not_need_context;

  AdvanceBodies() { }

  template<typename Context>
  void operator()(Body* bb, Context&) {
    Body& b = *bb;
    Point dvel(b.acc);
    dvel *= config.dthf;

    Point velh(b.vel);
    velh += dvel;

    for (int i = 0; i < 3; ++i){
      b.pos[i] += velh[i] * config.dtime;
    }

    for (int i = 0; i < 3; ++i){
      b.vel[i] = velh[i] + dvel[i];
    }
  }
};


struct ReduceBoxes {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  BoundingBox& initial;

  ReduceBoxes(BoundingBox& _initial): initial(_initial) { }

  template<typename Context>
  void operator()(Body* b, Context&) {
    initial.merge(b->pos);
  }
};

double nextDouble() {
  return rand() / (double) RAND_MAX;
}

    
//------------------------------------------------------GENERATING INPUT-----------------------------------------------------------------    
/**
 * Generates random input according to the Plummer model, which is more
 * realistic but perhaps not so much so according to astrophysicists
 */
void generateInput(Bodies& bodies, int nbodies, int seed) {
  double v, sq, scale;
  Point p;
  double PI = acos(-1.0);

  srand(seed);

  double rsc = (3 * PI) / 16;
  double vsc = sqrt(1.0 / rsc);

  for (int body = 0; body < nbodies; body++) {
    double r = 1.0 / sqrt(pow(nextDouble() * 0.999, -2.0 / 3.0) - 1);
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);

    Body b;
    b.mass = 1.0 / nbodies;
    for (int i = 0; i < 3; i++)
      b.pos[i] = p[i] * scale;

    do {
      p.x = nextDouble();
      p.y = nextDouble() * 0.1;
    } while (p.y > p.x * p.x * pow(1 - p.x * p.x, 3.5));
    v = p.x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      for (int i = 0; i < 3; i++)
        p[i] = nextDouble() * 2.0 - 1.0;
      sq = p.x * p.x + p.y * p.y + p.z * p.z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    for (int i = 0; i < 3; i++)
      b.vel[i] = p[i] * scale;

    bodies.push_back(b);
  }
}

//---------------------------------------------------------------TIMING THE CODE--------------------------------------------------------    
// since nvidia profiling is not good for sequential code analysis    
struct timeval tm0, tm1, tm2;

double gettime(struct timeval t1, struct timeval t2){
	return (t2.tv_sec-t1.tv_sec)*1000.0+(t2.tv_usec-t1.tv_usec)/1000.0;
}
//--------------------------------------------------------------------------------------------------------------------------------------
    
//----------------------------------------------------------RUNNING SIMULATION---------------------------------------------------------    
    
    
void run(int nbodies, int ntimesteps, int seed) {
  static Bodies bodies;
  generateInput(bodies, nbodies, seed);

  for (int step = 0; step < ntimesteps; step++) {
    // Do tree building sequentially

    size_t i, k, n=bodies.size(), block=64;

    gettimeofday(&tm1, NULL);
    BoundingBox box;
    ReduceBoxes reduceBoxes(box);
	for(Bodies::iterator it = bodies.begin(); it != bodies.end(); ++it){
        (ReduceBoxes(box))(&*it,bodies);
	}

	OctreeInternal* top = new OctreeInternal(box.center());
	// Double loop for iteration of a recursion
	for( k = 0; k < block; k++){
	    for(i=n*k/block; i<n*(k+1)/block; i++){
		BuildOctree(top, box.radius())(&bodies[i],bodies);
	    }
	}

    ComputeCenterOfMass computeCenterOfMass(top);

    computeCenterOfMass();

    for(k=0;k<block;k++){
#pragma tls forkpoint id 1 maybe
        for(i=n*k/block; i<n*(k+1)/block; i++){
   	    ComputeForces(top, box.diameter())(&bodies[i],bodies);
	}
#pragma tls joinpoint id 1
    }
#pragma tls barrierpoint id 1

	for(k=0;k<block;k++){
	    for(i=n*k/block; i<n*(k+1)/block; i++){
	        AdvanceBodies()(&bodies[i],bodies);
	    }
	}

    gettimeofday(&tm2, NULL);
    printf("%lf ms\n", gettime(tm1, tm2));

    std::cout 
      << "Timestep " << step
      << " Center of Mass = " << top->pos << "\n";
    delete top;
  }
}

} // end namespace

//-----------------------------------------------------------------MAIN-----------------------------------------------------------------

int main(int argc, char** argv) {
  std::cout.setf(std::ios::right|std::ios::scientific|std::ios::showpoint);

  std::cerr << "configuration: "
            << nbodies << " bodies, "
            << ntimesteps << " time steps" << std::endl << std::endl;

  run(nbodies, ntimesteps, seed);
}