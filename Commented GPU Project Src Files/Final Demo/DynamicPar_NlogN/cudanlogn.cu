#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <omp.h>
#include "cudaBhtree.cu"
#include "Constants.h"
#include <sys/time.h>

/***************************************DECLARATIONS*******************************************/

void initializeBodies(struct body* bods);
void runSimulation(struct body* b, struct body* db);
__global__ void interactBodies(struct body* b);
__device__ void singleInteraction(struct body* a, struct body* b);
__host__ __device__ double magnitude(vec3 v);
__device__ void updateBodies(struct body* b);
double clamp(double x);
__global__ void recinsert(Bhtree *tree, body* insertBod);
__device__ inline bool contains(Octant *root, vec3 p);
__global__ void interactInTree(Bhtree *tree, struct body* b);
__host__ __device__ double magnitude(float x, float y, float z);

/***************************************MAIN*****************************************************/
int main()
{

    std::cout << "Nbodies: " <<NUM_BODIES << "\n";
	struct timeval stop,start;
    gettimeofday(&start,NULL);
    struct body *bodies = new struct body[NUM_BODIES];
	struct body *d_bodies;
	cudaMalloc(&d_bodies,NUM_BODIES*sizeof(struct body));
	
	initializeBodies(bodies);
	cudaMemcpy(d_bodies,bodies,NUM_BODIES*sizeof(struct body),cudaMemcpyHostToDevice);
	runSimulation(bodies, d_bodies);
	std::cout << "\nwe made it\n";
	delete[] bodies;
    gettimeofday(&stop,NULL);
    printf("took %lu sec\n",stop.tv_sec-start.tv_sec);
	return 0;
}

/***************************************** INITIALIZING BODIES *******************************************/

void initializeBodies(struct body* bods)
{
	using std::uniform_real_distribution;
	uniform_real_distribution<double> randAngle (0.0, 200.0*PI);
	uniform_real_distribution<double> randRadius (INNER_BOUND, SYSTEM_SIZE);
	uniform_real_distribution<double> randHeight (0.0, SYSTEM_THICKNESS);
	std::default_random_engine gen (0);
	double angle;
	double radius;
	double velocity;
	struct body *current;

	//STARS
	velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
	//STAR 1
	current = &bods[0];
	current->position.x = 0.0;///-BINARY_SEPARATION;
	current->position.y = 0.0;
	current->position.z = 0.0;
	current->velocity.x = 0.0;
	current->velocity.y = 0.0;//velocity;
	current->velocity.z = 0.0;
	current->mass = SOLAR_MASS;

	    ///STARTS AT NUMBER OF STARS///
	double totalExtraMass = 0.0;
	for (int index=1; index<NUM_BODIES; index++)
	{
		angle = randAngle(gen);
		radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
		velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
					  	  	  	  	  / (radius*TO_METERS)), 0.5);
		current = &bods[index];
		current->position.x =  radius*cos(angle);
		current->position.y =  radius*sin(angle);
		current->position.z =  randHeight(gen)-SYSTEM_THICKNESS/2;
		current->velocity.x =  velocity*sin(angle);
		current->velocity.y = -velocity*cos(angle);
		current->velocity.z =  0.0;
		current->mass = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
		totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
	}
	//std::cout << "\nTotal Disk Mass: " << totalExtraMass;
	std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
			  << "\n______________________________\n";
}

/**********************************************************RUN SIMULATION************************************/

void runSimulation(struct body* b, struct body* db)
{
	for (int step=1; step<STEP_COUNT; step++)
	{
		std::cout << "\nBeginning timestep: " << step;
		interactBodies<<<1,1>>>(db);
		cudaError_t error = cudaGetLastError();
                        if(error!=cudaSuccess)
                                printf("\nCUDA error:%s",cudaGetErrorString(error));
		cudaMemcpy(b,db,NUM_BODIES*sizeof(struct body),cudaMemcpyDeviceToHost);
		if (DEBUG_INFO) {std::cout << "\n-------Done------- timestep: "
			       << step << "\n" << std::flush;}
	}
}

__device__ Bhtree *Gtree;

/******************************************************** BOUNDING BOX ****************************************************/

__global__ void recinsert(Bhtree *tree, body* insertBod)
{
    // USING STREAMS TO OPTIMIZE THE TOTAL TIME
	const int num_streams=8;
	cudaStream_t streams[num_streams];

	for(int i=0; i<num_streams; i++)
		cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
		
	if (tree->myBod==NULL)
	{
		tree->myBod = insertBod;
	} else //if (!isExternal())
	{
		bool isExtern = tree->UNW==NULL && tree->UNE==NULL && tree->USW==NULL && tree->USE==NULL;
		isExtern = isExtern && tree->DNW==NULL && tree->DNE==NULL && tree->DSW==NULL && tree->DSE==NULL;
		body *updatedBod;
		if (!isExtern)
		{
			updatedBod = new struct body;
			updatedBod->position.x = (insertBod->position.x*insertBod->mass +
							       tree->myBod->position.x*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->position.y = (insertBod->position.y*insertBod->mass +
								   tree->myBod->position.y*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->position.z = (insertBod->position.z*insertBod->mass +
								   tree->myBod->position.z*tree->myBod->mass) /
							  (insertBod->mass+tree->myBod->mass);
			updatedBod->mass = insertBod->mass+tree->myBod->mass;
		//	delete myBod;
			if (tree->toDelete!=NULL) delete tree->toDelete;
			tree->toDelete = updatedBod;
			tree->myBod = updatedBod;
			updatedBod = insertBod;
		} else {
			updatedBod = tree->myBod;
		}
		Octant *unw = tree->octy->mUNW();
		if (contains(unw,updatedBod->position))
		{
			if (tree->UNW==NULL) { tree->UNW = new Bhtree(unw); }
			else { delete unw; }
			recinsert<<<1,1,0,streams[0]>>>(tree->UNW,updatedBod);
		} else {
			delete unw;
			Octant *une = tree->octy->mUNE();
			if (contains(une,updatedBod->position))
			{
				if (tree->UNE==NULL) { tree->UNE = new Bhtree(une); }
				else { delete une; }
				recinsert<<<1,1,0,streams[1]>>>(tree->UNE,updatedBod);
			} else {
				delete une;
				Octant *usw = tree->octy->mUSW();
				if (contains(usw,updatedBod->position))
				{
					if (tree->USW==NULL) { tree->USW = new Bhtree(usw); }
					else { delete usw; }
					recinsert<<<1,1,0,streams[2]>>>(tree->USW,updatedBod);
				} else {
					delete usw;
					Octant *use = tree->octy->mUSE();
					if (contains(use,updatedBod->position))
					{
						if (tree->USE==NULL) { tree->USE = new Bhtree(use); }
						else { delete use; }
						recinsert<<<1,1,0,streams[3]>>>(tree->USE,updatedBod);
					} else {
						delete use;
						Octant *dnw = tree->octy->mDNW();
						if (contains(dnw,updatedBod->position))
						{
							if (tree->DNW==NULL) { tree->DNW = new Bhtree(dnw); }
							else { delete dnw; }
							recinsert<<<1,1,0,streams[4]>>>(tree->DNW,updatedBod);
						} else {
							delete dnw;
							Octant *dne = tree->octy->mDNE();
							if (contains(dne,updatedBod->position))
							{
								if (tree->DNE==NULL) { tree->DNE = new Bhtree(dne); }
								else { delete dne; }
								recinsert<<<1,1,0,streams[5]>>>(tree->DNE,updatedBod);
							} else {
								delete dne;
								Octant *dsw = tree->octy->mDSW();
								if (contains(dsw,updatedBod->position))
								{
									if (tree->DSW==NULL) { tree->DSW = new Bhtree(dsw); }
									else { delete dsw; }
									recinsert<<<1,1,0,streams[6]>>>(tree->DSW,updatedBod);
								} else {
									delete dsw;
									Octant *dse = tree->octy->mDSE();
									if (tree->DSE==NULL) { tree->DSE = new Bhtree(dse); }
									else { delete dse; }
									recinsert<<<1,1,0,streams[7]>>>(tree->DSE,updatedBod);
									}
								}
							}
						}
					}
				}
			}
	//	delete updatedBod;
		if (isExtern) {
			recinsert<<<1,1>>>(tree,insertBod);
		}
	}
}
/******************************************************************************************************************/

/********************************************************** INTERACTION BETWEEN BODIES ****************************/

__global__ void interactBodies(struct body* bods)
{
	// Sun interacts individually
	printf("\ncalculating force from star...");
	struct body *sun = &bods[0];
	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		singleInteraction(sun, &bods[bIndex]);
	}

	//if (DEBUG_INFO) {std::cout << "\nBuilding Octree..." << std::flush;}
	printf("\nBuilding octree...");
	// Build tree
	vec3 *center = new struct vec3;
	center->x = 0;
	center->y = 0;
	center->z = 0.1374; /// Does this help?
	Octant *root = new Octant(center, 60*SYSTEM_SIZE);
	Gtree = new Bhtree(root);

	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		if (contains(root,bods[bIndex].position))
		{
			recinsert<<<1,1>>>(Gtree,&bods[bIndex]);
			cudaError_t error = cudaGetLastError();
			if(error!=cudaSuccess)
				printf("\nCUDA error:%s",cudaGetErrorString(error));
		}
	}
	printf("\ncalculating interactions...");
	//if (DEBUG_INFO) {std::cout << "\nCalculating particle interactions..." << std::flush;}
	
	// loop through interactions
	//#pragma omp parallel for
	for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
	{
		if (contains(root,bods[bIndex].position))
		{
			interactInTree<<<1,1>>>(Gtree,&bods[bIndex]);
			cudaError_t error = cudaGetLastError();
                        if(error!=cudaSuccess)
                                printf("\nCUDA error:%s",cudaGetErrorString(error));
		}
	}
	
	// Destroy tree
//	delete Gtree;
	//
	printf("\nupdating particle positions...");
	//if (DEBUG_INFO) {std::cout << "\nUpdating particle positions..." << std::flush;}
	updateBodies(bods);
}


/******************************************* SINGLE INTERACTIONS IN THE TREE *****************************************/ 

__global__ void interactInTree(Bhtree *tree, struct body* b)
{
	bool isExternal = tree->UNW==NULL && tree->UNE==NULL && tree->USW==NULL && tree->USE==NULL;
	isExternal = isExternal && tree->DNW==NULL && tree->DNE==NULL && tree->DSW==NULL && tree->DSE==NULL;
	const int num_streams=8;
	cudaStream_t streams[num_streams];
	for(int i=0; i<num_streams; i++)
		cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
	Octant *o = tree->octy;
	body *myb = tree->myBod;
	if(isExternal && myb!=b)
		singleInteraction(myb,b);
	else if(o->getLength()/magnitude(myb->position.x-b->position.x,
						  myb->position.y-b->position.y,
						  myb->position.z-b->position.z) < MAX_DISTANCE)
		singleInteraction(myb,b);
	else
	{
		if (tree->UNW!=NULL) interactInTree<<<1,1,0,streams[0]>>>(tree->UNW,b);
		if (tree->UNE!=NULL) interactInTree<<<1,1,0,streams[1]>>>(tree->UNE,b);
		if (tree->USW!=NULL) interactInTree<<<1,1,0,streams[2]>>>(tree->USW,b);
		if (tree->USE!=NULL) interactInTree<<<1,1,0,streams[3]>>>(tree->USE,b);
		if (tree->DNW!=NULL) interactInTree<<<1,1,0,streams[4]>>>(tree->DNW,b);
		if (tree->DNE!=NULL) interactInTree<<<1,1,0,streams[5]>>>(tree->DNE,b);
		if (tree->DSW!=NULL) interactInTree<<<1,1,0,streams[6]>>>(tree->DSW,b);
		if (tree->DSE!=NULL) interactInTree<<<1,1,0,streams[7]>>>(tree->DSE,b);
	}
}

/************************************************************ HELPER FUNCTION *************************************************/

__device__ inline bool contains(Octant *root, vec3 p)
{
	double length = root->getLength();
	vec3* mid = root->getMid();
	return p.x<=mid->x+length/2.0 && p.x>=mid->x-length/2.0 &&
			   p.y<=mid->y+length/2.0 && p.y>=mid->y-length/2.0 &&
			   p.z<=mid->z+length/2.0 && p.z>=mid->z-length/2.0;
}

__host__ __device__ double magnitude(float x, float y, float z)
{
	return sqrt(x*x+y*y+z*z);
}

/************************************************************ INTERACTION BETWEEN 2 INDIVIDUAL BODIES *****************************/

__device__ void singleInteraction(struct body* a, struct body* b)
{
	vec3 posDiff;
	posDiff.x = (a->position.x-b->position.x)*TO_METERS;
	posDiff.y = (a->position.y-b->position.y)*TO_METERS;
	posDiff.z = (a->position.z-b->position.z)*TO_METERS;
	double dist = magnitude(posDiff);
	double F = TIME_STEP*(G*a->mass*b->mass) / ((dist*dist + SOFTENING*SOFTENING) * dist);

	a->accel.x -= F*posDiff.x/a->mass;
	a->accel.y -= F*posDiff.y/a->mass;
	a->accel.z -= F*posDiff.z/a->mass;
	b->accel.x += F*posDiff.x/b->mass;
	b->accel.y += F*posDiff.y/b->mass;
	b->accel.z += F*posDiff.z/b->mass;
}

/***************************************************** HELPER FUNCTION *****************************************************/

__host__ __device__ double magnitude(vec3 v)
{
	return sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
}

/********************************************************* UPDATE/ADVANCE BODIES************************************************/

__device__ void updateBodies(struct body* bods)
{
	double mAbove = 0.0;
	double mBelow = 0.0;
	for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
	{
		struct body *current = &bods[bIndex];
		if (DEBUG_INFO)
		{
			if (bIndex==0)
			{
			//	std::cout << "\nStar x accel: " << current->accel.x
			//			  << "  Star y accel: " << current->accel.y;
			} else if (current->position.y > 0.0)
			{
				mAbove += current->mass;
			} else {
				mBelow += current->mass;
			}
		}
		current->velocity.x += current->accel.x;
		current->velocity.y += current->accel.y;
		current->velocity.z += current->accel.z;
		current->accel.x = 0.0;
		current->accel.y = 0.0;
		current->accel.z = 0.0;
		current->position.x += TIME_STEP*current->velocity.x/TO_METERS;
		current->position.y += TIME_STEP*current->velocity.y/TO_METERS;
		current->position.z += TIME_STEP*current->velocity.z/TO_METERS;
	}
	if (DEBUG_INFO)
	{
		//std::cout << "\nMass below: " << mBelow << " Mass Above: "
		//		  << mAbove << " \nRatio: " << mBelow/mAbove;
	}
}


