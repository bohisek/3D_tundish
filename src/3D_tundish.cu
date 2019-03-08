/*
 ============================================================================
 Name        : 3D_tundish.cu
 Author      : Jan Bohacek
 Version     :
 Copyright   : 
 Description : laminar flow in three-dimensional tundish in continuous casting
 ============================================================================
 */


#include <iostream>
//#include <stdio.h>
//#include <algorithm>
//#include <numeric>
#include <fstream>
#include <sstream>
#include <cstring>
//#include <ctime>
#include <math.h>

using namespace std;

typedef double T;	// precision of calculation

typedef struct {
	int Nx; 		// x-coordinate
	int Ny;			// y
	int Nz;			// z
	T dx; 			// dx = dy = dz
} Dimensions;		// dimensions of geometry

typedef struct {
	int steps;		// number of timesteps (-)
	int maxIterSIMPLE; // maximum number of SIMPLE iterations
	T CFL;			// Courant number
	T dt;			// timestep size
	T UZ;			// inlet velocity
	T ac;			// volume of cell divided by timestep
	T blocks;		// for dot product
	T blockSize;	// -||-
	T maxResU;		// stopping criterion for velocity calculation
	T maxResP;		// 					      pressure
	T maxResSIMPLE; //						  SIMPLE 
	T urfU;			// under-relaxation factor U
	T urfP;			// 						   P
} Parameters;		// simulation settings

typedef struct {
	T nu; 			// kinematic viscosity (m2/s)
	T rho;			// density
	T cp;			// specific heat
	T k;			// thermal conductivity
	T alpha;    	// thermal diffusivity (m2/s)
	T beta; 		// thermal expansion coefficient
} MaterialProperties;

// declare CPU fields
Dimensions dims;
Parameters params;
MaterialProperties liquid;

// cache constant CUDA fields
__constant__ Dimensions d_dims;
__constant__ Parameters d_params;
__constant__ MaterialProperties d_liquid;

#include "cpuFunctions.h"
#include "cudaFunctions.h"



int main()
{
	
		
	cout << "--flow in 3D tundish---" << endl;
		
	// geometry
	dims.Nx = 256;
	dims.Ny = 64;
	dims.Nz = 64;
	dims.dx = 0.001;
	
	// paramaters
	params.steps     = 1000;
	params.CFL       = 1.0;
	params.UZ        = -0.5;
	params.dt        = params.CFL * dims.dx / fabs(params.UZ);
	params.ac        = dims.dx*dims.dx/params.dt;
	params.blocks    = 256;  
	params.blockSize = 128;  
	params.maxResU      = 1e-5;
	params.maxResP      = 1e-5;
	params.maxResSIMPLE = 1e-4;
	params.maxIterSIMPLE = 20;
	params.urfU         = 0.7;
	params.urfP         = 0.3;
	//params.maxIterSIMPLE=100;
	
	// material properties
	liquid.nu  = 0.000001;   // water 1e-6 m2/s
	liquid.rho = 1000;
	
	cout << "For Courant number of " << params.CFL << " the timestep size is " << params.dt << endl;
	
	// CPU fields
	T *ux;		// ux-component of velocity
	T *uy;		// uy
	T *uz;		// uy
	T *p;		// pressure
	T *m;		// mass balance
	T *hrh,*hsg;	// dot products
	T rhNew, rhOld, sg, ap, bt;
	T endIter, rhNewSIMPLE;
	int iter, iterSIMPLE;
		
	// GPU fields
	T *dux , *duy , *duz;		// velocity components
	T *duxo, *duyo, *duzo;		// old values
	T *dp, *dpo;				// pressure and old value
	T *dm;      				// mass balance 
	T *duxtemp, *duytemp, *duztemp;								// pointers for swapping fields
	T *duxc,*duxf,*duxs,*duxw,*dkuxc,*dkuxf,*dkuxs,*dkuxw; 		// Aux
	T *drx,*dqx,*dzx,*dpx;										// Aux
	T *duyc,*duyf,*duys,*duyw, *dkuyc,*dkuyf,*dkuys,*dkuyw; 	// Auy
	T *dry,*dqy,*dzy,*dpy;										// Auy
	T *duzc,*duzf,*duzs,*duzw, *dkuzc,*dkuzf,*dkuzs,*dkuzw; 	// Auz
	T *drz,*dqz,*dzz,*dpz;										// Auz
	T *dpc,*dpf,*dps,*dpw,*dkpc,*dkpf,*dkps,*dkpw; 				// Ap
	T *drp,*dqp,*dzp,*dpp;										// Ap
	T *drh,*dsg;												// dot products
	
	
	// GPU parameters
	int THREADS_PER_BLOCK = 1024;
	int BLOCKS = ((dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2)+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;	// larger in order to have BLOCKS*THREADS_PER_BLOCK > Nx*Ny*Nz
	// taken from CUDA by example
			
	// initialize fields 
	cpuInit(ux, uy, uz, p, m, hrh, hsg);
	cudaInit(dux, duy, duz, dp, dm, duxo, duyo, duzo, dpo,
			 duxc, duxf, duxs, duxw, dkuxc, dkuxf, dkuxs, dkuxw, 	// Aux
			 drx, dqx, dzx, dpx,									// Aux
			 duyc, duyf, duys, duyw, dkuyc, dkuyf, dkuys, dkuyw, 	// Auy
			 dry, dqy, dzy, dpy,									// Auy
			 duzc, duzf, duzs, duzw, dkuzc, dkuzf, dkuzs, dkuzw, 	// Auz
			 drz, dqz, dzz, dpz,									// Auz
			 dpc, dpf, dps, dpw, dkpc, dkpf, dkps, dkpw, 			// Ap
			 drp, dqp, dzp, dpp,									// Ap
			 drh, dsg);
	
	// patch anything to dux
	//patchDux<<<BLOCKS,THREADS_PER_BLOCK>>>(dux);
	
	// patch anything to duy
	//patchDuy<<<BLOCKS,THREADS_PER_BLOCK>>>(duy);
	
	// patch anything to duz
	//patchDuz<<<BLOCKS,THREADS_PER_BLOCK>>>(duz);
	
	/*// copy back to host and save
	cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(uz, duz, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(p,   dp, sizeof(T)*dims.Nx    * dims.Ny   *(dims.Nz+2), cudaMemcpyDeviceToHost);
	saveDataInTime(ux, uy, uz, p, m, (T)0, "testTundish");*/
	
	// Aux (x-component of velocity)
	Aux<<<BLOCKS,THREADS_PER_BLOCK>>>(duxc, duxf, duxs, duxw);
	// AuxInlet not necessary, velocity inlet condition ux=0 is the same as no slip condition at wall
	AuxOutlet<<<1,100>>>(duxc,200, 15);
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkuxc,dkuxf,dkuxs,dkuxw,duxc,duxf,duxs,duxw,dims.Nx-1,dims.Ny,dims.Nz);
	
	// Auy (y-component of velocity)
	Auy<<<BLOCKS,THREADS_PER_BLOCK>>>(duyc, duyf, duys, duyw);
	//AuyInlet not necessary
	AuyOutlet<<<1,100>>>(duyc,200,15);
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkuyc,dkuyf,dkuys,dkuyw,duyc,duyf,duys,duyw,dims.Nx,dims.Ny-1,dims.Nz);
	
	// Auz (z-component of velocity)
	Auz<<<BLOCKS,THREADS_PER_BLOCK>>>(duzc, duzf, duzs, duzw);
	//AuzInlet not necessary
	AuzOutlet<<<1,100>>>(duzc,200,15);
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkuzc,dkuzf,dkuzs,dkuzw,duzc,duzf,duzs,duzw,dims.Nx,dims.Ny,dims.Nz-1);
		
	// Ap (pressure)
	Ap<<<BLOCKS,THREADS_PER_BLOCK>>>(dpc, dpf, dps, dpw);
	ApOutlet<<<1,100>>>(dpc,200,15); // Dirichlet, p=0
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkpc,dkpf,dkps,dkpw,dpc,dpf,dps,dpw,dims.Nx,dims.Ny,dims.Nz);
	
	
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	
	for (int miter=0; miter<params.steps; miter++) {
	
		// boundary conditions
		bcVelWallNoslip<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duy, duz);	// no slip at walls
		bcVelInlet<<<1,100>>>(duz, params.UZ, 50, 45);					// bcVelInlet<<<1,inletWidth>>>(dux, duy, duz, velocity, first index in x, first index in y);
		bcVelOutlet<<<1,100>>>(dux, duy, duz, 200, 15);					// bcVelOutlet<<<1,outletwidth>>>(dux, duy, duz, first index in x, first index in y);
		
		//swap old and new arrays for next timestep
		duxtemp = duxo; duxo = dux; dux = duxtemp;
		duytemp = duyo; duyo = duy; duy = duytemp;
		duztemp = duzo; duzo = duz; duz = duztemp;
		
		// advect horizontal and vertical velocity components
		advectUx<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duxo, duyo, duzo);
		advectUy<<<BLOCKS,THREADS_PER_BLOCK>>>(duy, duxo, duyo, duzo);
		advectUz<<<BLOCKS,THREADS_PER_BLOCK>>>(duz, duxo, duyo, duzo);
		
		bcVelWallNoslip<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duy, duz);	// no slip at walls
		bcVelInlet<<<1,100>>>(duz, params.UZ, 50, 45);					// bcVelInlet<<<1,inletWidth>>>(dux, duy, duz, velocity, first index in x, first index in y);
		bcVelOutlet<<<1,100>>>(dux, duy, duz, 200, 15);					// bcVelOutlet<<<1,outletwidth>>>(dux, duy, duz, first index in x, first index in y);
		
		// ************ BEGIN SIMPLE **********
		iterSIMPLE    = 0;
		rhNewSIMPLE   = 1;
		
		/*// copy back to host and save
		cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(uz, duz, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(p,   dp, sizeof(T)*dims.Nx    * dims.Ny   *(dims.Nz+2), cudaMemcpyDeviceToHost);
		saveDataInTime(ux, uy, uz, p, m, (T)0, "testTundish");*/
		
		
		while  (rhNewSIMPLE > params.maxResSIMPLE) {  //(iterSIMPLE < params.maxIterSIMPLE) {
		
			iterSIMPLE++;
		
			// ********** BEGIN solve UX **********
			duToDr<<<BLOCKS,THREADS_PER_BLOCK>>>(drx, dux,dims.Nx-1,dims.Ny,dims.Nz);						// drx := dux
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqx,duxc,duxf,duxs,duxw,drx,dims.Nx-1,dims.Ny,dims.Nz);		// q := Aux ux 
			b<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dims.Nx-1,dims.Ny,dims.Nz);									// drx := bx
			// bxInlet not necessary as ux=0 there
			bpx<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dpo,dims.Nx-1,dims.Ny,dims.Nz);	    					// add grad(p) to rhs of Ax=b
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dqx,(T)-1.,(T)1.,dims.Nx-1,dims.Ny, dims.Nz); 	  		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzx,dkuxc,dkuxf,dkuxs,dkuxw,drx,dims.Nx-1,dims.Ny,dims.Nz);	// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh,drx,dzx,dims.Nx-1,dims.Ny,dims.Nz);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "Ux residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResU * params.maxResU;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpx, dzx, sizeof(T)*(dims.Nx-1)*dims.Ny*(dims.Nz+2),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpx,dzx,(T)1.,bt,dims.Nx-1,dims.Ny,dims.Nz);   			// p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqx,duxc,duxf,duxs,duxw,dpx,dims.Nx-1,dims.Ny,dims.Nz);		// q := Aux p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg, dpx, dqx, dims.Nx-1, dims.Ny,dims.Nz);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dqx,-ap,(T)1.,dims.Nx-1,dims.Ny,dims.Nz); 				// r = r - alpha*q
				AXPY2<<<BLOCKS,THREADS_PER_BLOCK>>>(dux,dpx, ap,(T)1.,dims.Nx-1,dims.Ny,dims.Nz);  				// x = x + alpha*p; Note: sizeof(dux) != sizeof(dpx)
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzx,dkuxc,dkuxf,dkuxs,dkuxw,drx,dims.Nx-1,dims.Ny,dims.Nz);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drx, dzx, dims.Nx-1, dims.Ny,dims.Nz);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			//cout << "Ux iter number: " << iter << endl;
			// ********** END solve UX ************
			
			// ********** BEGIN solve UY **********
			duToDr<<<BLOCKS,THREADS_PER_BLOCK>>>(dry, duy,dims.Nx,dims.Ny-1,dims.Nz);					// dry := duy
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqy,duyc,duyf,duys,duyw,dry,dims.Nx,dims.Ny-1,dims.Nz);	// q := Auy uy 
			b<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dims.Nx,dims.Ny-1,dims.Nz);	
			//byOutlet not necessary due to zero gradient condition
			bpy<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dpo,dims.Nx,dims.Ny-1,dims.Nz);						// add grad(p) to rhs of Ax=b
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dqy,(T)-1.,(T)1.,dims.Nx,dims.Ny-1,dims.Nz);   		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzy,dkuyc,dkuyf,dkuys,dkuyw,dry,dims.Nx,dims.Ny-1,dims.Nz);	// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, dry, dzy, dims.Nx, dims.Ny-1,dims.Nz);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "Uy residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResU * params.maxResU;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpy, dzy, sizeof(T)*dims.Nx*(dims.Ny-1)*(dims.Nz+2),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpy,dzy,(T)1.,bt,dims.Nx,dims.Ny-1,dims.Nz);   		    // p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqy,duyc,duyf,duys,duyw,dpy,dims.Nx,dims.Ny-1,dims.Nz);		// q := Auy p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg,dpy,dqy,dims.Nx,dims.Ny-1,dims.Nz);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dqy,-ap,(T)1.,dims.Nx,dims.Ny-1,dims.Nz);  			    // r = r - alpha*q
				AXPY2<<<BLOCKS,THREADS_PER_BLOCK>>>(duy,dpy, ap,(T)1.,dims.Nx,dims.Ny-1,dims.Nz);  			    // x = x + alpha*p; Note: sizeof(duy) != sizeof(dpy)
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzy,dkuyc,dkuyf,dkuys,dkuyw,dry,dims.Nx,dims.Ny-1,dims.Nz);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, dry, dzy, dims.Nx, dims.Ny-1,dims.Nz);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			//cout << "Uy iter number: " << iter << endl;
			// ********** END solve UY ************
			
			
			// ********** BEGIN solve UZ **********
			duToDr<<<BLOCKS,THREADS_PER_BLOCK>>>(drz, duz,dims.Nx,dims.Ny,dims.Nz-1);					// dry := duy
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqz,duzc,duzf,duzs,duzw,drz,dims.Nx,dims.Ny,dims.Nz-1);	// q := Auy uy 
			b<<<BLOCKS,THREADS_PER_BLOCK>>>(drz,dims.Nx,dims.Ny,dims.Nz-1);
			bzInlet<<<1,100>>>(drz, duz, 50, 45);
			//bzOutlet not necessary due to zero gradient condition
			bpz<<<BLOCKS,THREADS_PER_BLOCK>>>(drz,dpo,dims.Nx,dims.Ny,dims.Nz-1);						// add grad(p) to rhs of Ax=b
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drz,dqz,(T)-1.,(T)1.,dims.Nx,dims.Ny,dims.Nz-1);   		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzz,dkuzc,dkuzf,dkuzs,dkuzw,drz,dims.Nx,dims.Ny,dims.Nz-1);	// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drz, dzz, dims.Nx, dims.Ny,dims.Nz-1);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "Uz residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResU * params.maxResU;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpz, dzz, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+1),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpz,dzz,(T)1.,bt,dims.Nx,dims.Ny,dims.Nz-1);   		    // p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqz,duzc,duzf,duzs,duzw,dpz,dims.Nx,dims.Ny,dims.Nz-1);		// q := Auz p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg,dpz,dqz,dims.Nx,dims.Ny,dims.Nz-1);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drz,dqz,-ap,(T)1.,dims.Nx,dims.Ny,dims.Nz-1);  			    // r = r - alpha*q
				AXPY2<<<BLOCKS,THREADS_PER_BLOCK>>>(duz,dpz, ap,(T)1.,dims.Nx,dims.Ny,dims.Nz-1);  			    // x = x + alpha*p; Note: sizeof(duz) != sizeof(dpz)
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzz,dkuzc,dkuzf,dkuzs,dkuzw,drz,dims.Nx,dims.Ny,dims.Nz-1);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drz, dzz, dims.Nx, dims.Ny,dims.Nz-1);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);


			}
			//cout << "Uz iter number: " << iter << endl;
			// ********** END solve UZ ************
			
				
			// update velocity at boundary
			bcVelOutlet<<<1,100>>>(dux, duy, duz, 200, 15);
			
			
			
			// ********** BEGIN solve P ***********
			// The finite volume method in computational fluid dynamics, F. Moukalled, L. Mangani, M. Darwish
			// Patankar's SIMPLE
			cudaMemcpy(drp, dp, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+2),cudaMemcpyDeviceToDevice);
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqp,dpc,dpf,dps,dpw,drp,dims.Nx,dims.Ny,dims.Nz);			// q := Ap p 
			bp<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dux,duy,duz,dims.Nx,dims.Ny,dims.Nz);						// should become at convergence == zero correction field
						
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh,drp,drp,dims.Nx,dims.Ny,dims.Nz);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNewSIMPLE = dot(hrh,params.blocks);
						
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dqp,(T)-1.,(T)1.,dims.Nx,dims.Ny,dims.Nz);   			// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzp,dkpc,dkpf,dkps,dkpw,drp,dims.Nx,dims.Ny,dims.Nz);		// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh,drp,dzp,dims.Nx,dims.Ny,dims.Nz);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "P residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResP * params.maxResP;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpp, dzp, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+2),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpp,dzp,(T)1.,bt,dims.Nx,dims.Ny,dims.Nz);   		// p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqp,dpc,dpf,dps,dpw,dpp,dims.Nx,dims.Ny,dims.Nz);		// q := Ap p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg,dpp,dqp,dims.Nx,dims.Ny,dims.Nz);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dqp,-ap,(T)1.,dims.Nx,dims.Ny,dims.Nz);  			// r = r - alpha*q
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dp ,dpp, ap,(T)1.,dims.Nx,dims.Ny,dims.Nz);  			// x = x + alpha*p
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzp,dkpc,dkpf,dkps,dkpw,drp,dims.Nx,dims.Ny,dims.Nz);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh,drp,dzp,dims.Nx,dims.Ny,dims.Nz);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			cout << "P iter number: " << iter << endl;
			// ********** END solve P ************
			
			
			// ***** BEGIN correct P, UX, UY fields ******
			correctUX<<<BLOCKS,THREADS_PER_BLOCK>>>(dux,dp,dims.Nx-1,dims.Ny,dims.Nz); 				// ux = -dt/rho*dp/dx
			correctUY<<<BLOCKS,THREADS_PER_BLOCK>>>(duy,dp,dims.Nx,dims.Ny-1,dims.Nz);				// uy = -dt/rho*dp/dy
			correctUZ<<<BLOCKS,THREADS_PER_BLOCK>>>(duz,dp,dims.Nx,dims.Ny,dims.Nz-1);				// uz = -dt/rho*dp/dz
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dp,dpo,(T)1.,params.urfP,dims.Nx,dims.Ny,dims.Nz);	// p = urfP*p + pold
			cudaMemcpy(dpo, dp, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+2),cudaMemcpyDeviceToDevice);	// pold = p
			cudaMemset(dp ,  0, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+2));
			
			bcVelOutlet<<<1,100>>>(dux, duy, duz, 200, 15);
			// ****** END correct P, UX, UY fields *******
			
			
			
			// ***** BEGIN check mass conservation *****
			bp<<<BLOCKS,THREADS_PER_BLOCK>>>(dm,dux,duy,duz,dims.Nx,dims.Ny,dims.Nz);
			cudaMemcpy(m, dm, sizeof(T)*dims.Nx*dims.Ny*(dims.Nz+2), cudaMemcpyDeviceToHost);
			// ****** END check mass conservation *******/
			
			
			/*// copy back to host and save
			cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
			cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
			cudaMemcpy(uz, duz, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
			cudaMemcpy(p,  dpo, sizeof(T)*dims.Nx    * dims.Ny   *(dims.Nz+2), cudaMemcpyDeviceToHost);
			saveDataInTime(ux, uy, uz, p, m, (T)iterSIMPLE, "testTundish");*/
			
			
			
		
		}
		// ************** END SIMPLE *****************
		
		cudaMemcpy(duxo, dux,  sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToDevice);
		cudaMemcpy(duyo, duy,  sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToDevice);
		cudaMemcpy(duzo, duz,  sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToDevice);
		
		
		// copy back to host and save
		cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(uz, duz, sizeof(T)*(dims.Nx+2)*(dims.Ny+2)*(dims.Nz+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(p,  dpo, sizeof(T)*dims.Nx    * dims.Ny   *(dims.Nz+2), cudaMemcpyDeviceToHost);
		saveDataInTime(ux, uy, uz, p, m, (T)miter, "testTundish");
		
		
		cout << "SIMPLE iter number: " << iterSIMPLE << endl;
	}
	cout << "simulation finished." << endl;
	
	
	
	
	/*// copy back to host and save
	cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(p , dpo, sizeof(T)*dims.Nx    *(dims.Ny+2), cudaMemcpyDeviceToHost);
	saveDataInTime(ux, uy, p, m, (T)1, "testTundish1e-4");*/
	
	
	
	
	
	
	
	
	
	
		
	
	
	
	
	/*cudaMemcpy(p, drp, sizeof(T)*(dims.Nx)*(dims.Ny+2), cudaMemcpyDeviceToHost);
				ofstream File;
				File.open("ckeck");
				for (int j=0;j<(dims.Ny+2);j++) {
					for (int i=0;i<(dims.Nx);i++) {
						File << p[i+j*(dims.Nx)] << ",";
					}
					File << endl;
				}
				File.close();*/
	
	
	
	
	
	
		
	
	
	
	
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cout<< "ellapsed time (cuda): " << elapsedTime	<< " miliseconds" << endl;
	
	
	
	
			
	cpuFinalize(ux, uy, uz, p, m, hrh, hsg);
	cudaFinalize(dux, duy, duz, dp, dm, duxo, duyo, duzo, dpo,
				 duxc, duxf, duxs, duxw, dkuxc, dkuxf, dkuxs, dkuxw, 	// Aux
				 drx, dqx, dzx, dpx,									// Aux
				 duyc, duyf, duys, duyw, dkuyc, dkuyf, dkuys, dkuyw, 	// Auy
				 dry, dqy, dzy, dpy,									// Auy
				 duzc, duzf, duzs, duzw, dkuzc, dkuzf, dkuzs, dkuzw, 	// Auz
				 drz, dqz, dzz, dpz,									// Auz
				 dpc, dpf, dps, dpw, dkpc, dkpf, dkps, dkpw, 			// Ap
				 drp, dqp, dzp, dpp,									// Ap
				 drh, dsg);
	
	
	return 0;
}
