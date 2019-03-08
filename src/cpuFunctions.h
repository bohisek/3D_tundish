/*
 * cpuFunctions.h
 *
 *  Created on: Jan 23, 2019
 *      Author: jbohacek
 */

#ifndef CPUFUNCTIONS_H_
#define CPUFUNCTIONS_H_


// initialize CPU fields
template <class T>
void cpuInit(T *&ux, T *&uy, T *&uz, T *&p, T *&m,
		     T *&hrh, T *&hsg)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int blocks = params.blocks;

	ux = new T[(Nx+2)*(Ny+2)*(Nz+2)];					// MINUS boundary cells and ghost cells i.e. NEITHER storing boundary cells NOR ghost cells
	uy = new T[(Nx+2)*(Ny+2)*(Nz+2)];						// -||-
	uz = new T[(Nx+2)*(Ny+2)*(Nz+2)];						// -||-
	p  = new T[Nx*Ny*Nz+2*Nx*Ny];
	m  = new T[Nx*Ny*Nz+2*Nx*Ny];
	hrh = new T[blocks];
	hsg = new T[blocks];


	memset(ux, 0, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(T));	// set to zero
	memset(uy, 0, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(T));
	memset(uz, 0, (Nx+2)*(Ny+2)*(Nz+2)*sizeof(T));
	memset(p,  0, (Nx*Ny*Nz+2*Nx*Ny)*sizeof(T));
	memset(m,  0, (Nx*Ny*Nz+2*Nx*Ny)*sizeof(T));
	memset(hrh,0, blocks*sizeof(T));
	memset(hsg,0, blocks*sizeof(T));
}


// free memory
template <class T>
void cpuFinalize(T *&ux, T *&uy, T*&uz, T *&p, T *&m,
		         T *&hrh, T *&hsg)
{
	delete[] ux;
	delete[] uy;
	delete[] uz;
	delete[] p;
	delete[] m;
	delete[] hrh;
	delete[] hsg;
}

// save data in time
template <class T>
void saveDataInTime(T *ux,
		T *uy,
		T *uz,
		T *p,
		T *m,
		const T t,
		const string name)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int second;
	T dx = dims.dx;
	ofstream File;
	stringstream fileName;
	fileName << name << "-" << fixed << (int)t << ".vtk";
	File.open(fileName.str().c_str());
	File << "# vtk DataFile Version 3.0" << endl << "vtk output" << endl;
	File << "ASCII" << endl << "DATASET STRUCTURED_POINTS" << endl;
	File << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << endl;
	File << "ORIGIN " << 0.5*dx << " " << 0.5*dx << " " << 0.5*dx << endl;
	File << "SPACING " << dx << " " << dx << " " << dx << endl;
	File << "POINT_DATA " << Nx*Ny*Nz << endl;
	File << "VECTORS " << "velocity" << " float" << endl;
	for (int i=0; i<(Nx+2)*(Ny+2)*(Nz+2); ++i) {
		int px =  i % (Nx+2);
		int py = (i / (Nx+2)) % (Ny+2);
		int pz =  i /((Nx+2)  * (Ny+2));
		if ((px<Nx) && (py<Ny) && (pz<Nz))
		{
		second = px+1+(py+1)*(Nx+2)+(pz+1)*(Nx+2)*(Ny+2);

		File << 0.5 * (ux[second-1]              +  ux[second]) << " "		// average from two x-velocities at faces
			 << 0.5 * (uy[second-(Nx+2)]         +  uy[second]) << " " 	  	//                  y-velocities
			 << 0.5 * (uz[second-(Nx+2)*(Ny+2)]  +  uz[second]) << endl;	//                  z-velocities
		}
	}
	File << "SCALARS " << "pressure" << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny*Nz; ++i) {
		File << p[i+Nx*Ny] <<endl;
	}
	File << "SCALARS " << "mass" << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny*Nz; ++i) {
		File << m[i+Nx*Ny] <<endl;
	}
	File.close();
	cout << "saving VTK (" << fileName.str() << ") at t = " << (int)t << " sec." << endl;
}

// finalize dot product on cpu
T dot(const T *x,
	  const int blocks)
{
	T y = 0;
	for (int i=0; i<blocks; i++) {
		y += x[i];
	}
	return y;
}

#endif /* CPUFUNCTIONS_H_ */
