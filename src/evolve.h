#ifndef EVOLVE_H
#define EVOLVE_H

#include <mpi.h>
#include <iostream>
#include "util.h"
#include "data.h"
#include "grid.h"
#include "eos.h"
#include "reconst.h"
#include <time.h>
#include "reconst.h"
#include "advance.h"
#include "u_derivative.h"
#include <fstream>
#include <sstream>
#include <iomanip>

// this is a control class for the hydrodynamic evolution
class Evolve
{
  private:
    EOS *eos; // declare EOS object
    Reconst *reconst; // declare Reconst object
    Grid *grid; // declare Grid object
    Util *util;
    Advance *advance;
    U_derivative *u_derivative;

    InitData *DATA_ptr;

    // simulation information
    int rk_order;
    int grid_nx, grid_ny, grid_neta;

    double SUM, SUM2;
    int warnings;
    int cells;
    int weirdCases;
    int facTau;
  
  public:
    Evolve(EOS *eos, InitData *DATA_in);//constructor
    ~Evolve();//destructor
    int EvolveIt(InitData *DATA, Grid ***arena, Grid ***Lneighbor, Grid ***Rneighbor, int size, int rank);
    
    int AdvanceRK(double tau, InitData *DATA, Grid ***arena, Grid ***Lneighbor, Grid ***Rneighbor, int size, int rank);
    
    int UpdateArena(double tau, Grid ***arena);
    
    void FindFreezeOutSurface(double tau, InitData *DATA, Grid ***arena, int size, int rank);
    void FindFreezeOutSurface2(double tau, InitData *DATA, Grid ***arena, int size, int rank);
    int FindFreezeOutSurface3(double tau, InitData *DATA, Grid ***arena, int size, int rank);
    int FindFreezeOutSurface_Cornelius(double tau, InitData *DATA, Grid ***arena, int size, int rank);
    int FindFreezeOutSurface_boostinvariant_Cornelius(double tau, InitData *DATA, Grid ***arena, int size, int rank);

    void initial_prev_variables(Grid ***arena);
    void storePreviousEpsilon(Grid ***arena);
    void storePreviousW(Grid ***arena);
    void storePreviousT(Grid ***arena);

    void regulate_qmu(double* u, double* q, double* q_regulated);
    void regulate_Wmunu(double* u, double** Wmunu, double** Wmunu_regulated);

  ////////////////////////////////////////////////////////////////////////////
  // all the following variables and functions seems to be outdated
  // all the corresponding functions can be found in class advance
  // so they are comment out here
  /////////////////////////////////////////////////////////////////////////////

  //typedef struct bdry_cells
  //{
  //  Grid *grid_p_h_L;
  //  Grid *grid_p_h_R;
  //  Grid *grid_m_h_L;
  //  Grid *grid_m_h_R;
  //  
  //  double **qiphL;
  //  double **qiphR;
  //  double **qimhL;
  //  double **qimhR;
  //  
  //} BdryCells;
  //typedef struct nbrs
  //{
  //  double **qip1;
  //  double **qip2;
  //  double **qim1;
  //  double **qim2;
  //} NbrQs;

  //int Advance(double tau_init, InitData *DATA, Grid *grid_pt, Grid *Lneighbor, 
  //   Grid *Rneighbor, Grid *Lneighbor2, Grid *Rneighbor2, int rk_flag, int size, int rank);
  //
  //void MakeTauFac(double tau, int alpha, double temp_fac[]);
  //void MakeMaxSpeedAs(double tau, BdryCells *HalfwayCells, 
  //   double aiph[], double aimh[], int rk_flag);
  //
  //double MaxSpeed (double tau, int direc, Grid *grid_p, int rk_flag);
  //void InitTempGrids(BdryCells *HalfwayCells, int rk_order);
  //void InitNbrQs(NbrQs *NbrCells);
  //int MakeQIHalfs(double *qi, NbrQs *NbrCells, BdryCells *HalfwayCells, 
  //   Grid *grid_pt, InitData *DATA);
  //
  //void GetQIs(double tau, Grid *grid_pt, Grid *Lneighbor, Grid *Rneighbor, 
  //   Grid *Lneighbor2, Grid *Rneighbor2, double *qi, NbrQs *NbrCells, 
  //   int rk_flag, InitData *DATA, int size, int rank);
  //
  //void MakeRs(double *qi, double qip1[][4], double qip2[][4],
  //          double qim1[][4], double qim2[][4], 
  //          double ri[][4], double rip1[][4], double rim1[][4]);
  //
  //int ConstHalfwayCells(double tau, BdryCells *HalfwayCells, double *qi, Grid *grid_pt,
  //   InitData *DATA, int rk_flag, int size, int rank);
  //
  //void MakeDeltaQI(double tau, Grid *grid_pt, Grid *Lneighbor, Grid *Rneighbor, 
  //   Grid *Lneighbor2, Grid *Rneighbor2, 
  //   double *qi, double *rhs, InitData *DATA, int rk_flag, int size, int rank);
  //  
  //void MakeKTCurrents(double tau, double **DFmmp, Grid *grid_pt, 
  //   BdryCells *HalfwayCells, int rk_flag);
  //
  //void AdvanceQI(double tau, double **qirk, double **Fiph, double **Fimh, double *qi,
  //   Grid *grid_pt, InitData *DATA, int rk_flag);
  //
  //double minmod_dx(double up1, double u, double um1, InitData *DATA);
  //
  //int ConstNewTJb(double tau, double *qi, double **qirk,
  //   Grid *grid_pt, InitData *DATA, int rk_flag, Grid *grid_rk);
  //
  //void UpdateTJbRK(Grid *grid_rk, Grid *grid_pt, InitData *DATA, int rk_flag);
  //
  //void MPISendReceive(InitData *DATA, Grid ***arena, Grid ***Lneighbor, 
  //   Grid ***Rneighbor, int size, int rank, int rk_flag);
};
#endif
  