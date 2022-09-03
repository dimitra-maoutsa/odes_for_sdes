
# Deterministic particle dynamics for simulating Fokker-Planck probability flows 


Read [here](https://gitlab.com/dimitra-maoutsa/odes_for_sdes/-/blob/master/README.md) the properly rendered readme file for now...


(under construction -  for more detailed info please read the [relevant article](https://www.mdpi.com/1099-4300/22/8/802/htm))


**Particle-based** framework for simulating **solutions of Fokker–Planck equations** that
- is **effortless** to set up
- provides **smooth transient solutions**
- is **computationally efficient**.


## A. **From SDEs to ODEs**
  - ### Systems with additive noise
     Consider a stochastic system described by the SDE 
      <p align="center">
     <img src="https://latex.codecogs.com/png.latex?%5Clarge%20dX_t%3D%20f%28X_t%29%20dt%20&plus;%20%5Csigma%20dW_t.">
     </p>
      The temporal evolution of the probability density of the system state is captured by the Fokker-Planck equation (FPE)

      <p align="center">
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cfrac%7B%5Cpartial%20p_t%28x%29%7D%7B%5Cpartial%20t%7D%20%3D%20-%5Cnabla%5Ccdot%20%5Cleft%5Bf%28x%29%20p_t%28x%29%20-%20%5Cfrac%7B%5Csigma%5E2%7D%7B2%7D%20%5Cnabla%20p_t%28x%29%5Cright%5D."> 
      </p>
      
      The FPE may be re-written in the form of a **_Liouville equation_**  
      ```diff
      ! [Eq.(3-5) in the main text]
      ```
      <p align="center">
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cfrac%7B%5Cpartial%20p_t%28x%29%7D%7B%5Cpartial%20t%7D%20%3D%20-%5Cnabla%5Ccdot%20%5Cleft%5B%7B%5Cleft%28f%28x%29%20-%20%5Cfrac%7B%5Csigma%5E2%7D%7B2%7D%20%5Cnabla%20%5Cln%20p_t%28x%29%5Cright%29%7D%5C%3B%20p_t%28x%29%20%5Cright%5D%2C"></p>
      </p>

      which in turn may be viewed as an evolution equation of the probability distribution of a statistical ensemble of **N** **_deterministic_** dynamical systems of the form _[Eq.(4-5) in the main text]_ 
  
      <p align="center"> 
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cfrac%7BdX_t%5E%7B%28i%29%7D%7D%7Bdt%7D%20%3D%20%7Bf%28X_t%5E%7B%28i%29%7D%29%29%20-%20%5Cfrac%7B%5Csigma%5E2%7D%7B2%7D%20%5Cnabla%20%5Cln%20p_t%28X_t%5E%7B%28i%29%7D%29%7D%2C%20%5Cquad%20%5Cquad%20%5Cquad%20%281%29">
      </p>
      
      with i=1,...,N.

  - ### Systems with multiplicative noise
    
      In a similar vain, for **_state dependent_** diffusion 

      <p align="center">
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20dX_t%3D%20f%28X_t%29%20dt%20&plus;%20%5Csigma%28X_t%29%20dW_t%2C">
   
      the associated deterministic particle dynamics are 
      <p align="center">
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cfrac%7BdX_t%5E%7B%28i%29%7D%7D%7Bdt%7D%20%3D%20%7Bf%28X_t%5E%7B%28i%29%7D%29%20-%20%5Cfrac%7B%5Csigma%28X_t%5E%7B%28i%29%7D%29%5Csigma%28X_t%5E%7B%28i%29%7D%29%5E%7B%5Cintercal%7D%7D%7B2%7D%20%5Cnabla%20%5Cln%20p_t%28X_t%5E%7B%28i%29%7D%29%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cnabla%20%5Ccdot%20%5Csigma%28X_t%5E%7B%28i%29%7D%29%5Csigma%28X_t%5E%7B%28i%29%7D%29%5E%7B%5Cintercal%7D%20%7D%2C">

      which, by setting <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20D%28x%29%20%3D%20%5Csigma%28x%29%20%5Csigma%28x%29%5E%7B%5Cintercal%7D%2C"> become 
      ```diff
      ! [Eq.(53) {in the main text] 
      ```


      <p align="center">
      <img src="https://latex.codecogs.com/png.latex?%5Clarge%20%5Cfrac%7BdX_t%5E%7B%28i%29%7D%7D%7Bdt%7D%20%3D%20%7Bf%28X_t%5E%7B%28i%29%7D%29%20-%20%5Cfrac%7BD%28X_t%5E%7B%28i%29%7D%29%7D%7B2%7D%20%5Cnabla%20%5Cln%20p_t%28X_t%5E%7B%28i%29%7D%29%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cnabla%20%5Ccdot%20D%28X_t%5E%7B%28i%29%7D%29%20%7D.%20%5Cquad%20%5Cquad%20%5Cquad%20%282%29">

  Eq.(1) and Eq.(2) imply that we may obtain transient solutions of the associated FPEs by simulating ensembles of **deterministic** trajectories/particles with initial conditions drawn from the starting distribution $p_0(x)$. 

  **However, the deterministic particle dynamics in Eq.(1) and Eq.(2) require the knowledge of $\nabla_x \ln p_t(x)$, i.e. the gradient of the logarithm of the quantity of interest. 
  Enter the gradient-log density estimator (score function estimator)!**


## B. **Gradient-log-density (score function) estimator**

## C. **Smooth transient solutions of Fokker-Planck equations**

<img src="high_speed.gif"  width="100%" height="100%">


<img src="OU2Dc.png"  width="20%" height="20%">


<img src="Linear_Ham.png"  width="30%" height="30%">


**Citations:**

1. Maoutsa, Dimitra; Reich, Sebastian; Opper, Manfred. [**Interacting Particle Solutions of Fokker–Planck Equations Through Gradient–Log–Density Estimation.**](https://www.mdpi.com/1099-4300/22/8/802/htm) _Entropy_ 2020, 22, 802. 

2. Hyvärinen, Aapo. **Estimation of non-normalized statistical models by score matching.** _Journal of Machine Learning Research_ 2005, 695-709.

