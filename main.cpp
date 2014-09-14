#include <iostream>
#include <armadillo>


using namespace std;
using namespace arma;

/* This program solves the the equation f(x) = 100*e^(-10*x) first by using the tridiagonal algorithm
 * using n step points between [0,1]*/

int main()
{
 //i defines the row number we are currently performing the algorithm on and n is the size of our N x N matrix
    int i=0;
    int n=0;

    cout << "Enter value for n:" << endl;
    cin >> n;

//this is the temporary variable used to store intermediate calculations
    double btemp;

//h is used as the increment counter between approximation terms and defined as 1/(n+1)
    double h;
    h = 1.0 / (n+1);

//the arrays as defined by armadillo used to store the tridiagonal matrix a,b,c as well as a temporary array temp used for
//intermediate calculations
    vec a(n+2);
    a.fill(-1);
    vec b(n+2);
    b.fill(2);
    vec c(n+2);
    c.fill(-1);

    //vector for the function we are solving
    vec f(n+2);

    //vector for the numerical solution
    vec u(n+2);

    //vector for the analytical solution
    vec af(n+2);

//begin timer for part D tridiagonal algorithm
wall_clock tridiagtimer;
tridiagtimer.tic();

//define the function f that we are solving through iterative steps of the linear approximation as
//btilda(i) = h^2*f(i)
    for(i=0; i<=n+1; i++){
        f(i) = h*h*100.0*exp(-10.0*i*h);
        af(i) = 1- (1 - exp(-10.0))*(i*h) - exp(-10.0*(i*h));
    }
//forward substitution
//and decomposition
for(i=2; i<=n; i++){
    btemp = a(i)/b(i-1);
    b(i) = b(i) - btemp*c(i-1);
    f(i) = f(i) - btemp*f(i-1);
}
u(n) = f(n)/b(n);

//backward substitution
for(i=n-1; i>=1; i--){
    u(i) = (f(i) - c(i)*u(i+1))/b(i);
}

//end timer for part D tridiagonal algorithm
double tri_num_sec = tridiagtimer.toc();

cout << "tridiagonal finished in " << tri_num_sec << " seconds." << endl;

//relative error calculation
double maxError;
maxError = 0;
double tempError;
double umax;
double afmax;
int j;
int jmax;
double logRelError;
for(j=1; j<=n; j++){
    tempError = abs((af(j) - u(j))/af(j));
    if(tempError > maxError){
        maxError = tempError;
        umax = u(j);
        afmax = af(j);
        jmax = j;
    }

    //calculates the relative error based on the values
    logRelError = log10(maxError);

    //returns the max relative error between the tridiagonal algorithm solution and the analytical solution
    cout << "max relative error is: " << logRelError << endl;

    //outputting the values for the numerical and analytical solutions. The results are graphed
    //in google docs and included in the report file
    cout << "numerical values tridiagonal: " << endl << u << endl;
    cout << " analytical solution values: " << endl << af << endl;

    //LU Decomposition code to solve the same problem. used for part D to compare run times vs.
    // the tridiagonal algorithm above.

    //set up timer for lu decomposition
    wall_clock lutimer;
    lutimer.tic();

    //set up variables for the decomposition
    mat a = zeros<mat>(n,n);
    mat L;
    mat U;
    vec y;
    vec fsoln(n);

    for(i=0; i<n; i++){
          for(j=0; j<n; j++){
              if(i == j){
                  a(i,j) = 2.0;
              }
              else if(fabs(i-j) == 1){
                  a(i,j) = -1.0;
              }
          }
      }

//introduce the original function with length n to match lu algorithm.
for(i=1; i<n; i++){
    fsoln(i) = h*h*100.0*exp(-10.0*i*h);
}

//perform LU decomposition on a and solve for Lx=w(n) and then Ux=y(w(n)). output for LU
//does not include boundary points
lu(L,U,a);
y = solve(L,fsoln);

cout << "LU solution is: " << endl << solve(U,y) << endl;

//end timer for lu decomposition

double lu_num_sec = lutimer.toc();

cout << "and it took: " << lu_num_sec << " seconds." << endl;

}

    return 0;

}




