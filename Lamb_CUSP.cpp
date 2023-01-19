#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <mvt.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace arma;
using namespace std;

static double const log2pi = std::log(2.0 * M_PI);

//utility function =============================================================

// multivariate t log density
inline double log_t_density(const double nu, 
                            const vec &x, 
                            const vec &mu, 
                            const mat &lower_chol){
  int k=x.n_elem;
  double det_sig_half= sum(log(lower_chol.diag() ) );
  vec resid = solve(trimatl ( lower_chol ) , x - mu );
  
  double quad_form=dot(resid,resid)/nu;
  
  double density =gsl_sf_lnpoch(nu/2 , ((double)k)/2)-
    (k* log( (datum::pi)*nu) + (nu+k)*log1p(quad_form) )/2 -det_sig_half;
  
  return density;
}

// multivariate t log density  with independent components
inline double log_t_density_diag(const double nu,  
                                 const double sigma, 
                                 const vec &x){
  int k=x.n_elem;
  double density = k * (lgamma((nu+1)/2 ) - lgamma(nu/2)) -  
    k * log((datum::pi)*nu)/2 - k * log(sigma)/2- 
    (nu+1)/2 * sum(log(1 + ( x % x ) / (nu+sigma)));
  return density;
}

// rescale log probability to avoid underflow issues 
double log_sum_exp(const arma::vec &x) {
  double maxVal= x.max();
  double sum_exp=sum(exp(x-maxVal));
  return log(sum_exp)+maxVal ;
}

// delete void cluster e reorder labels 
void clean_clusters(uvec &c, uword &c_old, cube &Delta, uvec &n_k, 
                    mat &m_eta_temp, cube &V_eta_temp, 
                    cube &psi_eta_k, cube &chol_psi_eta_k){
  
  uword K = Delta.n_slices;
  uword H = Delta.n_cols;
  int cut = 0;
  for (uword i=0; i<K; i++){
    
    if (accu(c == i) == 0){
      
      for (uword j=(K-1); j>i; j--){
        
        if (accu(c == j) > 0){
          
          c(find(c == j)).fill(i);
          
          if (c_old == j){
            c_old = i;
          } 
          
          Delta.slice(i) = Delta.slice(j);
          n_k(i) = n_k(j);
          m_eta_temp.col(i) = m_eta_temp.col(j);
          V_eta_temp.slice(i) = V_eta_temp.slice(j);
          psi_eta_k.slice(i) = psi_eta_k.slice(j);
          chol_psi_eta_k.slice(i) = chol_psi_eta_k.slice(j);
          
          break;
        }
      }
    }
  }
  
  while( sum(c == cut) > 0 ){
    cut++;
  }
  n_k.resize(cut);
  Delta.resize(H, H, cut);
  m_eta_temp.resize(H, cut);
  V_eta_temp.resize(H, H, cut);
  psi_eta_k.resize(H, H, cut);
  chol_psi_eta_k.resize(H, H, cut);
}

// update alpha of DP 
double update_alpha(double alpha, 
                    double a, 
                    double b,
                    unsigned n, 
                    unsigned k){
  double phi=R::rbeta(alpha+1,(double) n);
  double gam1=a+k, gam2=b- log(phi);
  double pi=(gam1-1)/(n* gam2 );
  return (log(randu()) < log(pi)-log1p(pi)  ) ? 
  (randg(  distr_param(gam1,1/gam2) )) : (randg(  distr_param(gam1-1,1/gam2) ));
}

// =============================================================================
//' @export
//' @name Lamb_CUSP
//' @title C++ function to estimate the Lamb-CUSP
//'  clustering model via Gibbs sampling
//' @keywords internal
//'
//' @param y a matrix of observations (n x p)
//' @param n_iter nuumber of iteration
//' @param n_burn number of burn-in iterations
//' @param start_adapt iteration to start adapting latent space dimension
//' @param prior_par list of prior parameters:
//'   * xi variance of Delta
//'   * nu_0 degree of freedom of Delta
//'   * kappa_0 precision of mu
//'   * a_sigma shape of sigma
//'   * b_sigma rate of sigma
//'   * a_dir shape of precision par. of DP in clustering
//'   * b_dir rate of precision par. of DP ni clustering
//    * a_theta shape of theta in CUSP
//'   * b_theta rate of theta in CUSP
//'   * alpha_f precision par. of DP in CUSP
//'   * theta_lim variance of spike in CUSP
//' @param start list of initial values of parameters:
//'   * H dimension of latent space
//'   * K number of clusters
//'   * Lambda matrix of loadings (p x H)
//'   * eta matrix of latent factors (n x H)
//'   * sigma vector of diagonal variances of data (p x 1)
//'   * c vector of cluster menbership [0, K-1] (n x 1)
//'   
//' @return list:
//'   * H latent space dimension
//'   * n_cluster number of clusters
//'   * c clusters allocation
//

// [[Rcpp::export]]
Rcpp::List Lamb_CUSP(mat y, uword n_iter, uword n_burn, uword start_adapt, 
                    Rcpp::List prior_par, Rcpp::List start){
  
  Progress progress(n_iter, true);
  // prior parameters
  double a_sigma   = prior_par["a_sigma"];
  double b_sigma   = prior_par["b_sigma"];
  double a_theta   = prior_par["a_theta"];
  double b_theta   = prior_par["b_theta"];
  double xi        = prior_par["xi"];
  double kappa_0   = prior_par["kappa_0"];
  uword  nu_0      = prior_par["nu_0"];
  double a_dir     = prior_par["a_dir"];
  double b_dir     = prior_par["b_dir"];
  double alpha_f   = prior_par["alpha_f"];
  double theta_lim = prior_par["theta_lim"];
  
  // values for the adaptation of latent space dimension
  double alpha_0 = -1;
  double alpha_1 = -0.0005;
  
  uword p = y.n_cols; // number of features
  uword n = y.n_rows; // number of observation
  uword H = start["H"]; // latent space dimension
  uword K = start["K"]; // number of cluster
  
  // initialization ============================================================
  mat Lambda = start["Lambda"];// p x H 
  mat eta = start["eta"];      // n x H
  vec sigma = start["sigma"];  // p x 1
  vec theta = start["theta"];  // H x 1
  uvec c = start["c"]; 
  cube Delta(H, H, K);
  uvec z(H);
  vec v(H);
  vec w(H, fill::value(1));
  w = w/H;
  double alpha_cl = 1;
  uword H_star=H;
  uvec active(H);
  
  // temp variables=============================================================
  
  //// temp for Lambda
  mat V_Lambda_temp(H, H);
  mat eta2(H, H);
  
  //// temp for eta
  mat L_sigma_L_prod(H, H);
  mat L_sigma_prod(p, H);
  cube chol_Omega_temp(H, H, K);
  cube chol_Delta(H, H, K);
  vec rho_k(H);
  mat m_eta_temp(H, K);
  cube V_eta_temp(H,H, K);
  cube psi_eta_k(H,H, K);
  cube chol_psi_eta_k(H,H, K);
  
  //// temp for c
  uvec n_k(K);
  uword n_k_wo_i;
  uvec idx_k(1), idx_k_wo_i(1);
  vec m_eta_wo_i(H);
  mat V_eta_wo_i(H, H);
  mat psi_eta_wo_i(H, H);
  
  //// temp for sigma
  double a_sigma_temp;
  double b_sigma_temp;
  double m1_temp;
  double m2_temp;
  
  //// temp for CUSP
  double n1_v;
  double n2_v;
  double Lambda_sum_squared=0;
  
  //// other temp variables
  mat Lambda_temp(p, H);
  mat eta_temp(n, H);
  vec theta_temp(H);
  vec w_temp(H);
  vec v_temp(H);
  cube Delta_temp(H, H, K);
  bool req_clean = false;
  double flag = 0;
  
  // output ====================================================================
  Rcpp::List out;
  vec H_out(n_iter-n_burn);
  umat c_out(n, n_iter-n_burn);
  uvec n_cluster( n_iter-n_burn);
  
  // initial comoutation of temp variables =====================================
  for (uword k=0; k<K; k++){
    n_k(k) = accu(c == k);
    idx_k.set_size(n_k(k));
    idx_k = find(c == k);
    
    m_eta_temp.col(k) = sum(eta.rows(idx_k),0).as_col();
    V_eta_temp.slice(k) = trans(eta.rows(idx_k)) * eta.rows(idx_k);
    
    psi_eta_k.slice(k) = symmatu(V_eta_temp.slice(k) - 
      m_eta_temp.col(k) * trans(m_eta_temp.col(k)) / (kappa_0 + n_k(k)));
    psi_eta_k.slice(k).diag() += xi;
    
  }
  
  // START GIBBS ===============================================================
  for(uword it=0; it<n_iter; it++){
    
    // update Delta ============================================================
    
    for (uword k=0; k<K; k++){
      
      Delta.slice(k) = iwishrnd(psi_eta_k.slice(k), nu_0 + n_k(k));
      
      chol( chol_Delta.slice(k), Delta.slice(k) ); 
    }
    
    if (Delta.has_nan()){
      Rcpp::stop("Delta has nan");
    }
    
    // update Lambda ===========================================================
    
    eta2 = eta.t() * eta;
    for (uword j=0; j<p; j++){
      V_Lambda_temp = chol(diagmat(1/theta) + 1/sigma(j)*eta2);
      // back forward substitution
      Lambda.row(j) = (solve(trimatu(V_Lambda_temp ), 
                       randn<vec>(H) +solve(trimatl((V_Lambda_temp ).t()), 
                                  1/sigma(j) * (eta.t() * y.col(j)) ) ) ).t();
    }
    
    if (Lambda.has_nan()){
      Rcpp::stop("Lambda has nan");
    }
    
    // update eta ==============================================================
    L_sigma_prod = Lambda.each_col() % (1/sigma);
    L_sigma_L_prod =  symmatu(L_sigma_prod.t() * Lambda);
    
    //// compute Omega 
    for (uword k=0; k<K; k++){
      chol_Omega_temp.slice(k) = (chol(L_sigma_L_prod + 
        inv_sympd(Delta.slice(k))));
    } 
    
    for (uword i=0; i<n; i++){
      
      rho_k = L_sigma_prod.t() * y.row(i).t();
      m_eta_temp.col(c(i)) = m_eta_temp.col(c(i)) - eta.row(i).as_col();
      
      rho_k += inv_sympd(Delta.slice(c(i))) *
        m_eta_temp.col(c(i)) / (n_k(c(i))-1+kappa_0);
      rho_k += solve(trimatu(chol_Delta.slice(c(i))), 
                     randn<vec>(H)) / sqrt(n_k(c(i))-1+kappa_0);
      
      eta.row(i) = solve(trimatu(chol_Omega_temp.slice(c(i))), 
              randn<vec>(H) + 
                solve(trimatl((chol_Omega_temp.slice(c(i))).t()), rho_k)).t();
      m_eta_temp.col(c(i)) = m_eta_temp.col(c(i)) + eta.row(i).as_col();
    }
  
    if (eta.has_nan()){
      Rcpp::stop("eta has nan");
    }
    
    // update of temp variables ================================================
    for (uword k=0; k<K; k++){
      
      idx_k.set_size(n_k(k));
      idx_k = find(c == k);
      
      V_eta_temp.slice(k) = trans(eta.rows(idx_k)) * eta.rows(idx_k);
      
      psi_eta_k.slice(k) = symmatu(V_eta_temp.slice(k) - 
        m_eta_temp.col(k) * trans(m_eta_temp.col(k)) / (kappa_0 + n_k(k)));
      psi_eta_k.slice(k).diag() += xi;
      chol_psi_eta_k.slice(k) = chol(psi_eta_k.slice(k), "lower");
    }
    
    // update clusters =========================================================
    for (uword i=0; i<n; i++){
      
      uword c_old = c(i);
      req_clean = false;
      
      // if the cluster of unit i is a singleton, I remove that cluster
      if (accu(c == c(i)) == 1){
        req_clean = true;
        c_old = n;
      }
      
      // assign i to a "not-possible" cluster for convenience of code writing
      c(i) = n;
      
      // clusters clean
      if (req_clean == true){
        clean_clusters(c, c_old, Delta, n_k, m_eta_temp, V_eta_temp, 
                       psi_eta_k, chol_psi_eta_k);
        K = Delta.n_slices;
      }
      
      Rcpp::NumericVector log_prob_cluster(K+1);
      Rcpp::NumericVector prob_cluster(K+1);
      
      for (uword k=0; k<K; k++){
        
        if (c_old == k){ // for the old cluster of unit i
          n_k_wo_i = n_k(k)-1;
          m_eta_wo_i = m_eta_temp.col(k) - eta.row(i).as_col();
          V_eta_wo_i = V_eta_temp.slice(k) - trans(eta.row(i)) * eta.row(i);
          
          psi_eta_wo_i = symmatu(V_eta_wo_i - m_eta_wo_i * m_eta_wo_i.t() / 
            (kappa_0 + n_k_wo_i));
          psi_eta_wo_i.diag() += xi;
          
          log_prob_cluster(k) = log(n_k_wo_i) + 
            log_t_density(nu_0 + n_k_wo_i - H + 1 ,  
                          (eta.row(i)).t(),
                           m_eta_wo_i / (kappa_0 + n_k_wo_i), 
                           chol((kappa_0 + n_k_wo_i + 1) / 
                             ((kappa_0 + n_k_wo_i) * (nu_0 + n_k_wo_i - H + 1))
                                  *psi_eta_wo_i, "lower"));
          prob_cluster(k) = exp(log_prob_cluster(k));
        } else{ // for all other clusters
          
          log_prob_cluster(k) = log(n_k(k)) + 
            log_t_density(nu_0 + n_k(k) - H + 1 ,  
                          (eta.row(i)).t(),
                           m_eta_temp.col(k) / (kappa_0 + n_k(k)), 
                           sqrt((kappa_0 + n_k(k) + 1) / 
                             ((kappa_0 + n_k(k)) * (nu_0 + n_k(k) - H + 1))) * 
                             chol_psi_eta_k.slice(k));
          prob_cluster(k) = exp(log_prob_cluster(k));
        }
      }
      // prob of a new cluster
      log_prob_cluster(K) = log(alpha_cl) + 
        log_t_density_diag((double) (nu_0 - H + 1), 
                           (kappa_0+1)/(kappa_0*(nu_0-H+1))*xi, 
                            (eta.row(i)).t());
      prob_cluster(K) = exp(log_prob_cluster(K));
      
      Rcpp::IntegerVector c_idx = Rcpp::seq(0, K);
      c(i) = Rcpp::sample(c_idx, 1, false, prob_cluster)(0);
      
      // update of utility variables
      // if the unit i is assigned to a different cluster
      if (c(i) != c_old){ 
        // and if the unit i was not a sigleton
        if (c_old != n){  
          // remove its contribution from its old cluster
          n_k(c_old)--;
          m_eta_temp.col(c_old) = m_eta_temp.col(c_old) - eta.row(i).as_col();
          V_eta_temp.slice(c_old) = V_eta_temp.slice(c_old) - 
            trans(eta.row(i)) * eta.row(i);
          
          psi_eta_k.slice(c_old) = symmatu(V_eta_temp.slice(c_old) - 
            m_eta_temp.col(c_old) * m_eta_temp.col(c_old).t() / 
            (kappa_0 + n_k(c_old)));
          psi_eta_k.slice(c_old).diag() += xi;
          chol_psi_eta_k.slice(c_old) = chol(psi_eta_k.slice(c_old), "lower");
        }
        
        if (c(i) == K){ 
          // if unit i is assigned to a completely new cluster
          K = K+1;
          Delta.resize(H, H, K);
          Delta.slice(K-1) = iwishrnd(xi * eye(H,H), nu_0);
          n_k.resize(K);
          n_k(K-1) = 1;
          m_eta_temp.resize(H, K);
          m_eta_temp.col(K-1) = eta.row(i).as_col();
          V_eta_temp.resize(H, H, K);
          V_eta_temp.slice(K-1) = trans(eta.row(i)) * eta.row(i);
          psi_eta_k.resize(H, H, K);
          psi_eta_k.slice(K-1) = symmatu(V_eta_temp.slice(K-1) - 
            m_eta_temp.col(K-1) * m_eta_temp.col(K-1).t() / 
            (kappa_0 + n_k(K-1)));
          psi_eta_k.slice(K-1).diag() += xi;
          chol_psi_eta_k.resize(H, H, K);
          chol_psi_eta_k.slice(K-1) = chol(psi_eta_k.slice(K-1), "lower");
          
        } else { 
          // if unit i is assigned to a new cluster among those thta already exist
          n_k(c(i))++;
          
          m_eta_temp.col(c(i)) = m_eta_temp.col(c(i)) + eta.row(i).as_col();
          V_eta_temp.slice(c(i)) = V_eta_temp.slice(c(i)) + 
            trans(eta.row(i)) * eta.row(i);
          
          psi_eta_k.slice(c(i)) = symmatu(V_eta_temp.slice(c(i)) - 
            m_eta_temp.col(c(i)) * m_eta_temp.col(c(i)).t() / 
            (kappa_0 + n_k(c(i))));
          psi_eta_k.slice(c(i)).diag() += xi;
          chol_psi_eta_k.slice(c(i)) = chol(psi_eta_k.slice(c(i)), "lower");
        } 
      }
    }
    
    chol_Omega_temp.set_size(H, H, K);
    chol_Delta.set_size(H, H, K);
    
    // update CUSP parameters ==================================================
    // update z ================================================================
    for (uword h=0; h<H; h++){
      Rcpp::NumericVector log_prob_z(H);
      Rcpp::NumericVector prob_z(H);
      for (uword l=0; l<H; l++){
        if (l<=h){
          log_prob_z(l) = log(w(l)) + 
            as_scalar(sum(log_normpdf(Lambda.col(h), 0, sqrt(theta_lim))));
        } else{
          log_prob_z(l) = log(w(l)) + 
            log_t_density_diag( 2*a_theta, b_theta/a_theta, Lambda.col(h));
        }
      }
      
      // to avoid underflow issues
      prob_z = exp(log_prob_z - log_sum_exp(log_prob_z));
      
      if (sum(prob_z) == 0){
        cout << "WARNING: all probabilities for z equal to 0..." << endl;
        prob_z.fill(0);
        prob_z(H-1) = 1;
      }
      
      Rcpp::IntegerVector h_index = Rcpp::seq(0, H-1);
      z(h) = Rcpp::sample(h_index, 1, false, prob_z)(0);
    }
    
    // update v ================================================================
    for (uword l=0; l<(H-1); l++){
      n1_v = sum(z == l);
      n2_v = sum(z > l);
      v(l) = Rcpp::as<double>(Rcpp::wrap(R::rbeta(1 + n1_v, alpha_f + n2_v)));
    }
    v(H-1) = 1;
    
    // update w ================================================================
    w(0) = v(0);
    for (uword l=1; l<H; l++){
      w(l) = v(l) * (1-v(l-1)) * (w(l-1)) / (v(l-1));
    }
    
    // update theta ============================================================
    for (uword h=0; h<H; h++){
      if (z(h) <= h){
        theta(h) = theta_lim;
      } else{
        
        Lambda_sum_squared = as_scalar(trans(Lambda.col(h)) * Lambda.col(h));
        theta(h) = 1 / randg(distr_param(a_theta + 0.5*p, 
                             1/( b_theta + 0.5*Lambda_sum_squared)));
      }
    }
    
    // update precision parameter of DP =========================================================
    alpha_cl = update_alpha(alpha_cl, a_dir, b_dir, n, K);
    
    // update sigma ============================================================
    for (uword j=0; j<p; j++){
      m1_temp = 0;
      m2_temp = 0;
      a_sigma_temp = a_sigma + 0.5*n;
      for (uword i=0; i<n; i++){
        m1_temp = as_scalar(Lambda.row(j) * eta.row(i).as_col());
        m2_temp = m2_temp + pow(y(i,j) - m1_temp, 2);
      }
      b_sigma_temp = b_sigma + 0.5*m2_temp;
      
      sigma(j) = 1 / randg(distr_param(a_sigma_temp, 1/b_sigma_temp));
    }
    
    // update latent space dimension============================================
    if ((it > start_adapt) && (randu() < exp(alpha_0+alpha_1*it))){
      
      Lambda_temp = Lambda;
      eta_temp = eta;
      Delta_temp = Delta;
      theta_temp = theta;
      w_temp = w;
      v_temp = v;
      
      uvec H_index = linspace<uvec>(0, H-1, H);
      H_star = sum(z > H_index);
      active.resize(H_star);
      active = find(z > H_index);
      
      
      if ((H_star < (H-1)) & (H_star > 0)){
        flag = 1;
        H = H_star + 1;
        Lambda.set_size(p, H);
        Lambda.cols(0, H-2) = Lambda_temp.cols(active);
        eta.set_size(n, H);
        eta.cols(0, H-2) = eta_temp.cols(active);
        theta.set_size(H);
        theta.subvec(0, H-2) = theta_temp(active);
        v.set_size(H);
        v.subvec(0, H-2) = v_temp(active);
        w.set_size(H);
        w.subvec(0, H-2) = w_temp(active);
        Delta.set_size(H ,H, K);
        eta.tail_cols(1) = randn(n) * 
          sqrt(1 / randg(distr_param(nu_0/2, 1/(xi/2)))); // from the prior (to check)
        theta.tail(1) = theta_lim;
        w.tail(1) = 0;
        w.tail(1) = 1 - sum(w);
        Lambda.tail_cols(1) = randn(p) * sqrt(theta_lim);
      } else {
        
        flag = 1;
        H = H + 1;
        Lambda.resize(p,H);
        eta.resize(n, H);
        theta.resize(H);
        v.resize(H);
        w.resize(H);
        Delta.set_size(H ,H, K);
        eta.tail_cols(1) = randn(n) * 
          sqrt(1 / randg(distr_param(nu_0/2, 1/(xi/2 // from the prior (to check)
        theta.tail(1) = theta_lim;
      
        
        if (H > 1){
          v(H-2) = Rcpp::as<double>(Rcpp::wrap(R::rbeta(1 , alpha_f)));
          v(H-1) = 1; 
          if (H > 2){
            w(H-2) = v(H-2) * (1-v(H-3)) * (w(H-3)) / (v(H-3));
          } else {
            w(H-2) = v(H-2);
          }
          w.tail(1) = v.tail(1) * (1-v(H-2)) * (w(H-2)) / (v(H-2));
        } else{
          v.tail(1) = 1;
          w.tail(1) = 1;
        }
        
        Lambda.tail_cols(1) =  randn(p) * sqrt(theta_lim);
        
        
      }
      
      if (flag==1){ // resize the temp variable accordly to the new latent dimension H
        flag = 0;
        Lambda_temp.set_size(p, H);
        eta_temp.set_size(n, H);
        theta_temp.set_size(H);
        w_temp.set_size(H);
        v_temp.set_size(H);
        V_Lambda_temp.set_size(H, H);
        eta2.set_size(H,H);
        z.set_size(H);
        Delta_temp.resize(H, H, K);
          
        m_eta_temp.resize(H, K);
        V_eta_temp.resize(H,H,K);
        psi_eta_k.resize(H,H,K);
        chol_psi_eta_k.resize(H,H,K);
        
        for (uword k=0; k<K; k++){
          idx_k.set_size(n_k(k));
          idx_k = find(c == k);
          
          m_eta_temp.col(k) = sum(eta.rows(idx_k),0).as_col();
          V_eta_temp.slice(k) = trans(eta.rows(idx_k)) * eta.rows(idx_k);
          
          psi_eta_k.slice(k) = symmatu(V_eta_temp.slice(k) - m_eta_temp.col(k) * 
            trans(m_eta_temp.col(k)) / (kappa_0 + n_k(k)));
          psi_eta_k.slice(k).diag() += xi;
          
        }
        
        L_sigma_prod.set_size(p,H);
        L_sigma_L_prod.set_size(H,H);
        
        chol_Omega_temp.set_size(H, H, K);
        chol_Delta.set_size(H, H, K);
        m_eta_wo_i.set_size(H);
        V_eta_wo_i.set_size(H, H);
        
        rho_k.set_size(H);
        psi_eta_wo_i.set_size(H, H);
      }
      

    }
    // uncomment for debuging
    // if (it % 10 ==0){
    //   cout << endl << "dim latent space: " << H << endl;
    //   cout << "n. clusters: " << K << endl;
    // }
    
    // save output =============================================================
    if ((it >= n_burn)){
      H_out(it-n_burn) = H_star;
      c_out.col(it-n_burn) = c;
      n_cluster(it-n_burn) = Delta.n_slices;
    }
    
    progress.increment();
  }
  // END GIBBS =================================================================
  
  out["H"] = H_out;
  out["clust"] = c_out;
  out["n_cluster"] = n_cluster;
  
  return out;
}