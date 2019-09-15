#include "ukf.h"
#include "Eigen/Dense"
#include "iostream"
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5,5);
  /*
  P_ = MatrixXd(5, 5);
  P_ << 0.89, 0, 0, 0, 0, 
        0, 0.89, 0, 0, 0, 
        0, 0, 0.89, 0, 0,
        0, 0, 0, 0.89, 0, 
        0, 0, 0, 0, 0.89;
  */
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.50;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   n_x_ = 5;

   n_aug_ = n_x_ + 2;

   lambda_ = 3 - n_x_;

   weights_ = VectorXd(2*n_aug_ + 1);

   double weight_0 = lambda_ /(lambda_ + n_aug_);
   weights_(0) = weight_0;

   for(int i = 1; i < 2*n_aug_ + 1; ++i)
   {
     weights_(i) = 0.5/(lambda_ + n_aug_);
   }

   R_radar = MatrixXd(3,3);
   R_radar << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_laser = MatrixXd(2,2);
   R_laser << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if(!is_initialized_)
   {
     if(meas_package.sensor_type_== MeasurementPackage::RADAR)
     {
       double rho = meas_package.raw_measurements_(0);
       double phi = meas_package.raw_measurements_(1);
       double rho_dot = meas_package.raw_measurements_(2);

       double x = rho * cos(phi);
       double y = rho * sin(phi);

       double vx = rho_dot * cos(phi);
       double vy = rho_dot * sin(phi);

       double v = sqrt(vx*vx + vy*vy);
       x_<<x, y, v, 0, 0;
     }
     else
     {
       x_ << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1), 0, 0, 0 ;
     }

     time_us_ = meas_package.timestamp_;
     is_initialized_ = true;
     return;
   }

   double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
   time_us_ = meas_package.timestamp_;
   Prediction(dt);

   if(meas_package.sensor_type_ ==  MeasurementPackage::RADAR && use_radar_)
   {
     //std::cout << " Updating RADAR Measurement" << std::endl;
     UpdateRadar(meas_package);
   }

   if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
   {
    //std::cout << " Updating Lidar Measurement" << std::endl;
     UpdateLidar(meas_package);
   }
  
  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  VectorXd X_aug = VectorXd(n_aug_);
  X_aug.head(5) = x_;
  X_aug(5) = 0;
  X_aug(6) = 0;
  
  
 
   
   //Create Augmented covariance Matrix
   MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
   P_aug.fill(0.0);
   P_aug.topLeftCorner(5,5) = P_;
   P_aug(5,5) = std_a_ * std_a_;
   P_aug(6,6) = std_yawdd_ * std_yawdd_;
  
   MatrixXd Xsig_aug = GenerateSigmaPoints(n_aug_, P_aug, X_aug);
  
  Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = PredictSigmaPoints(n_x_, n_aug_, delta_t, Xsig_aug);
 

   x_.fill(0.0);
   for(int i =0; i < 2*n_aug_ + 1; ++i)
   {
     x_ = x_ + weights_(i) * Xsig_pred_.col(i);
   }

   P_.fill(0.0);
   for(int i =0; i < 2*n_aug_ + 1; ++i)
   {
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     
     //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
     P_ =P_ + weights_(i) * x_diff * x_diff.transpose();
   }
   //std::cout << " Covariant Matrix = " << P_<< "\n" << "size = " << P_.col(1).size()<< std::endl;
}

MatrixXd UKF::GenerateSigmaPoints(int n_aug, MatrixXd P_aug, VectorXd X_aug)
{
  MatrixXd L = P_aug.llt().matrixL();
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2*n_aug + 1);
  
  Xsig_aug.col(0) = X_aug;
  for(int i = 0; i < n_aug; ++i)
  {
    Xsig_aug.col(i + 1)         = X_aug + sqrt(this->lambda_ + n_aug) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug) = X_aug - sqrt(this->lambda_ + n_aug) * L.col(i);
  }
  
  return Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(int n_x_, int n_naug_, double delta_t, MatrixXd Xsig_aug)
{
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2*n_naug_+1);
  //std::cout << "Xsig_pred size = " << Xsig_pred.size() << std::endl;

  for(int i = 0; i <2*n_naug_+1; ++i)
  {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    double px_pred;
    double py_pred;



    if(fabs(yawd)>0.001)
    {
      px_pred = p_x + (v/yawd)*( sin(yaw + delta_t * yawd) - sin(yaw));
      py_pred = p_y + (v/yawd)*( cos(yaw) - cos(yaw + delta_t * yawd) );
    }
    else
    {
      px_pred = p_x + v*delta_t*cos(yaw);
      py_pred = p_y + v*delta_t*sin(yaw);
    }

    double v_pred = v;
    double yaw_pred = yaw + delta_t * yawd;
    double yawd_pred = yawd;

    //add noise to the Predictions,
    px_pred += 0.5 *nu_a * delta_t * delta_t * cos(yaw);
    py_pred += 0.5 *nu_a * delta_t * delta_t * sin(yaw);
    v_pred += 0.5*nu_a * delta_t;
    yaw_pred += 0.5*nu_yawdd * delta_t * delta_t;
    yawd_pred += 0.5*nu_yawdd * delta_t;

    //write the predictions into X_pred
    Xsig_pred(0,i) = px_pred;
    Xsig_pred(1,i) = py_pred;
    Xsig_pred(2,i) = v_pred;
    Xsig_pred(3,i) = yaw_pred;
    Xsig_pred(4,i) = yawd_pred;
  }
  //std::cout << X_pred << std::endl;
  return Xsig_pred;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z =2;
  MatrixXd Z_sig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for(int i = 0; i < 2*n_aug_ + 1; ++i)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v; 
    
    //Measurement Model
    Z_sig(0,i) = p_x;
    Z_sig(1,i) = p_y;
    
  }


  // mean Predicted measurement
  VectorXd z_pred = VectorXd(n_z); 
  z_pred.fill(0.0);
  for(int i =0; i< 2*n_aug_+1; ++i)
  {
    z_pred = z_pred + weights_(i) * Z_sig.col(i);
  }

  //measurement covariance Matrix;
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i =0; i < 2*n_aug_ + 1; ++i)
  {
    VectorXd z_diff = Z_sig.col(i) - z_pred;
   
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // Add Measurement noise covariance matrix.
  S = S + R_laser;

  //create matrix for cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);

//incoming measurement.
  VectorXd z = meas_package.raw_measurements_;
  
  Tc.fill(0.0);
  for(int i =0; i < 2*n_aug_+1; ++i)
  {
    VectorXd z_diff = Z_sig.col(i) - z_pred;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    Tc = Tc + weights_(i) * x_diff *z_diff.transpose();
  }

  //Kalman Gain K
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  double NIS_laser = z_diff.transpose() * S.inverse() * z_diff;
  //std::cout << "NIS_Laser = " << NIS_laser << std::endl;
}



void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

   //Predict Radar Measurement
   int n_z =3;
   MatrixXd Z_sig = MatrixXd(n_z, 2*n_aug_ + 1);
   for(int i = 0; i < 2*n_aug_ + 1; ++i)
   {
     double p_x = Xsig_pred_(0,i);
     double p_y = Xsig_pred_(1,i);
     double v = Xsig_pred_(2,i);
     double yaw = Xsig_pred_(3,i);

     double v1 = v*cos(yaw);
     double v2 = v*sin(yaw);
     Z_sig(0,i) = sqrt(p_x*p_x + p_y*p_y);
     Z_sig(1,i) = atan2(p_y,p_x);
     if(Z_sig(0,i) < 0.001)
     {
       Z_sig(2,i) = (p_x*v1 + p_y*v2)/0.001;
     }
     else
     {
       Z_sig(2,i) = (p_x*v1 + p_y*v2)/(sqrt(p_x*p_x + p_y*p_y));
     }
   }
   //std::cout << Z_sig << std::endl; 

   // mean Predicted measurement
   VectorXd z_pred = VectorXd(n_z);
   z_pred.fill(0.0);
   for(int i =0; i< 2*n_aug_+1; ++i)
   {
     z_pred = z_pred + weights_(i) * Z_sig.col(i);
   }

   //measurement covariance Matrix;
   MatrixXd S = MatrixXd(n_z,n_z);
   S.fill(0.0);
   for(int i =0; i < 2*n_aug_ + 1; ++i)
   {
     VectorXd z_diff = Z_sig.col(i) - z_pred;
     
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

     S =S + weights_(i) * z_diff * z_diff.transpose();
   }
   // Add Measurement noise covariance matrix.
   S =S + R_radar;

   MatrixXd Tc = MatrixXd(n_x_, n_z);
 
 //incoming measurement.
   VectorXd z = meas_package.raw_measurements_;
   
   Tc.fill(0.0);
   for(int i =0; i < 2*n_aug_+1; ++i)
   {
     VectorXd z_diff = Z_sig.col(i) - z_pred;
     
     //angle normalization
     while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
     while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
     
     VectorXd x_diff = Xsig_pred_.col(i) - x_;   
     
     //angle normalization
     while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
     while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
     
     Tc = Tc + weights_(i) * x_diff *z_diff.transpose();
     }
 
     //Kalman Gain K
 
     MatrixXd K = Tc * S.inverse();
 
     //residual
     VectorXd z_diff = z - z_pred;
 
     //update state mean and covariance
     x_ = x_ + K * z_diff;
     P_ = P_ - K*S*K.transpose();
 
     double NIS_radar = z_diff.transpose() * S.inverse() * z_diff;
 
}