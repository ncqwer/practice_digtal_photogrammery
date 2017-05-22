#ifndef _ABSOLUTE_ORIENTATION_KERNAL_H_
#define _ABSOLUTE_ORIENTATION_KERNAL_H_ 

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>

typedef Eigen::Matrix<double,3,1> Point;

void absolute_orientation_kernal(
		const std::vector<Point>& model_pts,
		const std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras,
		double& sigma_unti,
		Eigen::Matrix<double,7,1>& sigma_paras,
		std::vector<Eigen::Vector3d>& Vs);

void initial(
		std::vector<Point>& model_pts,
		std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras);

void update(
		Eigen::Matrix<double,7,1>& delta,
		Eigen::Matrix<double,7,1>& paras);

bool isInTresh(
		const Eigen::Matrix<double,7,1>& delta);

void calculate_R(
		const double phi,
		const double omega,
		const double kappa,
		Eigen::Matrix<double,3,3>& R);

void calculate_A(
		const Point& model_pt,
		const Point& grand_pt,
		const Eigen::Matrix<double,7,1>& paras,
		const Eigen::Matrix<double,3,3>& R,
		Eigen::Matrix<double,3,7>& A);

void calculate_L(
		const Point& model_pt,
		const Point& grand_pt,
		const Eigen::Matrix<double,7,1>& paras,
		const Eigen::Matrix<double,3,3>& R,
		Eigen::Matrix<double,3,1>& L);
#endif