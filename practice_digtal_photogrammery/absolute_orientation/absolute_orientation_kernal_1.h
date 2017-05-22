#ifndef _ABSOLUTE_ORIENTATION_KERNAL_1_H_
#define _ABSOLUTE_ORIENTATION_KERNAL_1_H_

#include <utility>

#include <Eigen/Dense>
#include "../adjustment_solution.h"
#include "absolute_orientation_kernal.h"


void calculate_R_wrapper(
		Eigen::Matrix<double,3,3>& R,
		const Eigen::Matrix<double,7,1>& paras);

void calculate_A_wrapper(
		Eigen::Matrix<double,3,7>& A,
		const Eigen::Matrix<double,7,1>& paras,
		const std::pair<Point,Point>& pt,
		const Eigen::Matrix<double,3,3>& R);

void calculate_L_wrapper(
		Eigen::Matrix<double,3,1>& L,
		const Eigen::Matrix<double,7,1>& paras,
		const std::pair<Point,Point>& pt,
		const Eigen::Matrix<double,3,3>& R);

void absolute_orientation_kernal_1(
		const std::vector<Point>& model_pts,
		const std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras,
		double& sigma_unti,
		Eigen::Matrix<double,7,1>& sigma_paras,
		std::vector<Eigen::Vector3d>& Vs);

#endif


