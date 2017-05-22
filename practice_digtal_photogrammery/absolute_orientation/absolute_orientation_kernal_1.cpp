#include "absolute_orientation_kernal_1.h"

void calculate_R_wrapper(
		Eigen::Matrix<double,3,3>& R,
		const Eigen::Matrix<double,7,1>& paras)
{
	calculate_R(paras(4),paras(5),paras(6),R);
}

void calculate_A_wrapper(
		Eigen::Matrix<double,3,7>& A,
		const Eigen::Matrix<double,7,1>& paras,
		const std::pair<Point,Point>& pt,
		const Eigen::Matrix<double,3,3>& R)
{
	calculate_A(pt.first,pt.second,paras,R,A);
}

void calculate_L_wrapper(
		Eigen::Matrix<double,3,1>& L,
		const Eigen::Matrix<double,7,1>& paras,
		const std::pair<Point,Point>& pt,
		const Eigen::Matrix<double,3,3>& R)
{
	calculate_L(pt.first,pt.second,paras,R,L);
}

void absolute_orientation_kernal_1(
		const std::vector<Point>& model_pts,
		const std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras,
		double& sigma_unti,
		Eigen::Matrix<double,7,1>& sigma_paras,
		std::vector<Eigen::Vector3d>& Vs)
{
	AdjustmentSolution_kernal<std::pair<Point,Point>,
					   Eigen::Matrix3d,
					   3,7> adjustment_solution;
	for(size_t i = 0; i < model_pts.size(); ++i)
	{
		adjustment_solution.addUnti(std::make_pair(
					model_pts[i],grand_pts[i]));
	}
	adjustment_solution
		.setFunc_A(calculate_A_wrapper)
		.setFunc_L(calculate_L_wrapper)
		.setFunc_Buffer(calculate_R_wrapper)
		.setInitalValue(paras);
	adjustment_solution
		.run()
		.data(paras,sigma_unti,sigma_paras,Vs);
}