#include <iostream>
#include <fstream>
#include <string>

#include "../practice_digtal_photogrammery/pdp.h"
void readFile(
		std::vector<Point>& model_pts,
		std::vector<Point>& grand_pts,
		const std::string& filename);


int main(int argc, char const *argv[])
{
	std::vector<Point> model_pts;
	std::vector<Point> grand_pts;

	// readFile(model_pts,grand_pts,"../data/work2.txt");
	moduleCall_kernal(
		"read file",
		std::cout,
		readFile,
		model_pts,grand_pts,
		"../data/work2.txt");

	Eigen::Matrix<double,7,1> paras;
	// initial(model_pts,grand_pts,paras);
	moduleCall_kernal(
		"initial parameters",
		std::cout,
		initial,
		model_pts,grand_pts,paras);
	Eigen::Matrix<double,7,1> sigma_paras;
	std::vector<Eigen::Vector3d> Vs;
	double sigma_unti;
	// absolute_orientation_kernal(model_pts,grand_pts,paras,sigma_unti,sigma_paras,Vs);
	moduleCall_kernal(
		"absolute orientation",
		std::cout,
		absolute_orientation_kernal,
		model_pts,grand_pts,
		paras,sigma_unti,sigma_paras,Vs);

	std::ofstream out("paras.txt");
	out<<"============================="<<std::endl;
	for(auto &V : Vs)
	{
		out<<"V:"<<std::endl<<V<<std::endl;
	}
	out<<"sigma_paras:"<<std::endl<<sigma_paras<<std::endl;
	out<<"sigma_unti:"<<sigma_unti<<std::endl;
	out<<"paras:"<<std::endl<<paras<<std::endl;
	return 0;
}

void readFile(
		std::vector<Point>& model_pts,
		std::vector<Point>& grand_pts,
		const std::string& filename)
{
	model_pts.clear();
	grand_pts.clear();
	std::ifstream fin(filename);
	std::string temp;
	int num;
	fin>>num;
	double m_x,m_y,m_z,g_x,g_y,g_z;
	for(size_t n = 0;n <num;++n)
	{
		fin>>temp
		   >>m_x>>m_y>>m_z
		   >>g_x>>g_y>>g_z;
		Eigen::Matrix<double,3,1> model_pt;
		model_pt<<m_x,m_y,m_z;
		Eigen::Matrix<double,3,1> grand_pt;
		grand_pt<<g_x,g_y,g_z;

		// std::cout<<"============================="<<std::endl;
		// std::cout<<"model_pt:"<<std::endl<<model_pt<<std::endl;
		// std::cout<<"grand_pt:"<<std::endl<<grand_pt<<std::endl;
		model_pts.push_back(model_pt);
		grand_pts.push_back(grand_pt);
	}
}