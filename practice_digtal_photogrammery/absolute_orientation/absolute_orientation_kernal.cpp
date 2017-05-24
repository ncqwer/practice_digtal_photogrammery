#include "absolute_orientation_kernal.h"

void absolute_orientation_kernal(
		const std::vector<Point>& model_pts,
		const std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras,
		double& sigma_util,
		Eigen::Matrix<double,7,1>& sigma_paras,
		std::vector<Eigen::Vector3d>& Vs)
{
	Vs.clear();

	for(size_t count = 0; count < 7; ++count)
	{
		Eigen::Matrix<double,7,7> ATA;
		ATA = Eigen::Matrix<double,7,7>::Zero();
		Eigen::Matrix<double,7,1> ATL;
		ATL = Eigen::Matrix<double,7,1>::Zero();

		auto model_begin = model_pts.cbegin();
		auto grand_begin = grand_pts.cbegin();

		auto model_end = model_pts.cend();
		auto grand_end = grand_pts.cend();

		Eigen::Matrix<double,3,7> A;
		Eigen::Matrix<double,3,1> L;
		for(;
			model_begin != model_end && grand_begin != grand_end;
			++model_begin,++grand_begin)
		{
			Eigen::Matrix<double,3,3> R;
			calculate_R(paras(4),paras(5),paras(6),R);
			calculate_A(*model_begin,*grand_begin,paras,R,A);
			calculate_L(*model_begin,*grand_begin,paras,R,L);
			// std::cout<<"============================="<<std::endl;
			// std::cout<<"R:"<<std::endl<<R<<std::endl;
			// std::cout<<"A:"<<std::endl<<A<<std::endl;
			// std::cout<<"L:"<<std::endl<<L<<std::endl;
			ATA += A.transpose()*A;
			ATL += A.transpose()*L;
		}
		Eigen::Matrix<double,7,1> delta = ATA.inverse()*ATL;
		std::cout<<"============================="<<std::endl;
		std::cout<<"ATA:"<<std::endl<<ATA<<std::endl;
		std::cout<<"ATL:"<<std::endl<<ATL<<std::endl;
		std::cout<<"paras:"<<std::endl<<paras<<std::endl;
		std::cout<<"delta:"<<std::endl<<delta<<std::endl;
		if(isInTresh(delta))
		{
			double VTV = 0;
			auto model_begin_nest = model_pts.cbegin();
			auto grand_begin_nest = grand_pts.cbegin();

			Eigen::Matrix<double,3,7> A;
			Eigen::Matrix<double,3,1> L;
			for(;
				model_begin_nest != model_end && grand_begin_nest != grand_end;
				++model_begin_nest,++grand_begin_nest)
			{
				Eigen::Matrix<double,3,3> R;
				calculate_R(paras(4),paras(5),paras(6),R);
				calculate_A(*model_begin_nest,*grand_begin_nest,paras,R,A);
				calculate_L(*model_begin_nest,*grand_begin_nest,paras,R,L);
				Eigen::Vector3d V = A*delta - L;
				Vs.push_back(V);
				VTV += V.transpose()*V;
			}
			sigma_util = std::sqrt(VTV/(3*grand_pts.size() - 7));
			Eigen::Matrix<double,7,7> Q;
			Q = ATA.inverse();
			sigma_paras = sigma_util*Q.diagonal().array().sqrt();
			update(delta,paras);
			break;
		}
		update(delta,paras);
	}
} 

void update(
		Eigen::Matrix<double,7,1>& delta,
		Eigen::Matrix<double,7,1>& paras)
{
	double temp_lamba = delta(3);
	// paras(3) *= 1 + temp_lamba;
	paras(3) += temp_lamba;
	// paras(0) += delta(0);
	// paras(1) += delta(1);
	// paras(2) += delta(2);
	paras(4) += delta(4);
	paras(5) += delta(5);
	paras(6) += delta(6);
}

void initial(
		std::vector<Point>& model_pts,
		std::vector<Point>& grand_pts,
		Eigen::Matrix<double,7,1>& paras)
{
	Point model_sum = Point::Zero();
	for(auto &model_pt : model_pts)
	{
		model_sum += model_pt;
	}
	model_sum /= model_pts.size();
	for(auto &model_pt : model_pts)
	{
		model_pt -= model_sum;
		std::cout<<"model_pt:"<<std::endl<<model_pt<<std::endl;
	}

	Point grand_sum = Point::Zero();
	for(auto &grand_pt : grand_pts)
	{
		grand_sum += grand_pt;
	}
	grand_sum /= grand_pts.size();
	for(auto &grand_pt : grand_pts)
	{
		grand_pt -= grand_sum;
		std::cout<<"grand_pt:"<<std::endl<<grand_pt<<std::endl;

	}
	std::cout<<"============================="<<std::endl;
	std::cout<<"model_sum:"<<std::endl<<model_sum<<std::endl;
	std::cout<<"grand_sum:"<<std::endl<<grand_sum<<std::endl;
	auto diff = grand_sum - model_sum;
	paras(0) = 0;
	paras(1) = 0;
	paras(2) = 0;
	paras(3) = 1;
	paras(4) = 0;
	paras(5) = 0;
	paras(6) = 0;
}

void calculate_R(
		const double phi,
		const double omega,
		const double kappa,
		Eigen::Matrix<double,3,3>& R)
{
	using std::cos;
	using std::sin;
	Eigen::Matrix3d phi_m, omega_m, kappa_m;
	omega_m << 1, 0, 0,
	        0, cos(omega), -sin(omega),
	        0, sin(omega), cos(omega);

	phi_m << cos(phi), 0, -sin(phi),
	      0, 1, 0,
	      sin(phi), 0, cos(phi);

	kappa_m << cos(kappa), -sin(kappa), 0,
	        sin(kappa), cos(kappa), 0,
	        0, 0, 1;
	R = phi_m * omega_m * kappa_m;
}

void calculate_A(
		const Point& model_pt,
		const Point& grand_pt,
		const Eigen::Matrix<double,7,1>& paras,
		const Eigen::Matrix<double,3,3>& R,
		Eigen::Matrix<double,3,7>& A)
{
	using std::sin;
	using std::cos;
	A = Eigen::Matrix<double,3,7>::Zero();
	auto temp_pt = R*model_pt;
	double temp_X = temp_pt(0);
	double temp_Y = temp_pt(1);
	double temp_Z = temp_pt(2);

	double lamba = paras(3);
	double sin_phi = sin(paras(4));
	double cos_phi = cos(paras(4));
	double sin_omega = sin(paras(5));
	double cos_omega = cos(paras(5));

	Eigen::Matrix<double,3,1> derivate_phi;
	derivate_phi(0) = -lamba*temp_Z;
	derivate_phi(1) = 0;
	derivate_phi(2) = lamba*temp_X;

	Eigen::Matrix<double,3,1> derivate_omega;
	derivate_omega(0) = -lamba*temp_Y*sin_phi;
	derivate_omega(1) = lamba*temp_X*sin_phi - lamba*temp_Z*cos_phi;
	derivate_omega(2) = lamba*temp_Y*cos_phi;

	Eigen::Matrix<double,3,1> derivate_kappa;
	derivate_kappa(0) = -lamba*temp_Y*cos_phi*cos_omega - lamba*temp_Z*sin_omega;
	derivate_kappa(1) = lamba*temp_X*cos_phi*cos_omega + lamba*temp_Z*sin_phi*cos_omega;
	derivate_kappa(2) = lamba*temp_X*sin_phi - lamba*temp_Y*sin_phi*cos_omega;
	
	A.block<3,3>(0,0) = Eigen::Matrix<double,3,3>::Identity();
	A.block<3,1>(0,3) = temp_pt;
	A.block<3,1>(0,4) = derivate_phi;
	A.block<3,1>(0,5) = derivate_omega;
	A.block<3,1>(0,6) = derivate_kappa;
}

void calculate_L(
		const Point& model_pt,
		const Point& grand_pt,
		const Eigen::Matrix<double,7,1>& paras,
		const Eigen::Matrix<double,3,3>& R,
		Eigen::Matrix<double,3,1>& L)
{
	L = Eigen::Matrix<double,3,1>::Zero();
	auto temp_pt = paras(3)*R*model_pt;
	Eigen::Matrix<double,3,1> tvec;
	tvec(0) = paras(0);
	tvec(1) = paras(1);
	tvec(2) = paras(2);
	L = grand_pt - temp_pt - tvec;
}

bool isInTresh(
		const Eigen::Matrix<double,7,1>& delta)
{
	using std::fabs;
	double _thresh = 10e-6;
	if (
		// (fabs(delta(0)) > _thresh) || 
		// (fabs(delta(1)) > _thresh) || 
		// (fabs(delta(2)) > _thresh) ||
		// (fabs(delta(3)) > _thresh) ||
		(fabs(delta(4)) > _thresh) ||
		(fabs(delta(5)) > _thresh) ||
		(fabs(delta(6)) > _thresh) )
	{
		return false;
	}
	return true;
}