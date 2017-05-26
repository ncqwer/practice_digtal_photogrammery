#ifndef _ADJUSTMENT_SOLUTION_KERNAL_H_
#define _ADJUSTMENT_SOLUTION_KERNAL_H_


#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

template<typename UntiType, 
		 typename BufferType, 
		 std::size_t _num__one_until, 
		 std::size_t _num_paras>
class AdjustmentSolution_kernal
{
public:
	typedef Eigen::Matrix<double,_num__one_until,_num_paras> TypeA;
	typedef Eigen::Matrix<double,_num__one_until,1> TypeL;
	typedef Eigen::Matrix<double,_num__one_until,1> TypeV;
	typedef Eigen::Matrix<double,_num_paras,1> TypeParas;

	typedef std::function<void(TypeA&,
	                           const TypeParas&,
	                           const UntiType&,
	                           const BufferType&)> FuncTypeA;
	typedef std::function<void(TypeL&,
	                           const TypeParas&,
	                           const UntiType&,
	                           const BufferType&)> FuncTypeL;
	typedef std::function<void(BufferType&,
	                           const TypeParas&)> FuncTypeBuffer;
	AdjustmentSolution_kernal() {}
	~AdjustmentSolution_kernal() {}

	AdjustmentSolution_kernal(const AdjustmentSolution_kernal& rhs) = delete;
	AdjustmentSolution_kernal(AdjustmentSolution_kernal&& rhs) = delete;

	AdjustmentSolution_kernal& operator= (AdjustmentSolution_kernal rhs_copy) = delete;

	//==============================

	AdjustmentSolution_kernal& run()
	{
		for (std::size_t iter_index = 0; iter_index < _max_num_iter; ++iter_index)
		{
			_func_buff(_buffer_data, _paras);
			size_t index_rows = 0;
			// cout<<"the R is "<<endl<<_buffer_data<<endl;
			TypeA A;
			TypeL L;
			Eigen::Matrix<double,_num_paras,_num_paras> ATA=Eigen::Matrix<double,_num_paras,_num_paras>::Zero();
			Eigen::Matrix<double,_num_paras,1> ATL=Eigen::Matrix<double,_num_paras,1>::Zero();
			for (auto &unti : _untis)
			{
				_func_A(A, _paras, unti, _buffer_data);
				_func_L(L, _paras, unti, _buffer_data);
				ATA+=(A.transpose() * A);
				ATL+=(A.transpose()*L);
			}
			Eigen::MatrixXd delta = ATA.inverse() * ATL;
			Eigen::MatrixXd new_paras = _paras + delta;
			// std::cout<<"============================="<<std::endl;
			// std::cout<<"ATA:"<<std::endl<<ATA<<std::endl;
			// std::cout<<"ATL:"<<std::endl<<ATL<<std::endl;
			// std::cout<<"paras:"<<std::endl<<_paras<<std::endl;
			// std::cout<<"delta:"<<std::endl<<delta<<std::endl;
			if (isInThresh(delta))
			{
				double VTV=0.0;
				for (auto &unti : _untis)
				{
					_func_A(A, _paras, unti, _buffer_data);
					_func_L(L, _paras, unti, _buffer_data);
					Eigen::MatrixXd V = A*delta - L;
					_Vs.push_back(V);
					VTV += (V.transpose() * V)(0);
				}
				_sigma_util = sqrt(VTV/(_num__one_until*_untis.size() - _num_paras));
				Eigen::MatrixXd Q = ATA.inverse();
				_sigmas = _sigma_util*Q.diagonal().array().sqrt();
				_paras = new_paras;
				break;
			}
			_paras = new_paras;
		}
		return *this;
	}

	AdjustmentSolution_kernal& addUnti(
			const UntiType& unti)
	{
		_untis.push_back(unti);
		return *this;
	}

	AdjustmentSolution_kernal& setFunc_A(
			FuncTypeA func)
	{
		_func_A = func;
		return *this;
	}
	AdjustmentSolution_kernal& setFunc_L(
			FuncTypeL func)
	{
		_func_L = func;
		return *this;
	}
	AdjustmentSolution_kernal& setFunc_Buffer(
			FuncTypeBuffer func)
	{
		_func_buff = func;
		return *this;
	}
	AdjustmentSolution_kernal& setInitalValue(
		const TypeParas& paras)
	{
		_paras = paras;
		return *this;
	}
	AdjustmentSolution_kernal& data(
			TypeParas& paras,
			double& sigma_util,
			TypeParas& sigmas,
			std::vector<TypeV>& Vs)
	{
		paras = _paras;
		sigma_util = _sigma_util;
		sigmas = _sigmas;
		Vs = _Vs;
	}
private:
	bool isInThresh(
			const TypeParas &delta)
	{
		using std::fabs;
		if (
			(fabs(delta(4)) > _thresh) || 
			(fabs(delta(5)) > _thresh) || 
			(fabs(delta(6)) > _thresh))
		{
			return false;
		}
		return true;
	}

	//====================
	FuncTypeA _func_A;
	FuncTypeL _func_L;
	FuncTypeBuffer _func_buff;

	double _thresh=1 * 10e-6;

	std::size_t _max_num_iter = 10;

	TypeParas _paras;
	TypeParas _sigmas;
	BufferType _buffer_data;
	double _sigma_util;
	std::vector<TypeV> _Vs;

	std::vector<UntiType> _untis;
};


#endif

