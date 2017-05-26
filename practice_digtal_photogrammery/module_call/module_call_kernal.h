#ifndef _MODULE_CALL_KERNAL_H_HSJ_
#define _MODULE_CALL_KERNAL_H_HSJ_ 

#include <iostream>
#include <utility>
#include <time.h>

template <typename Func,
		  typename... Args>
void moduleCall_kernal(const std::string& message,
				std::ostream& out,
				Func functor,
				Args&&... args)
{
	out<<"================"<<std::endl;
	out<<message<<std::endl;
	auto start = std::clock();
	functor(std::forward<Args>(args)...);
	auto end = std::clock();
	out<<"END"<<"(total:"<<(end - start)<<")"<<std::endl;
}

#endif