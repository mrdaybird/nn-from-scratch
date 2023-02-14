#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <mlpack.hpp>
#include <memory>
#include <queue>
#include "backpropv3.h"

using namespace mlpack;
using namespace arma;
using namespace std;

using mat = arma::mat;

class Function{
	public:
		virtual shared_ptr<Variable> calc() = 0;
		virtual void repr() = 0;
		virtual std::vector<std::shared_ptr<Variable>> const& get_inputs() const = 0;
		virtual std::shared_ptr<Variable> const& get_out() const = 0;
		virtual void calc_grad() = 0;
};


class Variable{
	public:
		Variable(){
		}
		Variable(mat const& in, bool requires_grad = false) : _value(in), _require_grad(requires_grad){
			_grad.set_size(_value.n_rows, _value.n_cols);
			_grad.zeros();
		}
		/*
		Variable(Variable const& other) : _value(other.m){
			_value = other.m();
			_grad = other.grad();
			_require_grad = other.requires_grad();
		}
		*/
		void set_m(mat const& m){
			_value = m;
		}
		mat const& m() const{
			return _value;
		}
		mat& grad(){
			return _grad;
		}
		bool requires_grad() const{
			return _require_grad;
		}
		void set_func(std::shared_ptr<Function> func){
			_func = func;
		}
		auto const& get_func() const{
			return _func;
		}
		void add_consumer(shared_ptr<Variable> const& v){
			_consumers.emplace_back(v);
		}
		
		~Variable(){
			cerr << "Variable Destructor called." << endl;
		}
		void set_visited(bool v){
			visited = v;
		}
		bool is_visited(){
			return visited;
		}
		auto const& get_consumers(){
			return _consumers;
		}
	private:
		mat _value;
		mat _grad;
		bool _require_grad = false;
        std::shared_ptr<Function> _func;
		bool visited = false;
		vector<shared_ptr<Variable>> _consumers;
};

class SumBP : public Function{
	public:
		SumBP(shared_ptr<Variable> const& x1, shared_ptr<Variable> const& x2){
			_in.emplace_back(x1);
			_in.emplace_back(x2);
		}
		void repr(){
			cout << "Sum" << ' ';
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat ym = _x1.m() + _x2.m();
			bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
			_out = make_shared<Variable>(ym, requires_grad);
			// _out->_func = make_shared<SumBP>(*this);
			return _out;
		}

		void calc_grad(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat m = ones<mat>(_x1.m().n_rows, _x2.m().n_cols);
			_x1.grad() += m%_out->grad();
			_x2.grad() += m%_out->grad();
		}

		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
		~SumBP(){
			cerr << "SumBP destructor called" << endl;
		}
		// auto const& inputs() const{
		// 	return _in;
		// }
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		std::shared_ptr<Variable> _out;
};

class Multiply : public Function{
	public:
		Multiply(shared_ptr<Variable> const& x1, shared_ptr<Variable> const& x2){
			_in.emplace_back(x1);
			_in.emplace_back(x2);
		}
		void repr(){
			cout << "Multiply" << ' ';
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat ym = _x1.m();
			ym %=_x2.m();
			bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
			_out = make_shared<Variable>(ym, requires_grad);

			return _out;
		}
		void calc_grad(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];

			_x1.grad() += _x2.m()%_out->grad();
			_x2.grad() += _x1.m()%_out->grad();
		}
		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		std::shared_ptr<Variable> _out;
};

class Subtract : public Function{
	public:
		Subtract(shared_ptr<Variable> const& x1, shared_ptr<Variable> const& x2){
			_in.emplace_back(x1);
			_in.emplace_back(x2);
		}
		void repr(){
			cout << "Subtract";
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat ym = _x1.m() - _x2.m();
			bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
			_out = make_shared<Variable>(ym, requires_grad);
			// _out->_func = make_shared<SumBP>(*this);
			return _out;
		}
		void calc_grad(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat m = ones<mat>(_x1.m().n_rows, _x1.m().n_cols);
			_x1.grad() += m%_out->grad();
			_x2.grad() -= m%_out->grad();
		}
		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		std::shared_ptr<Variable> _out;
};
class Power : public Function{
	public:
		Power(shared_ptr<Variable> const& x1, int n) : _n(n){
			_in.emplace_back(x1);
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			mat ym = arma::pow(_x1.m(), _n);
			_out = make_shared<Variable>(ym, _x1.requires_grad());
			return _out;
		}
		void repr(){
			cout << "Power" << ' ';
		}
		void calc_grad(){
			auto& _x1 = *_in[0];
			mat g = arma::pow(_x1.m(), _n - 1);
			g = _n*g;
			_x1.grad() += g%_out->grad();
		}
		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		int _n;
		shared_ptr<Variable> _out;
};
class Exp : public Function{
	public:
		Exp(shared_ptr<Variable> const& x1){
			_in.emplace_back(x1);
		}
		void repr(){
			cout << "Exp" << ' ';
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			mat ym = exp(_x1.m());
			_out = make_shared<Variable>(ym, _x1.requires_grad());
			return _out;
		}	
		void calc_grad(){
			auto& _x1 = *_in[0];
			_x1.grad() += _out->m()%_out->grad();
		}
		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		shared_ptr<Variable> _out;
};

class MatMul : public Function{
	public:
		MatMul(shared_ptr<Variable> const& x1, shared_ptr<Variable> const& x2){
			_in.emplace_back(x1);
			_in.emplace_back(x2);
		}
		void repr(){
			cout << "MatMul" << ' ';
		}
		shared_ptr<Variable> calc(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat ym = _x1.m() * _x2.m();
			_out = make_shared<Variable>(ym, _x1.requires_grad());
			return _out;
		}
		void calc_grad(){
			auto& _x1 = *_in[0];
			auto& _x2 = *_in[1];
			mat const& grad = _out->grad();

			_x1.grad() += grad * _x2.m().t();
			_x2.grad() += _x1.m().t() * grad;
 		}
		std::vector<shared_ptr<Variable>> const& get_inputs() const{
			return _in;
		}
		std::shared_ptr<Variable> const& get_out() const{
			return _out;
		}
	private:
		std::vector<std::shared_ptr<Variable>> _in;
		shared_ptr<Variable> _out;
};

inline auto operator*(shared_ptr<Variable> const& x, shared_ptr<Variable> const& y){
	// SumBP s = new SumBP(x, y)
	auto m = make_shared<MatMul>(x, y);
	auto z = m->calc();
	z->set_func(m);
	return z;
}

inline auto operator+(shared_ptr<Variable> const& x, shared_ptr<Variable> const& y){
	// SumBP s = new SumBP(x, y)
	auto s = make_shared<SumBP>(x, y);
	auto z = s->calc();
	z->set_func(s);
	// cout << s->get_inputs().size() << endl;
	return z;
}

inline auto operator+(shared_ptr<Variable> const& x, double scalar){
	auto const& m = x->m();
	mat sm = ones<mat>(m.n_rows, m.n_cols);
	sm *= scalar;
	auto sv = make_shared<Variable>(sm, false);
	return sv + x;
}
inline auto operator+(double scalar, shared_ptr<Variable> const& x){
	return x + scalar;
}

inline auto operator-(shared_ptr<Variable> const& x, shared_ptr<Variable> const& y){
	// SumBP s = new SumBP(x, y)
	auto s = make_shared<Subtract>(x, y);
	auto z = s->calc();
	z->set_func(s);
	// cout << s->get_inputs().size() << endl;
	return z;
}
inline auto operator-(shared_ptr<Variable> const& x, double scalar){
	auto const& m = x->m();
	mat sm = ones<mat>(m.n_rows, m.n_cols);
	sm *= scalar;
	auto sv = make_shared<Variable>(sm, false);

	return x - sv;
}
inline auto operator-(double scalar, shared_ptr<Variable> const& x){
	auto const& m = x->m();
	mat sm = ones<mat>(m.n_rows, m.n_cols);
	sm *= scalar;
	auto sv = make_shared<Variable>(sm, false);

	return sv - x;
}

inline auto operator-(shared_ptr<Variable> const& x){
	auto& xm = x->m();
	mat ym = zeros<mat>(xm.n_rows, xm.n_cols);
	auto y = make_shared<Variable>(ym, false);
	return (y - x);
}

inline auto operator%(shared_ptr<Variable> const& x, shared_ptr<Variable> const& y){
	auto mut = make_shared<Multiply>(x, y);
	auto z = mut->calc();
	z->set_func(mut);
	return z;
}

inline auto exp(shared_ptr<Variable> const& x){
	auto e = make_shared<Exp>(x);
	auto z = e->calc();
	z->set_func(e);

	return z;
}
inline auto pow(shared_ptr<Variable> const& x, int pow){
	auto p = make_shared<Power>(x, pow);
	auto z = p->calc();
	z->set_func(p);

	return z;
}

void display(shared_ptr<Variable> const& x){
	cout << arma::size(x->m()) << "->";
	auto const& func = x->get_func();
	if(func != nullptr){
		func->repr();
		auto& inputs = func->get_inputs();
		// for(auto const& i : inputs){
		// 	cout << arma::size(i->m()) << ' ';
		// }
		// cout << endl;
		// cout << arma::size(inputs[0]->m()) << endl;
		cout << inputs.size() << endl;
		for(auto const& i : inputs){
			display(i);
		}
	}
	cout << endl;
}

void find_consumers(shared_ptr<Variable> const& x){
	auto& func = x->get_func();
	if(func != nullptr){
		auto const& inputs = func->get_inputs();
		for(auto const& i : inputs){
			i->add_consumer(x);
			find_consumers(i);
		}
	}
}



/*
	Sigmoid Graph:
	e^(-x)*((1+e^(-x))^-1)
	|	\
	|	(1+e^(-x))^-1
	|	|
	|  1+e^(-x)
	|  /	\
	e^(-x)  1
	|
	-x
	/ \
	0 x
*/

void build_grad(shared_ptr<Variable> const& x){
	if(!x->is_visited()){
		auto const& consumers = x->get_consumers();
		for(auto const& i : consumers){
			build_grad(i);
		}
		x->set_visited(1);
		auto const& func = x->get_func();
		if(func != nullptr)
			func->calc_grad();
	}
}
void backward(shared_ptr<Variable> const& z, vector<shared_ptr<Variable>> vx){
	find_consumers(z);

	auto const& m = z->m();
	auto& grad = z->grad();
	grad = ones<mat>(m.n_rows, m.n_cols);
	// z->set_visited();
	for(auto const& x : vx){
		build_grad(x);
	}
}
// void backprop(shared_ptr<Variable> const& x){
// 	queue<shared_ptr<Variable>> q;
// 	q.push(x);
// 	x->set_visited(true);
// 	while(!q.empty()){
// 		auto const& top = q.front();
// 		q.pop();
// 		auto const& func = top->get_func();
// 		if(func != nullptr){
// 			func->calc_grad();
// 			auto const& inputs = func->get_inputs();
// 			for(auto const& i : inputs){
// 				if(!i->is_visited()){
// 					q.push(i);
// 					i->set_visited(true);
// 				}
// 			}
// 		}
// 	}
// }


// void backward(shared_ptr<Variable> const& x){
// 	auto const& m = x->m();
// 	auto& grad = x->grad();
// 	grad = ones<mat>(m.n_rows, m.n_cols);
// 	backprop(x);
// }

int main(){
	// -0.687126 0.789846 0.201126 0.0948613 -0.0550408
	mat xm(2, 2, arma::fill::randn);
	cout << endl;
	auto x = make_shared<Variable>(xm, true);
	cout << xm << endl;
	// -2.13557 0.283815 0.454843 -1.68281 0.137096
	mat ym(2, 2, arma::fill::randn);
	cout << endl;
	auto y = make_shared<Variable>(ym, true);
	cout << ym << endl;

	// auto x1 = exp(-x);
	// cout << x1->m() << endl;
	// auto ones_v = make_shared<Variable>(ones<mat>(1,5), false);
	// auto z = x1%pow(ones_v+x1, -1);c
	auto c = x * y;
	auto z = exp(-c)%pow(1.0 + exp(-c), -1);
	cout << "z:\n";
	cout << z->m() << endl;

	cout << "The computational graph is as follows:" << endl;
	display(z);
	cout << "Complete." << endl;

	backward(z, {x});
	cout << "z.grad()\n" << z->grad() << endl;
	cout << "x.grad()\n" << x->grad() << endl;
	cout << "y.grad()\n" << y->grad() << endl;
	// cout << "x1.grad()\n" << x1->grad() << endl;
}