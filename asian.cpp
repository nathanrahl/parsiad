#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <tuple>

////////////////////////////////////////////////////////////////////////////////

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;
typedef Eigen::BiCGSTAB<matrix, Eigen::IncompleteLUT<double>> solver;

////////////////////////////////////////////////////////////////////////////////

/**
 * A struct corresponding to a parameterization of the problem.
 */
struct parameters {
	const double T;
	const double w_ubar;
	const double w_obar;
	const double x0;
	const double x1;
	double (*const f)(double);

	parameters(
		double T,
		double w_ubar, double w_obar,
		double x0, double x1,
		double (*f)(double)
	) : T(T), w_ubar(w_ubar), w_obar(w_obar), x0(x0), x1(x1), f(f) {}

	// Delete copy constructor and assignment operator
	parameters(const parameters&) = delete;
	parameters &operator=(const parameters&) = delete;
};

class approximation {
	const parameters &p;

	const int N;
	const int M1;
	const int M2;
	const int M;

	const double dt;
	const double dxi;
	const double db;

	solver s;

	double abs_tol;
	double rel_tol;

	vector xi_d;
	vector b_d;

	std::deque<vector> u_vecs;
	std::deque<vector> w_vecs;
	std::deque<int> policy_its;

	// Build left-hand side matrix and right-hand side vector
	std::tuple<matrix, vector> get_lhs_rhs(
		int n,
		const vector &w_vec,
		const vector &u_vec
	) {
		matrix lhs = matrix(M, M);
		vector rhs = vector::Zero(M);
		lhs.reserve(5 * M); // At most 5 zeros per row

		int i = 0;
		const int e1 = M2 + 1;
		const int e2 = 1;

		const double t = n * dt;

		// i1 = 0
		for(int i2 = 0; i2 <= M2; ++i2) {
			lhs.insert(i, i) = 1.;
			rhs[i++] = p.f(t * b_d[i2] + (p.T - t) * p.x0);
		}

		// 0 < i1 < M1
		for(int i1 = 1; i1 < M1; ++i1) {
			const double xi = xi_d[i1];

			// 0 <= i2 <= M2
			// TODO: Explicitly optimize so there is no branching here
			for(int i2 = 0; i2 <= M2; ++i2) {
				const double w_i = w_vec[i];
				const double u_i = u_vec[i];
				const double k_i = p.x0 * (1. - xi) + p.x1 * xi - b_d[i2];
				const double uwind = (k_i > 0.) ? 1. : 0.;
				const double dwind = (k_i < 0.) ? 1. : 0.;
				const double tmp_1 = (1. - w_i) * std::fabs(k_i) / t / db * dt;
				const double tmp_2 = w_i / (dxi * dxi) * dt;
				if(i2 >  0) { lhs.insert(i, i - e2) = - dwind * tmp_1; }
				lhs.insert(i, i - e1) = - tmp_2 / 2.;
				lhs.insert(i, i     ) = tmp_1 + tmp_2 + (1. - w_i);
				lhs.insert(i, i + e1) = - tmp_2 / 2.;
				if(i2 < M2) { lhs.insert(i, i + e2) = - uwind * tmp_1; }
				rhs[i++] = (1. - w_i) * u_i;
			}
		}

		// i1 = M1
		for(int i2 = 0; i2 <= M2; ++i2) {
			lhs.insert(i, i) = 1.;
			rhs[i++] = p.f(t * b_d[i2] + (p.T - t) * p.x1);
		}

		return std::make_tuple(lhs, rhs);
	}

	// One policy iteration
	std::tuple<vector, vector> policy_iteration(
		int n,
		const vector &u_vec
	) {
		const double ws[] = {p.w_ubar, p.w_obar};

		int l = 0;
		vector u_curr = u_vec;
		while(true) {
			// Policy improvement
			vector w_opt = vector::Zero(M, 1);
			vector best = -std::numeric_limits<double>::infinity()
			            * vector::Ones(M, 1);
			for(int k = 0; k < 2; ++k) {
				vector w_vec = ws[k] * vector::Ones(M, 1);
				auto tmp = get_lhs_rhs(n, w_vec, u_vec);
				vector candidate = std::get<0>(tmp) * (-u_curr)
				                 + std::get<1>(tmp);
				for(int i = 0; i < M; ++i) {
					if(candidate(i) > best(i)) {
						best(i) = candidate(i);
						w_opt(i) = ws[k];
					}
				}
			}

			// Policy evaluation
			auto tmp = get_lhs_rhs(n, w_opt, u_vec);
			s.compute(std::get<0>(tmp));
			vector u = s.solveWithGuess(std::get<1>(tmp), u_curr);

			// Break if error is small enough
			const double err = (u - u_curr).cwiseAbs().maxCoeff();
			const double rel = u.cwiseAbs().maxCoeff();
			if(err < abs_tol + rel_tol * rel) {
				policy_its.push_front(l);
				return std::make_tuple(u, w_opt);
			}

			u_curr = u;
			++l;
		}
	}

	// Solve the problem
	void solve() {
		// Initial condition
		vector u = vector::Zero(M);
		int i = 0;
		for(int i1 = 0; i1 <= M1; ++i1) {
			for(int i2 = 0; i2 <= M2; ++i2) {
				u[i++] = p.f(p.T * b_d[i2]);
			}
		}
		u_vecs.push_front(u);
		w_vecs.push_front(vector::Zero(M));

		// Timestep from n=N-1 to n=1
		for(int n = N-1; n >= 1; --n) {
			auto tmp = policy_iteration(n, u);
			u_vecs.push_front(std::get<0>(tmp));
			w_vecs.push_front(std::get<1>(tmp));
		}
	}

public:

	approximation(
		const parameters &p,
		int N, int M1, int M2,
		double abs_tol=1e-12, double rel_tol=1e-6
	) : p(p), N(N), M1(M1), M2(M2), M((M1 + 1) * (M2 + 1)),
	    dt(p.T / N), dxi(1. / M1), db((p.x1 - p.x0) / M2),
	    abs_tol(abs_tol), rel_tol(rel_tol) {
		xi_d = vector::LinSpaced(M1+1, 0., 1.);
		b_d = vector::LinSpaced(M2+1, p.x0, p.x1);
		solve();
	}

	// Delete copy constructor and assignment operator
	approximation(const approximation&) = delete;
	approximation &operator=(const approximation&) = delete;

	const std::deque<vector> &get_solution_vectors() { return u_vecs; }
	const std::deque<vector> &get_control_vectors() { return w_vecs; }
};

////////////////////////////////////////////////////////////////////////////////

int main() {

	// European call parameters
	const double T = 1.;
	const double sigma_ubar = 0.1;
	const double sigma_obar = 0.3;
	const double x0 = 0.;
	const double x1 = 1.;
	const auto f = [] (double a) -> double {
		return std::max(a - 0.5, 0.);
	};

	// Discretization parameters
	const int N  = 50; // Number of timesteps
	const int M1 = 50; // Number of points in xi axis
	const int M2 = 50; // Number of points in b = a/t axis

	// Prepare parameters struct
	const double sigma_ubar2 = sigma_ubar * sigma_ubar;
	const double sigma_obar2 = sigma_obar * sigma_obar;
	const double delta2 = (x1 - x0) * (x1 - x0);
	const double w_ubar = (sigma_ubar2) / (delta2 - sigma_ubar2);
	const double w_obar = (sigma_obar2) / (delta2 - sigma_obar2);
	parameters p(T, w_ubar, w_obar, x0, x1, f);

	// Approximate
	approximation u_approx(p, N, M1, M2);
	const std::deque<vector> &sols = u_approx.get_solution_vectors();
	std::cout << sols[0] << std::endl;

	return 0;

}
