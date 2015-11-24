package anomaly

import "github.com/jhorwit2/go-running-statistics"

// The linear regression code is from http://www.johndcook.com/running_regression.html

// LeastSquaresRegression structure
type LeastSquaresRegression struct {
	xStats runningstats.RunningStats
	yStats runningstats.RunningStats
	sumXY  float64
	n      int
}

// NewLeastSquaresRegression returns a new least squares linear regression
func NewLeastSquaresRegression() *LeastSquaresRegression {
	return &LeastSquaresRegression{}
}

// AddAll puts all the data from the two arrays into the regression
func (r *LeastSquaresRegression) AddAll(xArray, yArray []float64) {
	for i, x := range xArray {
		y := yArray[i]
		r.Push(x, y)
	}
}

// Push adds data to the running regression
func (r *LeastSquaresRegression) Push(x, y float64) {
	r.sumXY += (r.xStats.Mean() - x) * (r.yStats.Mean() - y) * float64(r.n) / float64(r.n+1)

	r.xStats.Add(x)
	r.yStats.Add(y)
	r.n++
}

// Len returns the number of observations in the linear regression
func (r *LeastSquaresRegression) Len() int {
	return r.n
}

// Slope returns the m in y = m*x + b
func (r *LeastSquaresRegression) Slope() float64 {
	sumXX := r.xStats.Var() * float64(r.n-1)
	return r.sumXY / sumXX
}

// Intercept returns the b in y = m*x + b
func (r *LeastSquaresRegression) Intercept() float64 {
	return r.yStats.Mean() - r.Slope()*r.xStats.Mean()
}

// Coefficients returns the slope and intercept
func (r *LeastSquaresRegression) Coefficients() (float64, float64) {
	return r.Slope(), r.Intercept()
}

// Correlation returns the correlation for the regression
func (r *LeastSquaresRegression) Correlation() float64 {
	t := r.xStats.Stddev() * r.yStats.Stddev()
	return r.sumXY / (float64(r.n-1) * t)
}
