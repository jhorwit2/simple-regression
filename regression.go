package linear

import "github.com/jhorwit2/running-statistics"

// The linear regression code is from http://www.johndcook.com/running_regression.html

// Regression structure
type Regression struct {
	xStats runningstats.RunningStats
	yStats runningstats.RunningStats
	sumXY  float64
	n      int
}

// NewRegression returns a new least squares linear regression
func NewRegression() *Regression {
	return &Regression{}
}

// PushAll puts all the data from the two arrays into the regression
func (r *Regression) PushAll(xArray, yArray []float64) {
	for i, x := range xArray {
		y := yArray[i]
		r.Push(x, y)
	}
}

// Push adds data to the running regression
func (r *Regression) Push(x, y float64) {
	r.sumXY += (r.xStats.Mean() - x) * (r.yStats.Mean() - y) * float64(r.n) / float64(r.n+1)

	r.xStats.Add(x)
	r.yStats.Add(y)
	r.n++
}

// Len returns the number of observations in the linear regression
func (r *Regression) Len() int {
	return r.n
}

// Slope returns the m in y = m*x + b
func (r *Regression) Slope() float64 {
	sumXX := r.xStats.Var() * float64(r.n-1)
	return r.sumXY / sumXX
}

// Intercept returns the b in y = m*x + b
func (r *Regression) Intercept() float64 {
	return r.yStats.Mean() - r.Slope()*r.xStats.Mean()
}

// Coefficients returns the slope and intercept
func (r *Regression) Coefficients() (float64, float64) {
	return r.Slope(), r.Intercept()
}

// Predict takes the passed in x value and returns the predicted value based
// on the equation y = m*x + b where y is the value to be predicted based on x.
func (r *Regression) Predict(x float64) float64 {
	return r.Slope()*x + r.Intercept()
}

// Correlation returns the correlation for the regression
func (r *Regression) Correlation() float64 {
	t := r.xStats.Stddev() * r.yStats.Stddev()
	return r.sumXY / (float64(r.n-1) * t)
}
