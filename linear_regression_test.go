package anomaly

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLinearRegression(t *testing.T) {
	// http://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Norris.dat
	data := [][]float64{{0.1, 0.2}, {338.8, 337.4}, {118.1, 118.2},
		{888.0, 884.6}, {9.2, 10.1}, {228.1, 226.5}, {668.5, 666.3}, {998.5, 996.3},
		{449.1, 448.6}, {778.9, 777.0}, {559.2, 558.2}, {0.3, 0.4}, {0.1, 0.6}, {778.1, 775.5},
		{668.8, 666.9}, {339.3, 338.0}, {448.9, 447.5}, {10.8, 11.6}, {557.7, 556.0},
		{228.3, 228.1}, {998.0, 995.8}, {888.8, 887.6}, {119.6, 120.2}, {0.3, 0.3},
		{0.6, 0.3}, {557.6, 556.8}, {339.3, 339.1}, {888.0, 887.2}, {998.5, 999.0},
		{778.9, 779.0}, {10.2, 11.1}, {117.6, 118.3}, {228.9, 229.2}, {668.4, 669.1},
		{449.2, 448.9}, {0.2, 0.5}}

	var xArray []float64
	var yArray []float64

	for _, values := range data {
		xArray = append(xArray, values[1])
		yArray = append(yArray, values[0])
	}

	regression := NewLeastSquaresRegression()
	regression.AddAll(xArray, yArray)
	m, c := regression.Coefficients()

	assert := assert.New(t)
	assert.InDelta(m, 1.0021168180204547, 10E-12, "slope is not equal to spec")
	assert.InDelta(c, -0.262323073774029, 10E-12, "intercept is not equal to spec within reason")
}
