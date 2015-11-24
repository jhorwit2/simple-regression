# simple-regression


Ordinary Least Squares Linear Regression implementation in Go

This implementation is a direct port of the C++ implementation found [here](from http://www.johndcook.com/running_regression.html)

## Install

`go get github.com/jhorwit2/simple-regression`

## Usage

```
r := linear.NewRegression()
r.Push(0, 0)
r.Push(1, 5)
r.Push(2, 10)
y := r.Predict(5)
```
