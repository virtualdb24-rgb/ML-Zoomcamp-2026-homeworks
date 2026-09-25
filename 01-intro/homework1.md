# Homework 1 - Introduction to Machine Learning

Source: `cohorts/2026/homework/01-intro/homework.md`
Submit: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01
Deadline: 28 September 2026, 23:00

## Setup

Install Python, NumPy, Pandas, Matplotlib, Seaborn (see `06-environment.md`).

## Getting the data

Dataset: `car_fuel_efficiency_2026.csv`

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

Then read it with Pandas.

## Q1. Pandas version

What version of Pandas did you install?

```python
pd.__version__
```

## Q2. Records count

How many records are in the dataset?

- 5000
- 9000
- 10000
- 15000

## Q3. Fuel types

How many fuel types are presented in the dataset?

- 1
- 2
- 3
- 4

## Q4. Missing values

How many columns in the dataset have missing values?

- 0
- 1
- 2
- 3
- 4

## Q5. Max fuel efficiency

What's the maximum fuel efficiency of cars from Asia?

- 21.2
- 31.2
- 41.2
- 51.2

## Q6. Median value of horsepower

1. Find the median value of the `horsepower` column in the dataset.
2. Next, calculate the most frequent value of the same `horsepower` column.
3. Use the `fillna` method to fill the missing values in the `horsepower`
   column with the most frequent value from the previous step.
4. Now, calculate the median value of `horsepower` once again.

Has it changed?

- Yes, it increased
- Yes, it decreased
- No

## Q7. Sum of weights

1. Select all the cars from Asia
2. Select only columns `vehicle_weight` and `model_year`
3. Select the first 7 values
4. Get the underlying NumPy array. Let's call it `X`.
5. Compute matrix-matrix multiplication between the transpose of `X` and `X`.
   To get the transpose, use `X.T`. Let's call the result `XTX`.
6. Invert `XTX`.
7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.
8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply
   the result by `y`. Call the result `w`.
9. What's the sum of all the elements of the result?

- 0.0369
- 0.369
- 3.69
- 36.9

> **Note**: You just implemented linear regression. We'll talk about it in the
> next lesson.

## Submit the results

- Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01
- The numerical options are calculated from the pinned 2026 release. Use the
  value that matches your calculation.
