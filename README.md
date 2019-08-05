# Corporaci-n-Favorita-Grocery-Sales-Forecasting
Predict sales for grocery stores in a horizontal period window of 16 days

![Alt text](https://images.unsplash.com/photo-1506617564039-2f3b650b7010?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80)

# Overview
Brick-and-mortar grocery stores are always in a delicate dance with purchasing and sales forecasting. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

The problem becomes more complex as retailers add new locations with unique needs, new products, ever transitioning seasonal tastes, and unpredictable product marketing. Corporación Favorita, a large Ecuadorian-based grocery retailer, knows this all too well. They operate hundreds of supermarkets, with over 200,000 different products on their shelves.

Corporación Favorita has challenged to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They’re excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

# Problem Reframe
The prediction of items sales in this task lie within a 16-day window (2018-08-16 to 2018-08-31), given a more sufficient historical sales. The information that can be used for building ML models includes sotre category, items characteristics and sales days. In my working, I reframed this problem as a 1-day prediction out of 16 days each time, and the training data was segmentated at similar weekday with the prediction dates. The training primarily relied on 6 series of this reframed daily sales.

# Models
•	Interaction effects among products, store types and store location

•	Seasonality or periodicity effects on sales

•	Promotion effects on sales

1.	16-day sales forecasting with series of 16 1-day prediction models
2.	16-day sales forecasting with one sequence model

















