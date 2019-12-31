import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression


"""Create the NHL standings database title 'standings' as of 5:30 PM EST on 12/20/19"""

standings = pd.DataFrame([["Captials", 35, 24, 6, 5, 53, 125, 100],
                         ["Bruins", 36, 21, 7, 8, 50, 120, 93],
                         ["Blues", 36, 22, 8, 6, 50, 109, 96],
                         ["Islanders", 33, 23, 8, 2, 48, 98, 82],
                         ["Avalanche", 35, 22, 10, 3, 47, 124, 95],
                         ["Hurricans", 35, 22, 11, 2, 46, 116, 90],
                         ["Penguins", 34, 20, 10, 4, 44, 114, 90],
                         ["Stars", 36, 20, 12, 4, 44, 95, 85],
                         ["Coyotes", 37, 20, 13, 4, 44, 105, 94],
                         ["Golden Knights", 38, 19, 13, 6, 44, 116, 110],
                         ["Flyers", 35, 19, 11, 5, 43, 111, 101],
                         ["Jets", 35, 20, 13, 2, 42, 105, 101],
                         ["Oilers", 37, 19, 14, 4, 42, 109, 112],
                         ["Flames", 37, 18, 14, 5, 41, 99, 112],
                         ["Canadiens", 35, 17, 12, 6, 40, 112, 111],
                         ["Wild", 36, 17, 14, 5, 39, 115, 120],
                         ["Sabres", 36, 16, 13, 7, 39, 111, 114],
                         ["Lightning", 33, 17, 12, 4, 38, 117, 107],
                         ["Predators", 34, 16, 12, 6, 38, 119, 111],
                         ["Maple Leafs", 35, 17, 14, 4, 38, 115, 112],
                         ["Canucks", 36, 17, 15, 4, 38, 116, 112],
                         ["Panthers", 33, 16, 12, 5, 37, 115, 109],
                         ["Rangers", 33, 16, 13, 4, 36, 105, 106],
                         ["Blue Jackets", 35, 15, 14, 6, 36, 90, 103],
                         ["Sharks", 36, 16, 18, 2, 34, 98, 125],
                         ["Blackhawks", 36, 14, 16, 6, 34, 99, 115],
                         ["Kings", 37, 15, 18, 4, 34, 96, 117],
                         ["Senators", 36, 15, 18, 3, 33, 99, 117],
                         ["Ducks", 35, 14, 17, 4, 32, 89, 104],
                         ["Devils", 33, 11, 17, 5, 27, 80, 116],
                         ["Red Wings", 36, 9, 24, 3, 21, 79, 141]], columns=('Team', 'Games Played', 'Wins', 'Losses',
                                                                             'OT Loss', 'Points', 'Goals For',
                                                                             'Goals Against'))

"""Looking at the first 5 rows of the standings DataFrame"""
#print(standings.head()

"""Looking at the standard stats for all 31 teams combined"""
#print(standings.describe())

"""Let's start by defining x and y variables. In the first case we want to predict points based on goals for 
other variables."""
x = standings['Goals For']
y = standings["Points"]

"""Creating an initial scatter plot to show points depending on goals for"""
plt.scatter(x, y, label=standings['Team'])
plt.xlabel("Goals For")
plt.ylabel("Points")
plt.axis([0, 200, 20, 100])
plt.legend()
#plt.show()



# Converting x into a 2D shape
x_matrix = x.values.reshape(-1, 1)

"""Create a regression model for points based on goals for"""
reg = LinearRegression()
reg.fit(x_matrix, y)

"""Calculating the p-value for Goals For"""
f_regression(x_matrix, y)
p_value = f_regression(x_matrix, y)[1]

"""Creating a Summary table for Goals For"""

reg_summary = pd.DataFrame(["Goals For"], columns=["Features"])
reg_summary["Coefficient"] = reg.coef_
reg_summary["Intercept"] = reg.intercept_
reg_summary["P-Value"] = p_value

# Creating y_hat variable for Goals For
gf_y_hat = 0.395 * standings["Goals For"] - 2.270

"""Plotting the regressing line on the scatter plot"""
plt.scatter(x, y, c='blue')
plt.plot(x_matrix, gf_y_hat, lw=4, c="red")
plt.xlabel("Goals For")
plt.ylabel("Points")
plt.axis([0, 200, 0, 100])
plt.legend()
#plt.show()

"""Predicting points based on if we scored 200 goals"""

#print(reg.predict([[200]]))
"""The conclusion here is if we scored 200 goals, we would end up with 77 points."""

"""Adding additional variables to our regression"""
estimators = ['Wins', 'Goals For', 'Goals Against']

x2 = standings[estimators]
y2 = standings['Points']


"""Creating new regression with additional variables"""
reg2 = LinearRegression()
reg2.fit(x2, y2)

"""Checking each variables p-value"""
f_stat2 = f_regression(x2, y2)
p_value_2 = f_stat2[1]


"""Testing our model to predict how many points we would get based on wins, goals for, and goals against"""

reg2.predict([[30, 500, 400]])


"""Based on the input above if we had 30 wins, but scored 500 goals for and had 400 goals against we would
have a total of 77 points. Basically this tells us that we would end up with 17 Overtime losses!"""



"""Creating a summary table of the second regression, including coefficients, p-values, and each variables f-stat."""
new_df = pd.DataFrame(data=x2.columns.values, columns=(['Features']))
new_df['Coefficients'] = reg2.coef_
new_df["p-value"] = p_value_2.round()
new_df['F-Stat'] = f_regression(x2, y2)[0]
#print(new_df)

"""Creating a while loop to collect user input to predict points based off of user input data."""

while True:
    wins = input("\nPlease enter total wins\n"
                       "(You can press 'q' at anytime to quit)")
    if wins == 'q':
        break
    goals_for = input("\nPlease enter total goals for\n"
                        "(You can press 'q' at anytime to quit)")
    if goals_for == 'q':
        break
    goals_against = input("\nPlease enter total goals against\n"
                        "(You can press 'q' at anytime to quit)")
    if goals_against == 'q':
        break


    p_df = pd.DataFrame([[wins, goals_for, goals_against]])
    reg2.fit(x2, y2)
    print(reg2.predict(p_df))

