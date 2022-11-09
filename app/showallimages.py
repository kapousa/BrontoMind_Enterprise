import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# print(df.columns)

# Show ranges of selected columns (distribution of the continuous variables, other columns values are 1/0)
df[['age', 'creatinine_phosphokinase',
    'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']].hist(bins=20,
                                                                                figsize=(15, 15))
plt.show()


# This column has 0 and 1 values. I will change it to ‘yes’ and ‘no’.
# Also want to change the ‘sex’ column and replace 0 and 1 with ‘male’ and ‘female’.
def replace_01vlues():
    df['sex1'] = df['sex'].replace({1: "Male", 0: "Female"})
    df['death'] = df['DEATH_EVENT'].replace({1: "yes", 0: "no"})


# Here, the red color shows the death event and the green color represents no death.
# This plot shows how each of these variables is segregated between death events.
def view_segregated_affect():
    sns.pairplot(df[["creatinine_phosphokinase", "ejection_fraction",
                     "platelets", "serum_creatinine",
                     "serum_sodium", "time", "death"]], hue="death",
                 diag_kind='kde', kind='scatter', palette='husl')
    plt.show()


# Show mean of each continuous variable for death events
# Some numeric data of the mean of each continuous variable for death events
# and no death events is necessary for a good report
def show_means_of_continuous_variables():
    continous_var = ['age', 'creatinine_phosphokinase',
                     'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    plt.figure(figsize=(16, 25))
    for i, col in enumerate(continous_var):
        plt.subplot(6, 4, i * 2 + 1)
        plt.subplots_adjust(hspace=.25, wspace=.3)

        plt.grid(True)
        plt.title(col)
        sns.kdeplot(df.loc[df["death"] == 'no', col], label="alive", color="green", shade=True, kernel='gau', cut=0)
        sns.kdeplot(df.loc[df["death"] == 'yes', col], label="dead", color="red", shade=True, kernel='gau', cut=0)
        plt.subplot(6, 4, i * 2 + 2)
        sns.boxplot(y=col, data=df, x="death", palette=["green", "red"])
    plt.show()


# we have five other categorical variables in this dataset.
# It is worth examining their relationship with the ‘death’ variable.
# I will use barplot or in the seaborn library, it is called the ‘countplot’ to do that.
def show_three_values_relationship():
    binary_var = ['anaemia', 'diabetes', 'high_blood_pressure',
                  'sex1', 'smoking']
    plt.figure(figsize=(13, 9))
    for i, var in enumerate(binary_var):
        plt.subplot(2, 3, i + 1)
        plt.title(var, fontsize=14)
        plt.xlabel(var, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        sns.countplot(data=df, x=var, hue="death", palette=['gray', "coral"])
    plt.show()


# Shows the distribution of ‘time’ across smoking and no smoking males and females.
def show_distribution():
    plt.figure(figsize=(8, 6))
    a = sns.violinplot(df.smoking, df.time, hue=df.sex1, split=True)
    plt.title("Smoking vs Time Segregated by Gender", fontsize=14)
    plt.xlabel("Smoking", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    plt.show()


# See the relation between ‘ejection_fraction’ and ‘time’ segregated by ‘death’.
def show_relationships_with3():
    sns.lmplot(x="ejection_fraction", y="time",
               hue="death", data=df, scatter_kws=dict(s=40, linewidths=0.7,
                                                      edgecolors='black'))
    plt.xlabel("Ejection Fraction", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    plt.title("Ejection fraction vs time segregated by death", fontsize=14)
    plt.show()


# See another comparison between the male and female population. How ‘time’ changes with ‘age’:
def show_change_between_values():
    fig = plt.figure(figsize=(20, 8), dpi=80)
    g = sns.lmplot(x='age', y='time',
                   data=df,
                   robust=True,
                   palette="Set1", col="sex1",
                   scatter_kws=dict(s=60, linewidths=0.7, edgecolors="black"))
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize='x-large')
        ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')
        ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
    plt.show()


# show the heat maps are used in feature selection for machine learning and also in data analytics
# to understand the correlation between the variables.
def show_heatmap():
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, linewidths=0.5, cmap="crest")
    plt.show()
