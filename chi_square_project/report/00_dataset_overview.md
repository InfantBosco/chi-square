# ACS Income Dataset Overview (CA 2018)

## Overview
The ACSIncome dataset is derived from the U.S. Census Bureau American Community Survey (ACS) Public Use Microdata Sample (PUMS) data for the year 2018 in the state of California (CA). It is commonly utilized in statistical parity and fairness research.

## Features Matrix
| Feature | Full Name | Description | Type |
|---|---|---|---|
| `AGEP` | Age | Person's age in years | Numeric |
| `COW` | Class of Worker | Employer type (private, government, self-employed) | Categorical |
| `SCHL` | Educational Attainment | Highest level of education completed | Categorical/Ordinal |
| `MAR` | Marital Status | Married, widowed, divorced, separated, never married | Categorical |
| `OCCP` | Occupation Code | Standard occupational classification code | Categorical |
| `POBP` | Place of Birth | US State or foreign country code | Categorical |
| `RELP` | Relationship | Relationship to reference person | Categorical |
| `WKHP` | Usual Hours Worked Per Week | Usual hours worked per week past 12 months | Numeric |
| `SEX` | Sex | Biological sex (1 = Male, 2 = Female) | Categorical |
| `RAC1P` | Race Code | Recoded detailed race code | Categorical |

## Target Variable
- `PINCP`: Total person's annual income (continuous numeric variable in USD).
