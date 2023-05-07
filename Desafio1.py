{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "b5990956",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 2000 entries, 0 to 1999\n",
      "Data columns (total 9 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   x1      2000 non-null   float64\n",
      " 1   x2      2000 non-null   float64\n",
      " 2   x3      2000 non-null   object \n",
      " 3   x4      2000 non-null   object \n",
      " 4   x5      2000 non-null   int64  \n",
      " 5   x6      2000 non-null   int64  \n",
      " 6   x7      2000 non-null   int64  \n",
      " 7   x8      2000 non-null   float64\n",
      " 8   target  2000 non-null   int64  \n",
      "dtypes: float64(3), int64(4), object(2)\n",
      "memory usage: 140.8+ KB\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "data = pd.read_csv(\"dataset_Caso_1.csv\")\n",
    "data.head()\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "47b8c27a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>x1</th>\n",
       "      <th>x2</th>\n",
       "      <th>x5</th>\n",
       "      <th>x6</th>\n",
       "      <th>x7</th>\n",
       "      <th>x8</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.00000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>101.074885</td>\n",
       "      <td>-298.282145</td>\n",
       "      <td>0.01250</td>\n",
       "      <td>0.027000</td>\n",
       "      <td>3.440500</td>\n",
       "      <td>-5.343500</td>\n",
       "      <td>0.011000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>22.670474</td>\n",
       "      <td>16.596490</td>\n",
       "      <td>0.11113</td>\n",
       "      <td>0.162124</td>\n",
       "      <td>0.972591</td>\n",
       "      <td>1.570108</td>\n",
       "      <td>0.104329</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>50.134100</td>\n",
       "      <td>-326.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>-7.500000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>89.501675</td>\n",
       "      <td>-308.930400</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>-6.500000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>105.236100</td>\n",
       "      <td>-297.825600</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>-5.500000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>116.023175</td>\n",
       "      <td>-288.169025</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>-4.500000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>150.134100</td>\n",
       "      <td>-226.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>-1.500000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                x1           x2          x5           x6           x7  \\\n",
       "count  2000.000000  2000.000000  2000.00000  2000.000000  2000.000000   \n",
       "mean    101.074885  -298.282145     0.01250     0.027000     3.440500   \n",
       "std      22.670474    16.596490     0.11113     0.162124     0.972591   \n",
       "min      50.134100  -326.000000     0.00000     0.000000     3.000000   \n",
       "25%      89.501675  -308.930400     0.00000     0.000000     3.000000   \n",
       "50%     105.236100  -297.825600     0.00000     0.000000     3.000000   \n",
       "75%     116.023175  -288.169025     0.00000     0.000000     3.000000   \n",
       "max     150.134100  -226.000000     1.00000     1.000000     8.000000   \n",
       "\n",
       "                x8       target  \n",
       "count  2000.000000  2000.000000  \n",
       "mean     -5.343500     0.011000  \n",
       "std       1.570108     0.104329  \n",
       "min      -7.500000     0.000000  \n",
       "25%      -6.500000     0.000000  \n",
       "50%      -5.500000     0.000000  \n",
       "75%      -4.500000     0.000000  \n",
       "max      -1.500000     1.000000  "
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "208dfaa5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['SAT' 'LCV' 'XJB' 'QKP']\n"
     ]
    }
   ],
   "source": [
    "print(data[\"x3\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "07614219",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['MZBER' 'PQKE' 'YEQA' 'ZUQF']\n"
     ]
    }
   ],
   "source": [
    "print(data[\"x4\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "7a5fb521",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]\n",
    "y = data['target']\n",
    "obj_cols = X.select_dtypes(include=[\"object\"]).columns.tolist()\n",
    "preprocessor = ColumnTransformer(transformers=[(\"onehot\", OneHotEncoder(), obj_cols)], remainder=\"passthrough\")\n",
    "X_encoded = preprocessor.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "c3e3b0c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2000, 14)"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_encoded.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "244d1c31",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              x1        x2        x5        x6        x7        x8    target\n",
      "x1      1.000000  0.473990  0.030226 -0.000131 -0.244156  0.111917  0.030486\n",
      "x2      0.473990  1.000000  0.007558 -0.032315 -0.265814  0.171948  0.019244\n",
      "x5      0.030226  0.007558  1.000000  0.036790 -0.009315  0.060458  0.333312\n",
      "x6     -0.000131 -0.032315  0.036790  1.000000  0.013366  0.008940  0.189463\n",
      "x7     -0.244156 -0.265814 -0.009315  0.013366  1.000000 -0.131977  0.011384\n",
      "x8      0.111917  0.171948  0.060458  0.008940 -0.131977  1.000000  0.023078\n",
      "target  0.030486  0.019244  0.333312  0.189463  0.011384  0.023078  1.000000\n"
     ]
    }
   ],
   "source": [
    "matriz_correlacion = data.corr()\n",
    "print(matriz_correlacion)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "ada8b0cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify = y, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "2c637237",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GradientBoostingClassifier()"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "gb = GradientBoostingClassifier()\n",
    "gb.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "16397fa7",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_pred = gb.predict(X_train)\n",
    "y_test_pred = gb.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "3f1a3e9a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC train: 1.0000\n",
      "AUC test: 0.4987\n",
      "F-1 score (test): 0.0000\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import roc_auc_score, f1_score\n",
    "\n",
    "f1_test = f1_score(y_test, y_test_pred)\n",
    "auc_train = roc_auc_score(y_train, y_train_pred)\n",
    "auc_test = roc_auc_score(y_test, y_test_pred)\n",
    "print(\"AUC train: %.4f\" % auc_train)\n",
    "print(\"AUC test: %.4f\" % auc_test)\n",
    "print(\"F-1 score (test): %.4f\" % f1_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d97030ce",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f9d1585",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
