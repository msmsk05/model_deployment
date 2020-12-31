{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xdOyfC8jASCz"
   },
   "source": [
    "---\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "934v5Lm0ASCz"
   },
   "source": [
    "---\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "0ThMJSTCASCz"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "from sqlalchemy import create_engine\n",
    "import warnings\n",
    "from IPython.core.pylabtools import figsize\n",
    "from scipy.stats import zscore\n",
    "from scipy import stats\n",
    "from numpy import percentile\n",
    "font_title = {'family': 'times new roman', 'color': 'darkred',\n",
    "              'weight': 'bold', 'size': 14}\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_style(\"whitegrid\")\n",
    "plt.rcParams['figure.dpi'] = 100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OUxd8117ASCz"
   },
   "source": [
    "#### *ii. Load Dataset*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "nO4IbFeNASC0"
   },
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
       "      <th>Elevation</th>\n",
       "      <th>Aspect</th>\n",
       "      <th>Slope</th>\n",
       "      <th>Horizontal_Distance_To_Hydrology</th>\n",
       "      <th>Vertical_Distance_To_Hydrology</th>\n",
       "      <th>Horizontal_Distance_To_Roadways</th>\n",
       "      <th>Hillshade_9am</th>\n",
       "      <th>Hillshade_Noon</th>\n",
       "      <th>Hillshade_3pm</th>\n",
       "      <th>Horizontal_Distance_To_Fire_Points</th>\n",
       "      <th>Wilderness_Area1</th>\n",
       "      <th>Wilderness_Area2</th>\n",
       "      <th>Wilderness_Area3</th>\n",
       "      <th>Wilderness_Area4</th>\n",
       "      <th>Soil_Type1</th>\n",
       "      <th>Soil_Type2</th>\n",
       "      <th>Soil_Type3</th>\n",
       "      <th>Soil_Type4</th>\n",
       "      <th>Soil_Type5</th>\n",
       "      <th>Soil_Type6</th>\n",
       "      <th>Soil_Type7</th>\n",
       "      <th>Soil_Type8</th>\n",
       "      <th>Soil_Type9</th>\n",
       "      <th>Soil_Type10</th>\n",
       "      <th>Soil_Type11</th>\n",
       "      <th>Soil_Type12</th>\n",
       "      <th>Soil_Type13</th>\n",
       "      <th>Soil_Type14</th>\n",
       "      <th>Soil_Type15</th>\n",
       "      <th>Soil_Type16</th>\n",
       "      <th>Soil_Type17</th>\n",
       "      <th>Soil_Type18</th>\n",
       "      <th>Soil_Type19</th>\n",
       "      <th>Soil_Type20</th>\n",
       "      <th>Soil_Type21</th>\n",
       "      <th>Soil_Type22</th>\n",
       "      <th>Soil_Type23</th>\n",
       "      <th>Soil_Type24</th>\n",
       "      <th>Soil_Type25</th>\n",
       "      <th>Soil_Type26</th>\n",
       "      <th>Soil_Type27</th>\n",
       "      <th>Soil_Type28</th>\n",
       "      <th>Soil_Type29</th>\n",
       "      <th>Soil_Type30</th>\n",
       "      <th>Soil_Type31</th>\n",
       "      <th>Soil_Type32</th>\n",
       "      <th>Soil_Type33</th>\n",
       "      <th>Soil_Type34</th>\n",
       "      <th>Soil_Type35</th>\n",
       "      <th>Soil_Type36</th>\n",
       "      <th>Soil_Type37</th>\n",
       "      <th>Soil_Type38</th>\n",
       "      <th>Soil_Type39</th>\n",
       "      <th>Soil_Type40</th>\n",
       "      <th>Cover_Type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2596</td>\n",
       "      <td>51</td>\n",
       "      <td>3</td>\n",
       "      <td>258</td>\n",
       "      <td>0</td>\n",
       "      <td>510</td>\n",
       "      <td>221</td>\n",
       "      <td>232</td>\n",
       "      <td>148</td>\n",
       "      <td>6279</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2590</td>\n",
       "      <td>56</td>\n",
       "      <td>2</td>\n",
       "      <td>212</td>\n",
       "      <td>-6</td>\n",
       "      <td>390</td>\n",
       "      <td>220</td>\n",
       "      <td>235</td>\n",
       "      <td>151</td>\n",
       "      <td>6225</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2804</td>\n",
       "      <td>139</td>\n",
       "      <td>9</td>\n",
       "      <td>268</td>\n",
       "      <td>65</td>\n",
       "      <td>3180</td>\n",
       "      <td>234</td>\n",
       "      <td>238</td>\n",
       "      <td>135</td>\n",
       "      <td>6121</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2785</td>\n",
       "      <td>155</td>\n",
       "      <td>18</td>\n",
       "      <td>242</td>\n",
       "      <td>118</td>\n",
       "      <td>3090</td>\n",
       "      <td>238</td>\n",
       "      <td>238</td>\n",
       "      <td>122</td>\n",
       "      <td>6211</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2595</td>\n",
       "      <td>45</td>\n",
       "      <td>2</td>\n",
       "      <td>153</td>\n",
       "      <td>-1</td>\n",
       "      <td>391</td>\n",
       "      <td>220</td>\n",
       "      <td>234</td>\n",
       "      <td>150</td>\n",
       "      <td>6172</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Elevation  Aspect  Slope  Horizontal_Distance_To_Hydrology  \\\n",
       "0       2596      51      3                               258   \n",
       "1       2590      56      2                               212   \n",
       "2       2804     139      9                               268   \n",
       "3       2785     155     18                               242   \n",
       "4       2595      45      2                               153   \n",
       "\n",
       "   Vertical_Distance_To_Hydrology  Horizontal_Distance_To_Roadways  \\\n",
       "0                               0                              510   \n",
       "1                              -6                              390   \n",
       "2                              65                             3180   \n",
       "3                             118                             3090   \n",
       "4                              -1                              391   \n",
       "\n",
       "   Hillshade_9am  Hillshade_Noon  Hillshade_3pm  \\\n",
       "0            221             232            148   \n",
       "1            220             235            151   \n",
       "2            234             238            135   \n",
       "3            238             238            122   \n",
       "4            220             234            150   \n",
       "\n",
       "   Horizontal_Distance_To_Fire_Points  Wilderness_Area1  Wilderness_Area2  \\\n",
       "0                                6279                 1                 0   \n",
       "1                                6225                 1                 0   \n",
       "2                                6121                 1                 0   \n",
       "3                                6211                 1                 0   \n",
       "4                                6172                 1                 0   \n",
       "\n",
       "   Wilderness_Area3  Wilderness_Area4  Soil_Type1  Soil_Type2  Soil_Type3  \\\n",
       "0                 0                 0           0           0           0   \n",
       "1                 0                 0           0           0           0   \n",
       "2                 0                 0           0           0           0   \n",
       "3                 0                 0           0           0           0   \n",
       "4                 0                 0           0           0           0   \n",
       "\n",
       "   Soil_Type4  Soil_Type5  Soil_Type6  Soil_Type7  Soil_Type8  Soil_Type9  \\\n",
       "0           0           0           0           0           0           0   \n",
       "1           0           0           0           0           0           0   \n",
       "2           0           0           0           0           0           0   \n",
       "3           0           0           0           0           0           0   \n",
       "4           0           0           0           0           0           0   \n",
       "\n",
       "   Soil_Type10  Soil_Type11  Soil_Type12  Soil_Type13  Soil_Type14  \\\n",
       "0            0            0            0            0            0   \n",
       "1            0            0            0            0            0   \n",
       "2            0            0            1            0            0   \n",
       "3            0            0            0            0            0   \n",
       "4            0            0            0            0            0   \n",
       "\n",
       "   Soil_Type15  Soil_Type16  Soil_Type17  Soil_Type18  Soil_Type19  \\\n",
       "0            0            0            0            0            0   \n",
       "1            0            0            0            0            0   \n",
       "2            0            0            0            0            0   \n",
       "3            0            0            0            0            0   \n",
       "4            0            0            0            0            0   \n",
       "\n",
       "   Soil_Type20  Soil_Type21  Soil_Type22  Soil_Type23  Soil_Type24  \\\n",
       "0            0            0            0            0            0   \n",
       "1            0            0            0            0            0   \n",
       "2            0            0            0            0            0   \n",
       "3            0            0            0            0            0   \n",
       "4            0            0            0            0            0   \n",
       "\n",
       "   Soil_Type25  Soil_Type26  Soil_Type27  Soil_Type28  Soil_Type29  \\\n",
       "0            0            0            0            0            1   \n",
       "1            0            0            0            0            1   \n",
       "2            0            0            0            0            0   \n",
       "3            0            0            0            0            0   \n",
       "4            0            0            0            0            1   \n",
       "\n",
       "   Soil_Type30  Soil_Type31  Soil_Type32  Soil_Type33  Soil_Type34  \\\n",
       "0            0            0            0            0            0   \n",
       "1            0            0            0            0            0   \n",
       "2            0            0            0            0            0   \n",
       "3            1            0            0            0            0   \n",
       "4            0            0            0            0            0   \n",
       "\n",
       "   Soil_Type35  Soil_Type36  Soil_Type37  Soil_Type38  Soil_Type39  \\\n",
       "0            0            0            0            0            0   \n",
       "1            0            0            0            0            0   \n",
       "2            0            0            0            0            0   \n",
       "3            0            0            0            0            0   \n",
       "4            0            0            0            0            0   \n",
       "\n",
       "   Soil_Type40  Cover_Type  \n",
       "0            0           5  \n",
       "1            0           5  \n",
       "2            0           2  \n",
       "3            0           2  \n",
       "4            0           5  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.set_option('display.max_columns', 500)\n",
    "df=pd.read_csv(\"covtype.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[df[\"Cover_Type\"] == 1] = df[df[\"Cover_Type\"] == 1].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 2] = df[df[\"Cover_Type\"] == 2].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 3] = df[df[\"Cover_Type\"] == 3].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 4] = df[df[\"Cover_Type\"] == 4].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 5] = df[df[\"Cover_Type\"] == 5].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 6] = df[df[\"Cover_Type\"] == 6].sample(1100)\n",
    "df[df[\"Cover_Type\"] == 7] = df[df[\"Cover_Type\"] == 7].sample(1100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7700, 55)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=df[[\"Elevation\",\"Horizontal_Distance_To_Hydrology\",\"Vertical_Distance_To_Hydrology\",\"Horizontal_Distance_To_Roadways\",\"Horizontal_Distance_To_Fire_Points\"]]\n",
    "y=df.Cover_Type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fQeP_Dhtri3v"
   },
   "source": [
    "***\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kdwJqW45ASC0"
   },
   "source": [
    "## 2.  Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "h95eEjnEASC0"
   },
   "source": [
    "### Detect Missing Values and Outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TNvbnTdnASC0"
   },
   "source": [
    "#### *i. Missing Value Detection*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.metrics import classification_report, f1_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WjDFWFcgASC0"
   },
   "source": [
    "***\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "45zRLFdDASC0"
   },
   "source": [
    "***\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ylwp86BnIbfD"
   },
   "source": [
    "## 4. Prediction (Multi-class Classification)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5bKNukyEASC0"
   },
   "source": [
    "### Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "44t5jjukASC0"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=101)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         1.0       0.66      0.60      0.63       203\n",
      "         2.0       0.69      0.49      0.58       219\n",
      "         3.0       0.58      0.54      0.56       228\n",
      "         4.0       0.76      0.94      0.84       218\n",
      "         5.0       0.70      0.91      0.79       223\n",
      "         6.0       0.63      0.54      0.58       210\n",
      "         7.0       0.86      0.89      0.87       239\n",
      "\n",
      "    accuracy                           0.71      1540\n",
      "   macro avg       0.70      0.70      0.69      1540\n",
      "weighted avg       0.70      0.71      0.70      1540\n",
      "\n"
     ]
    }
   ],
   "source": [
    "rf=RandomForestClassifier(max_depth = 7,             \n",
    "                                  max_features = 4, \n",
    "                                  min_samples_split =8, \n",
    "                                  n_estimators = 300).fit(X_train, y_train)\n",
    "y_pred=rf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "pickle.dump(rf, open(\"model.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=pickle.load(open(\"model.pkl\", \"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n"
     ]
    }
   ],
   "source": [
    "print(int(model.predict([[2788, 2, 3555,221, 2984]])[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Tree_Types_Prediction_Student_V1.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
