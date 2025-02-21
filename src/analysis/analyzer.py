"""
데이터 분석 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.data.loader import convert_to_serializable

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, schema: Dict):
        """
        데이터 분석기 초기화
        
        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            schema (Dict): 데이터 스키마 정보
        """
        self.df = df
        self.schema = schema
        self.results = {}
    
    def _validate_column(self, col: str) -> bool:
        """컬럼 유효성 검사"""
        return col in self.df.columns
    
    def _get_clean_series(self, col: str) -> pd.Series:
        """결측치가 제거된 시리즈 반환"""
        return self.df[col].dropna()
    
    def analyze_numeric(self, col: str) -> Dict[str, Any]:
        """
        수치형 데이터 분석
        
        Args:
            col (str): 분석할 컬럼명
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            if not self._validate_column(col):
                raise ValueError(f"컬럼 '{col}'이 데이터프레임에 존재하지 않습니다.")
            
            series = self._get_clean_series(col)
            if series.empty:
                return self._get_empty_numeric_result()
            
            # 기본 통계량 계산
            stats = series.describe()
            
            # 이상치 분석
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # 분포 특성 분석
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            return {
                "기본통계": {
                    "개수": int(stats['count']),
                    "평균": float(stats['mean']),
                    "표준편차": float(stats['std']),
                    "최소값": float(stats['min']),
                    "25%": float(stats['25%']),
                    "중앙값": float(stats['50%']),
                    "75%": float(stats['75%']),
                    "최대값": float(stats['max'])
                },
                "분포특성": {
                    "왜도": float(skewness),
                    "첨도": float(kurtosis)
                },
                "이상치": {
                    "개수": len(outliers),
                    "비율": len(outliers) / len(series) * 100,
                    "범위": {
                        "하한": float(lower_bound),
                        "상한": float(upper_bound)
                    }
                }
            }
        except Exception as e:
            print(f"수치형 데이터 분석 중 오류 발생 ({col}): {str(e)}")
            return self._get_empty_numeric_result()
    
    def analyze_categorical(self, col: str) -> Dict[str, Any]:
        """
        범주형 데이터 분석
        
        Args:
            col (str): 분석할 컬럼명
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            if not self._validate_column(col):
                raise ValueError(f"컬럼 '{col}'이 데이터프레임에 존재하지 않습니다.")
            
            series = self._get_clean_series(col)
            if series.empty:
                return self._get_empty_categorical_result()
            
            value_counts = series.value_counts()
            value_ratios = (value_counts / len(series) * 100).round(2)
            
            return {
                "고유값": {
                    "개수": len(value_counts),
                    "분포": {
                        str(k): {
                            "빈도": int(v),
                            "비율": float(value_ratios[k])
                        } for k, v in value_counts.head(10).items()
                    }
                },
                "최빈값": {
                    "값": str(value_counts.index[0]),
                    "빈도": int(value_counts.iloc[0]),
                    "비율": float(value_ratios.iloc[0])
                },
                "결측치": {
                    "개수": int(series.isna().sum()),
                    "비율": float(series.isna().mean() * 100)
                }
            }
        except Exception as e:
            print(f"범주형 데이터 분석 중 오류 발생 ({col}): {str(e)}")
            return self._get_empty_categorical_result()
    
    def analyze_datetime(self, col: str) -> Dict[str, Any]:
        """
        시계열 데이터 분석
        
        Args:
            col (str): 분석할 컬럼명
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            if not self._validate_column(col):
                raise ValueError(f"컬럼 '{col}'이 데이터프레임에 존재하지 않습니다.")
            
            # datetime 형식으로 변환
            date_series = pd.to_datetime(self.df[col], errors='coerce')
            clean_series = date_series.dropna()
            
            if clean_series.empty:
                return self._get_empty_datetime_result()
            
            # 기간 분석
            start_date = clean_series.min()
            end_date = clean_series.max()
            date_diff = end_date - start_date
            
            # 시간 단위별 분포
            year_dist = clean_series.dt.year.value_counts().sort_index()
            month_dist = clean_series.dt.month.value_counts().sort_index()
            weekday_dist = clean_series.dt.dayofweek.value_counts().sort_index()
            
            return {
                "기간": {
                    "시작": start_date.strftime("%Y-%m-%d"),
                    "종료": end_date.strftime("%Y-%m-%d"),
                    "기간": {
                        "일수": int(date_diff.days),
                        "연도수": len(year_dist),
                        "월수": len(month_dist)
                    }
                },
                "분포": {
                    "연도별": {str(k): int(v) for k, v in year_dist.items()},
                    "월별": {str(k): int(v) for k, v in month_dist.items()},
                    "요일별": {str(k): int(v) for k, v in weekday_dist.items()}
                },
                "결측치": {
                    "개수": int(date_series.isna().sum()),
                    "비율": float(date_series.isna().mean() * 100)
                }
            }
        except Exception as e:
            print(f"시계열 데이터 분석 중 오류 발생 ({col}): {str(e)}")
            return self._get_empty_datetime_result()
    
    def analyze_correlations(self, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """
        수치형 변수 간 상관관계 분석
        
        Args:
            numeric_cols (List[str]): 수치형 컬럼 목록
            
        Returns:
            List[Dict[str, Any]]: 상관관계 분석 결과
        """
        try:
            if len(numeric_cols) < 2:
                return []
            
            # 유효한 컬럼만 선택
            valid_cols = [col for col in numeric_cols if self._validate_column(col)]
            if len(valid_cols) < 2:
                return []
            
            # 결측치 제거
            clean_df = self.df[valid_cols].dropna()
            if clean_df.empty:
                return []
            
            # 상관관계 계산
            correlations = clean_df.corr().round(3)
            high_correlations = []
            
            for i in range(len(valid_cols)):
                for j in range(i+1, len(valid_cols)):
                    corr = correlations.iloc[i, j]
                    if abs(corr) > 0.5:  # 유의미한 상관관계 기준
                        high_correlations.append({
                            "변수1": valid_cols[i],
                            "변수2": valid_cols[j],
                            "상관계수": float(corr),
                            "강도": self._get_correlation_strength(corr)
                        })
            
            return sorted(high_correlations, key=lambda x: abs(x["상관계수"]), reverse=True)
        except Exception as e:
            print(f"상관관계 분석 중 오류 발생: {str(e)}")
            return []
    
    def _get_correlation_strength(self, corr: float) -> str:
        """상관관계 강도 판단"""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "강함"
        elif abs_corr > 0.5:
            return "중간"
        else:
            return "약함"
    
    def _get_empty_numeric_result(self) -> Dict[str, Any]:
        """빈 수치형 분석 결과"""
        return {
            "기본통계": {k: 0 for k in ["개수", "평균", "표준편차", "최소값", "25%", "중앙값", "75%", "최대값"]},
            "분포특성": {"왜도": 0, "첨도": 0},
            "이상치": {"개수": 0, "비율": 0, "범위": {"하한": 0, "상한": 0}}
        }
    
    def _get_empty_categorical_result(self) -> Dict[str, Any]:
        """빈 범주형 분석 결과"""
        return {
            "고유값": {"개수": 0, "분포": {}},
            "최빈값": {"값": None, "빈도": 0, "비율": 0},
            "결측치": {"개수": 0, "비율": 0}
        }
    
    def _get_empty_datetime_result(self) -> Dict[str, Any]:
        """빈 시계열 분석 결과"""
        return {
            "기간": {
                "시작": None,
                "종료": None,
                "기간": {"일수": 0, "연도수": 0, "월수": 0}
            },
            "분포": {"연도별": {}, "월별": {}, "요일별": {}},
            "결측치": {"개수": 0, "비율": 0}
        }
    
    def analyze(self) -> Dict[str, Any]:
        """
        전체 데이터 분석 수행
        
        Returns:
            Dict[str, Any]: 전체 분석 결과
        """
        try:
            results = {
                "메타정보": {
                    "전체행수": len(self.df),
                    "전체컬럼수": len(self.df.columns),
                    "분석시작": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # 컬럼별 분석
            for col, info in self.schema.items():
                if not self._validate_column(col):
                    continue
                    
                if info["data_type"] == "numeric":
                    results[col] = self.analyze_numeric(col)
                elif info["data_type"] == "categorical":
                    results[col] = self.analyze_categorical(col)
                elif info["data_type"] == "datetime":
                    results[col] = self.analyze_datetime(col)
            
            # 상관관계 분석
            numeric_cols = [col for col, info in self.schema.items() 
                          if info["data_type"] == "numeric" and self._validate_column(col)]
            results["상관관계"] = self.analyze_correlations(numeric_cols)
            
            results["메타정보"]["분석종료"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return results
            
        except Exception as e:
            print(f"전체 데이터 분석 중 오류 발생: {str(e)}")
            return {"메타정보": {"오류": str(e)}}

def analyze_data(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """
    데이터 분석 수행
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        schema (Dict): 데이터 스키마 정보
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    analyzer = DataAnalyzer(df, schema)
    return analyzer.analyze()

def analyze_time_series(df, config):
    """시계열 분석을 수행합니다.
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        config (dict): 시계열 분석 설정
        
    Returns:
        dict: 시계열 분석 결과
    """
    results = {}
    
    try:
        # 시계열 데이터 준비
        date_col = config.get("date_column")
        value_col = config.get("value_column")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # 시계열 분해
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(df[value_col], period=config.get("period", 12))
        results["decomposition"] = {
            "trend": decomposition.trend.tolist(),
            "seasonal": decomposition.seasonal.tolist(),
            "resid": decomposition.resid.tolist()
        }
        
        # 추세 분석
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(df)), df[value_col]
        )
        results["trend_analysis"] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value
        }
        
        # 계절성 검정
        if config.get("check_seasonality", True):
            from statsmodels.stats.diagnostic import acf
            acf_values = acf(df[value_col], nlags=config.get("seasonal_lags", 24))
            results["seasonality"] = {
                "acf_values": acf_values.tolist(),
                "significant_lags": [i for i, v in enumerate(acf_values) if abs(v) > 0.2]
            }
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def analyze_clusters(df, config):
    """군집 분석을 수행합니다.
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        config (dict): 군집 분석 설정
        
    Returns:
        dict: 군집 분석 결과
    """
    results = {}
    
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # 데이터 준비
        features = config.get("features", [])
        X = df[features]
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 최적의 군집 수 찾기
        if config.get("find_optimal_k", True):
            from sklearn.metrics import silhouette_score
            silhouette_scores = []
            K = range(2, min(11, len(df)))
            
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            optimal_k = K[np.argmax(silhouette_scores)]
            results["optimal_k"] = optimal_k
        else:
            optimal_k = config.get("n_clusters", 3)
        
        # 군집 분석 수행
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 결과 저장
        results["labels"] = cluster_labels.tolist()
        results["centroids"] = kmeans.cluster_centers_.tolist()
        
        # 군집별 특성 분석
        cluster_stats = []
        for i in range(optimal_k):
            cluster_data = df[cluster_labels == i]
            stats = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df) * 100,
                "features": {}
            }
            for feature in features:
                stats["features"][feature] = {
                    "mean": cluster_data[feature].mean(),
                    "std": cluster_data[feature].std(),
                    "min": cluster_data[feature].min(),
                    "max": cluster_data[feature].max()
                }
            cluster_stats.append(stats)
        results["cluster_stats"] = cluster_stats
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def create_prediction_model(df, config):
    """예측 모델을 생성하고 평가합니다.
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        config (dict): 예측 모델 설정
        
    Returns:
        dict: 예측 모델 결과
    """
    results = {}
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 데이터 준비
        features = config.get("features", [])
        target = config.get("target")
        
        X = df[features]
        y = df[target]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 학습
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 특성 중요도
        feature_importance = dict(zip(features, model.feature_importances_))
        
        # 결과 저장
        results["metrics"] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2
        }
        results["feature_importance"] = feature_importance
        
        # 예측값 저장
        results["predictions"] = {
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist()
        }
    
    except Exception as e:
        results["error"] = str(e)
    
    return results 