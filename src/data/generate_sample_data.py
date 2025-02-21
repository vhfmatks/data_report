import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=20000):
    """샘플 데이터 생성"""
    
    # 시드 설정
    np.random.seed(42)
    random.seed(42)
    
    # 기준 날짜 범위 설정 (2023년 전체)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    # 코드 매핑 정의
    sido_codes = ['11', '26', '27', '28', '29', '30', '31', '36', '41', '42', '43', '44', '45', '46', '47', '48', '50']
    ccg_codes = [f"{i:02d}" for i in range(1, 26)]  # 01~25
    adng_codes = [f"{random.randint(1, 99999999):08d}" for _ in range(50)]  # 8자리 코드
    
    # 업종 분류
    buz_lcls = ['요식업', '소매업', '서비스업', '교육업', '의료업']
    buz_mcls = {
        '요식업': ['한식', '중식', '일식', '양식', '카페'],
        '소매업': ['편의점', '마트', '의류', '잡화', '가전'],
        '서비스업': ['미용', '세탁', '숙박', '레저', '운송'],
        '교육업': ['학원', '교습소', '직업학교', '어학원', '예체능'],
        '의료업': ['병원', '의원', '약국', '한의원', '치과']
    }
    buz_scls = {
        '한식': ['한정식', '분식', '국밥', '고기집', '해장국'],
        '중식': ['중화요리', '마라탕', '양꼬치', '딤섬', '짬뽕'],
        '일식': ['스시', '라멘', '돈까스', '우동', '이자카야'],
        '양식': ['피자', '파스타', '스테이크', '햄버거', '브런치'],
        '카페': ['커피전문점', '디저트카페', '베이커리', '주스바', '티하우스']
        # 다른 중분류에 대한 소분류는 생략
    }
    
    # 데이터 생성
    data = {
        'STRD_YYMM': [],
        'STRD_DATE': [],
        'WK_CD': [],
        'TIMC_CD': [],
        'MER_SIDO_CD': [],
        'MER_CCG_CD': [],
        'MER_ADNG_CD': [],
        'BUZ_LCLS_NM': [],
        'BUZ_MCLS_NM': [],
        'BUZ_SCLS_NM': [],
        'CST_SIDO_CD': [],
        'CST_CCG_CD': [],
        'CST_ADNG_CD': [],
        'SEX_CD': [],
        'AGE_CD': [],
        'HOSH_TYP_CD': [],
        'INCM_NR_CD': [],
        'AMT': [],
        'CNT': []
    }
    
    for _ in range(n_samples):
        # 날짜 관련
        date = random.choice(dates)
        weekday = date.weekday()
        
        # 기본 정보 생성
        buz_lcls = random.choice(['요식업', '소매업', '서비스업', '교육업', '의료업'])
        buz_mcls = random.choice(['한식', '중식', '일식', '양식', '카페'] if buz_lcls == '요식업' else ['기타'])
        buz_scls = random.choice(['한정식', '분식', '국밥', '고기집', '해장국'] if buz_mcls == '한식' else ['기타'])
        
        # 데이터 추가
        data['STRD_YYMM'].append(int(date.strftime('%Y%m')))  # YYYYMM 형식의 정수
        data['STRD_DATE'].append(int(date.strftime('%Y%m%d')))  # YYYYMMDD 형식의 정수
        data['WK_CD'].append(2 if weekday >= 5 else 1)  # 주말(5,6) = 2, 평일 = 1
        data['TIMC_CD'].append(random.choices([1,2,3,4], weights=[0.2,0.35,0.35,0.1])[0])
        data['MER_SIDO_CD'].append(random.choice(sido_codes))
        data['MER_CCG_CD'].append(random.choice(ccg_codes))
        data['MER_ADNG_CD'].append(random.choice(adng_codes))
        data['BUZ_LCLS_NM'].append(buz_lcls)
        data['BUZ_MCLS_NM'].append(buz_mcls)
        data['BUZ_SCLS_NM'].append(buz_scls)
        data['CST_SIDO_CD'].append(random.choice(sido_codes))
        data['CST_CCG_CD'].append(random.choice(ccg_codes))
        data['CST_ADNG_CD'].append(random.choice(adng_codes))
        data['SEX_CD'].append(random.choices([1,2], weights=[0.48,0.52])[0])
        data['AGE_CD'].append(random.choices([20,30,40,50,60], weights=[0.2,0.25,0.25,0.2,0.1])[0])
        data['HOSH_TYP_CD'].append(random.choices([1,2,3,4,5,9], weights=[0.3,0.2,0.2,0.15,0.1,0.05])[0])
        data['INCM_NR_CD'].append(random.choices([1,2,3,4,5,9], weights=[0.1,0.2,0.3,0.25,0.1,0.05])[0])
        
        # 매출 금액과 건수 생성 (업종별로 다른 분포 사용)
        if buz_lcls == '요식업':
            amt = int(np.random.normal(30000, 10000))
        elif buz_lcls == '소매업':
            amt = int(np.random.normal(50000, 20000))
        elif buz_lcls == '서비스업':
            amt = int(np.random.normal(100000, 30000))
        elif buz_lcls == '교육업':
            amt = int(np.random.normal(200000, 50000))
        else:  # 의료업
            amt = int(np.random.normal(150000, 40000))
        
        amt = max(1000, amt)  # 최소 금액 설정
        amt = (amt // 100) * 100  # 백원 단위 라운딩
        
        data['AMT'].append(amt * 2)  # 원 금액 * 2
        data['CNT'].append(random.choices([1,2,3], weights=[0.7,0.2,0.1])[0] * 2)  # 원 건수 * 2
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    # 데이터 생성
    df = generate_sample_data(20000)
    
    # CSV 파일로 저장
    df.to_csv('data.csv', index=False)
    print("샘플 데이터가 성공적으로 생성되었습니다.") 