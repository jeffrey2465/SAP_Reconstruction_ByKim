# Salt And Pepper Noise 제거 알고리즘 구현

2021.02.22

### 주요 결과
  * MF: opencv api   
  * AMF: NAMF의 Step1에 해당하는 내용 구현 (논문 구현 Code 파이썬 포팅)   
  * NAMF: NAMF 논문 Step1 + Step2 (논문 구현 Code 파이썬 포팅)   
  * Proposed1 : AMF+거리 기반 weighted sum    
  * Proposed2 : Proposed1 + bilateral   
  * Proposed3 : namf + 거리 기반 weighted sum + bilateral   
  * Proposed4 : proposed3 + 자신 weight 포함   
  * Proposed5 : Proposed1 + pixel 신뢰도 weight average   
