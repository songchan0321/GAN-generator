# :zap:프로젝트 개요


사용자의 손글씨(템플릿)를 제출하면 머신러닝을 통해 사용자의 필체와 유사한 폰트를 구현해내는 프로그램을 구현하였습니다.

# :alarm_clock: 개발 기간


2022/03/15 ~ 2022/11/26

# :full_moon_with_face: 멤버 구성


송인찬 : 분석 설계 및 개발 담당

김환엽 : 분석 설계 및 콘텐츠 담당

# :computer: 개발 환경


python 3.3


# :bell: 주요 기능


정해진 양식의 템플릿을 스캔하여 prerocessed폴더에 넣고 generate_handwrite.py를 실행시킵니다.

Generated_handwrites폴더에 템플릿이 입력되는데, 이후 parse_handwrite.py를 실행시키면 템플릿이 글자 1개 단위로 쪼개져서 papers폴더에 저장됩니다.

KoreanGAN.py를 실행시키면 쪼개진 글자 템플릿이 머신러닝되어 epoch 될 수록 정교한 글자가 탄생합니다.

epoch 횟수는 설정할 수 있지만 아직 불안정한 모델이기에 무작정 epoch 수를 늘린다고 정교한 글자가 탄생되지는 않습니다.

